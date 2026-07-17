# -*- coding: utf-8 -*-
"""Fetch a Finya profile by shared-link id, render it to an LLM-friendly
profile.xml, and download all of its pictures.

Usage:
    python finya_fetch.py <id>

Steps (mirrors the request):
  1. Take an <id> argument.
  2. GET https://www.finya.de/api/link/convert/?link=<id>   -> { "link": <link> }
  3. GET https://www.finya.de/api/v1/encounters/<ENCOUNTER>/users/<link>
        -> the person's profile JSON (same shape as BU/finya/profile.json)
  4. Convert the profile JSON to a structured profile.xml (same information as
     BU/finya/profile.js would render, but as XML), stored in
     profiles/finya_<nickname>/profile.xml
  5. Instead of a gallery section, download every picture into
     profiles/finya_<nickname>/pictures/

The Cookie used for the authenticated requests is read from the COOKIE entry
in the project's .env file.
"""

import os
import re
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

import finya_labels as L

BASE = "https://www.finya.de"
# The encounter id is part of the authenticated session (also seen embedded in
# the URLs inside profile.json). Override via the ENCOUNTER_ID env var if needed.
DEFAULT_ENCOUNTER = "o5geOOjO38An"

PROJECT_ROOT = Path(__file__).resolve().parent
PROFILES_DIR = PROJECT_ROOT / "profiles"


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def get_cookie() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    cookie = os.getenv("COOKIE")
    if not cookie:
        sys.exit(
            "ERROR: No COOKIE found in .env. Add a line like:\n"
            "  COOKIE=<your finya.de cookie header value>"
        )

    return cookie


def convert_link(session: requests.Session, link_id: str) -> str:
    """Step 2: resolve the shared-link id into the internal user link."""
    url = f"{BASE}/api/link/convert/"
    resp = session.get(url, params={"link": link_id})
    resp.raise_for_status()
    data = resp.json()
    link = data.get("link")
    if not link:
        sys.exit(f"ERROR: convert response had no 'link': {data!r}")
    return link


def fetch_profile(session: requests.Session, encounter: str, link: str) -> dict:
    """Step 3: fetch the full profile JSON for the resolved link."""
    url = f"{BASE}/api/v1/encounters/{encounter}/users/{link}"
    resp = session.get(
        url,
        headers={
            "time-zone": "UTC",
            "accept": "application/vnd.fyappbff.v1+json",
        },
    )
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# profile.json -> profile.xml  (same information as BU/finya/profile.js renders)
# --------------------------------------------------------------------------- #
def map_habit_ids(category_name, ids):
    d = L.HABITS.get(category_name, {})
    labels = []
    for o in ids or []:
        label = d.get(o.get("id"))
        if label:
            labels.append(label)
    return labels


def map_qna(qnas):
    res = []
    for q in qnas or []:
        key = f"question_{q.get('id')}"
        question = L.QNA.get(key, f"Frage {q.get('id')}")
        answer = q.get("answer") or ""
        res.append((question, answer))
    return res


def _el(parent, tag, text=None, **attrs):
    """Create a SubElement, optionally with text and attributes."""
    e = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        e.set(k, str(v))
    if text is not None:
        e.text = str(text)
    return e


def _list_el(parent, tag, values, item_tag="item"):
    """Create <tag><item>..</item>..</tag> only when there are values."""
    values = [v for v in (values or []) if v]
    if not values:
        return None
    container = ET.SubElement(parent, tag)
    for v in values:
        _el(container, item_tag, v)
    return container


def build_profile_xml(profile: dict) -> str:
    """Render the profile as structured XML.

    Design goals for LLM consumption: one concept per element, semantic English
    tag names, human-readable German values, empty sections omitted entirely so
    the model never has to reason about blank fields. The German source label
    for each section is kept in a `de` attribute for grounding.
    """
    p = profile.get("payload") or {}
    lifestyle = p.get("lifestyle") or {}
    education = p.get("education") or {}
    sentiment = p.get("sentiment") or {}

    root = ET.Element("dating_profile")
    root.set("source", "finya")
    root.set("lang", "de")

    # -- Identity ---------------------------------------------------------- #
    identity = ET.SubElement(root, "identity")
    if p.get("nickname"):
        _el(identity, "nickname", p["nickname"])
    birthdate = p.get("birthdate") or {}
    if birthdate.get("age"):
        _el(identity, "age", birthdate["age"], unit="years")
    body = (p.get("appearance") or {}).get("body") or {}
    if body.get("heightInCentimeters"):
        _el(identity, "height", body["heightInCentimeters"], unit="cm")
    residence = p.get("residence") or {}
    if residence.get("city"):
        _el(identity, "city", residence["city"])
    if not len(identity):
        root.remove(identity)

    # -- Free-text statement ---------------------------------------------- #
    statement = p.get("statement") or {}
    if statement.get("content"):
        _el(root, "statement", statement["content"], de="Persönliches Statement")

    # -- Current mood ------------------------------------------------------ #
    feels_like_id = (sentiment.get("feelsLike") or {}).get("id")
    feels_like = L.OPTIONS.get(f"feelsLike{feels_like_id}") if feels_like_id else None
    mood_id = (sentiment.get("mood") or {}).get("id")
    mood = L.OPTIONS.get(f"mood{mood_id}") if mood_id else None
    if feels_like or mood:
        mood_el = ET.SubElement(root, "current_mood")
        if feels_like:
            _el(mood_el, "wants", feels_like, de="Lust auf")
        if mood:
            _el(mood_el, "feeling", mood, de="Fühle mich")

    # -- Smoking ----------------------------------------------------------- #
    smoking_id = (lifestyle.get("smoking") or {}).get("id")
    if smoking_id == 1:
        _el(root, "smoking", L.OPTIONS.get("smoking_yes"), de="Raucher")
    elif smoking_id == -1:
        _el(root, "smoking", L.OPTIONS.get("smoking_no"), de="Raucher")

    # -- What they're looking for ----------------------------------------- #
    orientation = p.get("orientation") or {}
    orientation_labels = [
        L.OPTIONS.get(f"lookingFor{o.get('id')}")
        for o in orientation.get("lookingFors") or []
    ]
    lf = _list_el(root, "looking_for", orientation_labels)
    if lf is not None:
        lf.set("de", "Ich suche")

    # -- Education / occupation ------------------------------------------- #
    language_labels = []
    for l in education.get("languages") or []:
        lid = l.get("id")
        language_labels.append(L.OPTIONS.get(f"educationLanguage_{lid}", lid))

    school = education.get("school") or {}
    school_label = L.OPTIONS.get(f"school{school.get('id')}") if school.get("id") else None

    job = education.get("job") or {}
    job_label = L.OPTIONS.get(f"jobs{job.get('id')}") if job.get("id") else None
    job_desc = (job.get("description") or {}).get("content")

    if language_labels or school_label or job_label or job_desc:
        edu = ET.SubElement(root, "education")
        langs = _list_el(edu, "languages", language_labels, item_tag="language")
        if langs is not None:
            langs.set("de", "Sprachen")
        if school_label:
            _el(edu, "school", school_label, de="Schule")
        if job_label or job_desc:
            occ = ET.SubElement(edu, "occupation")
            occ.set("de", "Beruf")
            if job_label:
                _el(occ, "field", job_label)
            if job_desc:
                _el(occ, "description", job_desc)

    # -- Lifestyle habits -------------------------------------------------- #
    eating = map_habit_ids("eatingHabits", lifestyle.get("eatingHabits"))
    mobility = map_habit_ids("mobilityHabits", lifestyle.get("mobilityHabits"))
    tv = map_habit_ids("tvHabits", lifestyle.get("tvHabits"))
    music = map_habit_ids("musicHabits", lifestyle.get("musicHabits"))
    if eating or mobility or tv or music:
        ls = ET.SubElement(root, "lifestyle")
        for tag, values, de in [
            ("eating", eating, "Essgewohnheiten"),
            ("mobility", mobility, "Mobilität"),
            ("tv_and_film", tv, "TV/Film"),
            ("music", music, "Musik"),
        ]:
            el = _list_el(ls, tag, values)
            if el is not None:
                el.set("de", de)

    # -- Likes / dislikes from tags --------------------------------------- #
    tags = lifestyle.get("tags") if isinstance(lifestyle.get("tags"), list) else []
    like_labels = [t.get("content") for t in tags if t.get("isLike") is True]
    dislike_labels = [t.get("content") for t in tags if t.get("isLike") is False]
    likes = _list_el(root, "likes", like_labels)
    if likes is not None:
        likes.set("de", "Ich mag")
    dislikes = _list_el(root, "dislikes", dislike_labels)
    if dislikes is not None:
        dislikes.set("de", "Ich mag nicht")

    # -- Questionnaire ----------------------------------------------------- #
    qna_pairs = map_qna(lifestyle.get("qnas"))
    if qna_pairs:
        qna = ET.SubElement(root, "questionnaire")
        qna.set("de", "Fragen und Antworten")
        for question, answer in qna_pairs:
            qa = ET.SubElement(qna, "qa")
            _el(qa, "question", question)
            _el(qa, "answer", answer)

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body + "\n"


# --------------------------------------------------------------------------- #
# Pictures
# --------------------------------------------------------------------------- #
def to_abs(u):
    if not u:
        return None
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if u.startswith("/"):
        return f"{BASE}{u}"
    return u


def download_pictures(session: requests.Session, profile: dict, dest: Path) -> int:
    p = profile.get("payload") or {}
    pictures = p.get("pictures") if isinstance(p.get("pictures"), list) else []
    if not pictures:
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx, pic in enumerate(pictures, start=1):
        # Prefer the largest available rendition.
        url = to_abs(
            pic.get("largeSizeUrl")
            or pic.get("mediumSizeUrl")
            or pic.get("smallSizeUrl")
        )
        if not url:
            continue

        name = Path(urlparse(url).path).name or f"picture_{idx}.jpg"
        # Prefix with the ordinal so ordering is preserved and names stay unique.
        filename = f"{idx:02d}_{name}"
        try:
            r = session.get(url)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  ! Failed to download {url}: {e}", file=sys.stderr)
            continue
        (dest / filename).write_bytes(r.content)
        count += 1
        print(f"  - saved {filename} ({len(r.content)} bytes)")
    return count


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def safe_folder_name(name: str) -> str:
    """Make a nickname safe to use as a folder name on Windows."""
    name = (name or "unknown").strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    return name or "unknown"


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python finya_fetch.py <id>")
    link_id = sys.argv[1]

    cookie = get_cookie()
    encounter = os.getenv("ENCOUNTER_ID", DEFAULT_ENCOUNTER)

    session = requests.Session()
    session.headers.update(
        {
            "Cookie": cookie,
            "User-Agent": "Mozilla/5.0",
        }
    )

    # A 20-char id is a shared-link id that must be converted; a 12-char id is
    # already the internal user link and can be used directly.
    if len(link_id) == 12:
        link = link_id
        print(f"[1/4] Id is already a link ({len(link_id)} chars), skipping convert")
        print(f"      -> link: {link}")
    else:
        print(f"[1/4] Converting link id: {link_id} ({len(link_id)} chars)")
        link = convert_link(session, link_id)
        print(f"      -> link: {link}")

    print(f"[2/4] Fetching profile (encounter {encounter})")
    profile = fetch_profile(session, encounter, link)

    nickname = (profile.get("payload") or {}).get("nickname") or "unknown"
    folder = PROFILES_DIR / f"finya_{safe_folder_name(nickname)}"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"      -> nickname: {nickname}")
    print(f"      -> folder:   {folder}")

    # Keep the raw JSON alongside the rendered text (handy for debugging/re-runs).
    (folder / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[3/4] Writing profile.xml")
    content = build_profile_xml(profile)
    (folder / "profile.xml").write_text(content, encoding="utf-8")

    print("[4/4] Downloading pictures")
    n = download_pictures(session, profile, folder / "pictures")
    print(f"      -> {n} picture(s) downloaded")

    print(f"\nDone. Profile written to {folder}")


if __name__ == "__main__":
    main()
