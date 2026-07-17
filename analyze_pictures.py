# -*- coding: utf-8 -*-
"""Analyze every picture in a profile folder with Claude vision and record the
findings back into that profile's profile.xml.

Usage:
    python analyze_pictures.py <profile_folder_name>

Where <profile_folder_name> is a directory inside profiles/, e.g.:
    python analyze_pictures.py finya_marksh1234

Steps:
  1. Locate profiles/<folder>/pictures/ and profiles/<folder>/profile.xml.
  2. For each picture, ask Claude (Anthropic platform) for a structured reading:
       - vibe
       - whether it's a selfie
       - hobbies / activities detectable in the photo
       - whether it's shot inside or outside
       - the mood of the person
       - the type of clothes
       - any icons or text signs visible
  3. Write one <picture> record per image under a <pictures> element in
     profile.xml (replacing a previous <pictures> block if the script is re-run).

The ANTHROPIC_API_KEY is read from the project's .env file (via python-dotenv);
the Anthropic SDK picks it up from the environment automatically.
"""

import base64
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv
import anthropic

PROJECT_ROOT = Path(__file__).resolve().parent
PROFILES_DIR = PROJECT_ROOT / "profiles"

MODEL = "claude-opus-4-8"

# Extension -> media type for the base64 image source.
MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

SYSTEM_PROMPT = (
    "You analyze a single photo from a dating profile and describe what is "
    "visible. Judge only from the image itself; do not invent details you "
    "cannot see. Keep each field short and concrete."
)

# Structured-output schema: guarantees a parseable JSON object with exactly
# these fields. `additionalProperties: false` and `required` are mandatory.
RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "mood_energy": {
            "type": "string",
            "description": (
                "The energy and mood of the person (or the group, if several "
                "people) and the overall atmosphere of the photo, in a few words."
            ),
        },
        "photo_type": {
            "type": "string",
            "description": (
                "How the photo was taken / staged, signalling how much effort "
                "and personality went into it. E.g. professional shot, candid, "
                "mirror pic, group photo, posed, snapshot."
            ),
        },
        "setting_type": {
            "type": "string",
            "description": (
                "The recognizable type of setting, and its more specific "
                "type/origin if detectable (e.g. beach, forest, bar, club, boat, "
                "park; or more specific: arctic forest, hunting pub, dungeon "
                "club). Name any identifiable landmark (e.g. Eiffel Tower). "
                "Empty string if not detectable."
            ),
        },
        "social_context": {
            "type": "string",
            "description": (
                "Number of people and social context: e.g. solo, with friends, "
                "with a partner, at a group event."
            ),
        },
        "style_signals": {
            "type": "string",
            "description": (
                "Aesthetic / style signals: overall fashion sense and any "
                "subculture cues (e.g. skater, cottagecore, gym, minimalist, "
                "streetwear, formal)."
            ),
        },
        "hobbies": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Activities that can be confidently inferred as a genuine hobby "
                "or interest from the photo (e.g. hiking, travelling, playing "
                "guitar, cycling as sport). Do NOT include mundane necessities. "
                "Empty if none can be confidently inferred as a hobby."
            ),
        },
        "current_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "What the person is literally seen doing right now, regardless "
                "of whether it is a hobby (e.g. commuting, reading, cycling, "
                "hiking, playing guitar, eating). This is the raw observed "
                "action, not an inferred interest. Empty if not discernible."
            ),
        },
        "conversation_hooks": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Distinctive objects that invite comment or a conversation "
                "opener (e.g. a guitar, a specific book title, a dog breed, a "
                "mountain bike, a chess board). Be specific. Empty if none."
            ),
        },
        "pets_animals": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Pets or animals detectable in the photo, with breed/type if "
                "identifiable (e.g. golden retriever, cat, horse). Empty if none."
            ),
        },
        "text_and_icons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any icons, logos, or text/signs visible in the picture. Empty if none.",
        },
    },
    "required": [
        "mood_energy",
        "photo_type",
        "setting_type",
        "social_context",
        "style_signals",
        "hobbies",
        "current_actions",
        "conversation_hooks",
        "pets_animals",
        "text_and_icons",
    ],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# XML helpers (mirroring finya_fetch.py's style)
# --------------------------------------------------------------------------- #
def _el(parent, tag, text=None, **attrs):
    """Create a SubElement, optionally with text and attributes.

    Scalar fields with no text are skipped so we never emit empty tags.
    """
    has_text = text is not None and (not isinstance(text, str) or text.strip())
    if not has_text and not attrs:
        return None
    e = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        e.set(k, str(v))
    if text is not None:
        e.text = str(text)
    return e


def _list_el(parent, tag, values, item_tag="item"):
    """Create <tag><item>..</item>..</tag> only when there are values."""
    values = [v for v in (values or []) if v and str(v).strip()]
    if not values:
        return None
    container = ET.SubElement(parent, tag)
    for v in values:
        _el(container, item_tag, v)
    return container


# --------------------------------------------------------------------------- #
# Vision analysis
# --------------------------------------------------------------------------- #
def analyze_picture(client: anthropic.Anthropic, path: Path) -> dict:
    """Send one image to Claude and return the structured reading as a dict."""
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if not media_type:
        raise ValueError(f"Unsupported image type: {path.name}")

    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyze this dating-profile photo and return the structured reading.",
                    },
                ],
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": RESULT_SCHEMA}},
    )

    # output_config.format guarantees the first text block is valid JSON.
    import json

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def build_picture_element(parent, filename: str, result: dict):
    """Append a <picture> record built from one analysis result."""
    pic = ET.SubElement(parent, "picture")
    pic.set("file", filename)
    _el(pic, "mood_energy", result.get("mood_energy", ""))
    _el(pic, "photo_type", result.get("photo_type", ""))
    _el(pic, "setting_type", result.get("setting_type", ""))
    _el(pic, "social_context", result.get("social_context", ""))
    _el(pic, "style_signals", result.get("style_signals", ""))
    _list_el(pic, "hobbies", result.get("hobbies"))
    _list_el(pic, "current_actions", result.get("current_actions"))
    _list_el(pic, "conversation_hooks", result.get("conversation_hooks"))
    _list_el(pic, "pets_animals", result.get("pets_animals"))
    _list_el(pic, "text_and_icons", result.get("text_and_icons"))
    return pic


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python analyze_pictures.py <profile_folder_name>")

    folder = PROFILES_DIR / sys.argv[1]
    if not folder.is_dir():
        sys.exit(f"ERROR: profile folder not found: {folder}")

    xml_path = folder / "profile.xml"
    if not xml_path.is_file():
        sys.exit(f"ERROR: profile.xml not found: {xml_path}")

    pictures_dir = folder / "pictures"
    if not pictures_dir.is_dir():
        sys.exit(f"ERROR: pictures/ folder not found: {pictures_dir}")

    images = sorted(
        p for p in pictures_dir.iterdir()
        if p.is_file() and p.suffix.lower() in MEDIA_TYPES
    )
    if not images:
        sys.exit(f"ERROR: no images found in {pictures_dir}")

    load_dotenv(PROJECT_ROOT / ".env")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Replace any previous <pictures> block so the script is idempotent.
    for old in root.findall("pictures"):
        root.remove(old)
    pictures_el = ET.SubElement(root, "pictures")
    pictures_el.set("de", "Bilder")

    print(f"Analyzing {len(images)} picture(s) in {pictures_dir}")
    for idx, path in enumerate(images, start=1):
        print(f"  [{idx}/{len(images)}] {path.name}")
        try:
            result = analyze_picture(client, path)
        except Exception as e:  # keep going; record the failure in the XML
            print(f"      ! analysis failed: {e}", file=sys.stderr)
            err = ET.SubElement(pictures_el, "picture")
            err.set("file", path.name)
            err.set("error", str(e))
            continue
        build_picture_element(pictures_el, path.name, result)

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="unicode")
    xml_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_body + "\n",
        encoding="utf-8",
    )
    print(f"\nDone. Updated {xml_path}")


if __name__ == "__main__":
    main()
