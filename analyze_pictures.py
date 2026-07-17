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
        "vibe": {
            "type": "string",
            "description": "The overall vibe/atmosphere of the photo in a few words.",
        },
        "is_selfie": {
            "type": "boolean",
            "description": "True if the photo appears to be a selfie (taken by the subject).",
        },
        "hobbies_activities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Hobbies or activities detectable in the photo. Empty if none.",
        },
        "setting": {
            "type": "string",
            "enum": ["inside", "outside", "unknown"],
            "description": "Whether the photo was taken indoors or outdoors.",
        },
        "mood": {
            "type": "string",
            "description": "The mood of the person in the photo.",
        },
        "clothing": {
            "type": "string",
            "description": "The type of clothes the person is wearing.",
        },
        "text_and_icons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any icons, logos, or text/signs visible in the picture. Empty if none.",
        },
    },
    "required": [
        "vibe",
        "is_selfie",
        "hobbies_activities",
        "setting",
        "mood",
        "clothing",
        "text_and_icons",
    ],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# XML helpers (mirroring finya_fetch.py's style)
# --------------------------------------------------------------------------- #
def _el(parent, tag, text=None, **attrs):
    """Create a SubElement, optionally with text and attributes."""
    e = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        e.set(k, str(v))
    if text is not None:
        e.text = str(text)
    return e


def _list_el(parent, tag, values, item_tag="item"):
    """Create <tag><item>..</item>..</tag>, always present (may be empty)."""
    container = ET.SubElement(parent, tag)
    for v in values or []:
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
    _el(pic, "vibe", result.get("vibe", ""))
    _el(pic, "selfie", "true" if result.get("is_selfie") else "false")
    _el(pic, "setting", result.get("setting", "unknown"))
    _el(pic, "mood", result.get("mood", ""))
    _el(pic, "clothing", result.get("clothing", ""))
    _list_el(pic, "hobbies", result.get("hobbies_activities"))
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
