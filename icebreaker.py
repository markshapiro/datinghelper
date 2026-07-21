# -*- coding: utf-8 -*-
"""Generate opening-message ideas for a sender writing to a recipient.

Usage:
    python icebreaker.py <sender_folder> <recipient_folder>

Both arguments are directory names inside profiles/, e.g.:
    python icebreaker.py finya_marksh1234 finya_An_na

Steps:
  1. Read profiles/<sender>/profile.xml and profiles/<recipient>/profile.xml.
  2. Combine both into one <icebreaker_request> root, the sender's profile
     under <sender> and the recipient's under <recipient>.
  3. Ask Claude for 5 message ideas and print them in readable form.

The ANTHROPIC_API_KEY is read from the project's .env file (via python-dotenv);
the Anthropic SDK picks it up from the environment automatically.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv
import anthropic

PROJECT_ROOT = Path(__file__).resolve().parent
PROFILES_DIR = PROJECT_ROOT / "profiles"

MODEL = "claude-opus-4-8"

MESSAGE_COUNT = 5

# ANSI styling for terminal output.
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

SYSTEM_PROMPT = """You write opening messages for a dating app on behalf of the SENDER, to be sent to the RECIPIENT.

You receive two profiles in XML: <sender> and <recipient>. Write ONLY from the sender's perspective, addressing the recipient.

Goals for each message:
- Reference something SPECIFIC and genuine from the recipient's profile (a shared interest, a photo hook, a questionnaire answer) — never generic ("hey, how's it going").
- Prefer OVERLAP between sender and recipient where it exists (e.g. shared vegan/vegetarian eating, hiking, classical/jazz music).
- Warm, low-pressure, curious. Invite a reply with a light question. No pickup-line cheese, no negging, no comments on appearance/body, no assumptions about relationship intent.
- Match the recipient's apparent language (German profile → German message) unless told otherwise.
- Keep it to 1–3 sentences.
- Photo-derived details may be wrong. Only use a hook if it's plausible; never state a brand/detail as fact if it reads as a guess.

Produce a VARIETY of angles across the set (e.g. one playful, one sincere, one shared-interest, one question-led).

Output ONLY valid JSON, no markdown, in this shape:
{
  "messages": [
    {
      "id": "m1",
      "text": "...",
      "angle": "shared_interest | playful | sincere | question_led | photo_hook",
      "hook_source": "which profile element this draws on",
      "language": "de | en",
      "confidence": 0.0-1.0
    }
  ]
}"""


# --------------------------------------------------------------------------- #
# Profile loading / combining
# --------------------------------------------------------------------------- #
def load_profile(folder_name: str) -> ET.Element:
    """Return the <dating_profile> root of profiles/<folder_name>/profile.xml."""
    folder = PROFILES_DIR / folder_name
    if not folder.is_dir():
        sys.exit(f"ERROR: profile folder not found: {folder}")

    xml_path = folder / "profile.xml"
    if not xml_path.is_file():
        sys.exit(f"ERROR: profile.xml not found: {xml_path}")

    return ET.parse(xml_path).getroot()


def build_request_xml(sender: ET.Element, recipient: ET.Element) -> str:
    """Combine both profiles under a single <icebreaker_request> root."""
    root = ET.Element("icebreaker_request")
    ET.SubElement(root, "sender").append(sender)
    ET.SubElement(root, "recipient").append(recipient)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #
def request_generation(client: anthropic.Anthropic, history: list) -> tuple:
    """Send the running conversation and return (parsed_result, assistant_text).

    `history` is the list of user/assistant turns; the caller appends the next
    user turn before calling and appends the returned assistant text after, so
    the full context is preserved across regenerations.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=history,
    )

    text = next(b.text for b in response.content if b.type == "text").strip()
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        sys.exit(f"ERROR: model did not return valid JSON:\n{text}")


# --------------------------------------------------------------------------- #
# Console setup
# --------------------------------------------------------------------------- #
def setup_console():
    """Make stdin/stdout UTF-8 (German umlauts) and enable ANSI on Windows."""
    for stream in (sys.stdout, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    if sys.platform == "win32":
        import os

        os.system("")  # flip on the console's ANSI (VT) processing


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_messages(result: dict, sender_name: str, recipient_name: str):
    messages = result.get("messages", [])
    if not messages:
        print("No messages returned.")
        return

    print(f"\nIce breakers: {sender_name} -> {recipient_name}")
    print("=" * 70)
    for msg in messages:
        msg_id = msg.get("id", "?")
        text = msg.get("text", "")
        print(f"\n{DIM}[{msg_id}]{RESET} {BOLD}{CYAN}{text}{RESET}")
        details = [
            ("angle", msg.get("angle")),
            ("hook", msg.get("hook_source")),
            ("lang", msg.get("language")),
            ("confidence", msg.get("confidence")),
        ]
        line = "  " + "  |  ".join(
            f"{k}: {v}" for k, v in details if v is not None and str(v).strip()
        )
        print(f"{DIM}{line}{RESET}")
    print()


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
def collect_feedback(result: dict):
    """Prompt one line per suggestion and build the feedback JSON.

    For each message: an empty/space-only entry rejects it; any other text
    marks it as preferred and is stored as a note. Returns the feedback dict,
    or None if the user chose to quit (empty gate answer, EOF, or Ctrl-C).
    """
    messages = result.get("messages", [])
    if not messages:
        return None

    print(f"\n{BOLD}Feedback{RESET} — per suggestion: {DIM}space/Enter = reject, "
          f"any text = prefer (+ note). Blank at the gate = quit.{RESET}")
    try:
        gate = input(f"{YELLOW}Regenerate with feedback? [Enter=yes, q=quit] {RESET}")
    except (EOFError, KeyboardInterrupt):
        return None
    if gate.strip().lower() in ("q", "quit", "n", "no"):
        return None

    preferred, rejected, notes = [], [], {}
    for msg in messages:
        mid = msg.get("id", "?")
        try:
            answer = input(f"  {DIM}[{mid}]{RESET} note (space=reject): ")
        except (EOFError, KeyboardInterrupt):
            return None
        if answer.strip() == "":
            rejected.append(mid)
        else:
            preferred.append(mid)
            notes[mid] = answer.strip()

    return {"preferred": preferred, "rejected": rejected, "notes": notes}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python icebreaker.py <sender_folder> <recipient_folder>")

    setup_console()
    sender_name, recipient_name = sys.argv[1], sys.argv[2]

    sender = load_profile(sender_name)
    recipient = load_profile(recipient_name)
    request_xml = build_request_xml(sender, recipient)

    load_dotenv(PROJECT_ROOT / ".env")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    # Running conversation, resent each turn so the model keeps full context.
    history = [
        {
            "role": "user",
            "content": f"{request_xml}\n\nGenerate {MESSAGE_COUNT} message ideas.",
        }
    ]

    print(f"Generating {MESSAGE_COUNT} ice breakers for {sender_name} -> {recipient_name}...")
    result, assistant_text = request_generation(client, history)
    history.append({"role": "assistant", "content": assistant_text})
    print_messages(result, sender_name, recipient_name)

    # Feedback / regeneration loop.
    while True:
        feedback = collect_feedback(result)
        if feedback is None:
            print("Done.")
            break

        feedback_json = json.dumps(feedback, ensure_ascii=False, indent=2)
        history.append(
            {
                "role": "user",
                "content": f"{feedback_json}\nGenerate {MESSAGE_COUNT} new options based on this feedback JSON.",
            }
        )

        print(f"\nRegenerating {MESSAGE_COUNT} ice breakers based on your feedback...")
        result, assistant_text = request_generation(client, history)
        history.append({"role": "assistant", "content": assistant_text})
        print_messages(result, sender_name, recipient_name)


if __name__ == "__main__":
    main()
