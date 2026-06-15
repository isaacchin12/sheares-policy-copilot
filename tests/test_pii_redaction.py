"""
Smoke test: assert that after PII redaction, no obvious personal identifiers
remain in the text that would be passed to the embedder.

This test runs over data/redacted_samples/ (the committed corpus subset).
The full corpus is gitignored and never committed.
"""
import re
import sys
from pathlib import Path

# ── Patterns we must NOT see in indexed text ──────────────────────────────────
# These are deliberately permissive — false positives are ok, false negatives are not.
PII_PATTERNS = [
    # NUS matric numbers: A1234567B
    re.compile(r"\bA\d{7}[A-Z]\b"),
    # Singapore NRIC: S/T/F/G + 7 digits + letter
    re.compile(r"\b[STFG]\d{7}[A-HJKLMNPQRSTUVWXYZ]\b"),
    # Email addresses
    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    # Singapore mobile numbers: +65 8/9 followed by 7 digits, or 8/9XXXXXXX
    re.compile(r"(\+65[\s\-]?)?[89]\d{7}\b"),
]

SAMPLE_DIR = Path("data/redacted_samples")


def test_no_pii_in_samples() -> None:
    if not SAMPLE_DIR.exists():
        print(f"[SKIP] {SAMPLE_DIR} does not exist yet — no samples to check.")
        return

    text_files = list(SAMPLE_DIR.rglob("*.txt")) + list(SAMPLE_DIR.rglob("*.md"))
    if not text_files:
        print("[SKIP] No text files in redacted_samples yet.")
        return

    violations: list[str] = []
    for f in text_files:
        content = f.read_text(errors="ignore")
        for pattern in PII_PATTERNS:
            matches = pattern.findall(content)
            for m in matches:
                violations.append(f"{f.name}: matched '{m}' ({pattern.pattern})")

    if violations:
        print("PII DETECTED IN REDACTED SAMPLES:")
        for v in violations:
            print(f"  {v}")
        sys.exit(1)
    else:
        print(f"OK — checked {len(text_files)} files, zero PII patterns found.")


if __name__ == "__main__":
    test_no_pii_in_samples()
