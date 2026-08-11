"""
aristotelian_classifier.py
---------------------------
Takes the consolidated definitions.csv (from definitions.py) and, for each
term/meaning pair, asks Gemini whether the definition follows the classical
Aristotelian form -- "X is a [genus] that/which [differentia]" -- naming the
immediate broader category the term belongs to (genus), then what
distinguishes it from other members of that category (differentia).

For definitions that DON'T fit that form (circular, gives an example instead
of a category, describes a process/formula, no clear genus, etc.), it also
asks Gemini to suggest a plausible genus and differentia, grounded in the
textbook's own definition text.

Requires:
    pip install google-genai --break-system-packages
    export GEMINI_API_KEY=...

Usage:
    python aristotelian_classifier.py definitions.csv --out definitions_classified.csv
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

SYSTEM_PROMPT = """You are analyzing definitions from a college-level finance \
textbook (OpenStax "Principles of Finance") to determine whether each one \
follows the classical Aristotelian form of definition. 

An Aristotelian definition consists of two parts: (1) a genus (or family) -- An existing 
definition that serves as a portion of the new definition; all definitions with the
 same genus are considered members of that genus; (2) the differentia: The portion of the definition 
 that is not provided by the genus.  
 
 For example, consider the following two definitions:
a triangle: A plane figure that has 3 straight bounding sides.
a quadrilateral: A plane figure that has 4 straight bounding sides.

These definitions have one genus and two differentiae. The genus for both triangle and a quadrilateral 
is a plane figure. The differentia for a triangle is that has 3 straight bounding sides, and the 
differentia for a quadrilateral is that has 4 straight bounding sides.


A definition does NOT count as Aristotelian if it:
- is circular (restates the term using the term itself or a close synonym)
- only gives an example or illustration instead of a category

For each term + definition given, decide:
1. is_aristotelian: true if it clearly fits the genus+differentia pattern above.
2. If true: extract "genus" (the category noun phrase) and "differentia" \
(the distinguishing clauses), reusing the textbook's own wording as closely \
as possible.
3. If false: leave "genus"/"differentia" empty, give a short "reason" \
(a few words, e.g. "circular", "not a category", \
"gives an example instead of a genus", "no genus stated"), and propose \
"suggested_genus" and "suggested_differentia" -- your best-effort rewrite \
of what a genus and differentia could be, grounded in the textbook \
definition's own content (don't invent unrelated facts).

Output ONLY valid JSON, no markdown fences, no commentary.

JSON schema:
{
  "is_aristotelian": true or false,
  "genus": string or null,
  "differentia": string or null,
  "reason": string or null,
  "suggested_genus": string or null,
  "suggested_differentia": string or null
}
"""

USER_TEMPLATE = """Term: {term}
Textbook definition: {meaning}

Return the JSON now."""


@dataclass
class Classification:
    is_aristotelian: bool | None = None
    genus: str = ""
    differentia: str = ""
    reason: str = ""
    suggested_genus: str = ""
    suggested_differentia: str = ""
    error: str | None = None


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


class AristotelianClassifier:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None,
                 max_retries: int = 3, temperature: float = 0.0):
        from google import genai  # imported lazily so callers who only need CSV-merge logic don't need it installed
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature

    def classify(self, term: str, meaning: str) -> Classification:
        from google.genai import types

        user_msg = USER_TEMPLATE.format(term=term, meaning=meaning)

        last_err = None
        token_budget = 800  # bumped up each retry if we get truncated
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=self.temperature,
                        max_output_tokens=token_budget,
                        response_mime_type="application/json",
                        # Gemini 2.5 models "think" by default, and those thinking
                        # tokens count against max_output_tokens -- with thinking
                        # left on, the model can burn the whole budget on internal
                        # reasoning and get cut off mid-JSON before writing the
                        # actual answer. This is a plain extraction task with no
                        # need for chain-of-thought, so we turn thinking off.
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                candidates = getattr(resp, "candidates", None) or []
                finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
                finish_reason_name = getattr(finish_reason, "name", str(finish_reason))

                raw = resp.text or ""
                if finish_reason_name == "MAX_TOKENS" or not raw:
                    last_err = (f"response truncated (finish_reason=MAX_TOKENS, "
                                f"token_budget was {token_budget}); retrying with a larger budget")
                    token_budget *= 2
                    time.sleep(min(2 ** attempt, 10))
                    continue

                data = json.loads(_strip_fences(raw))
                return Classification(
                    is_aristotelian=bool(data.get("is_aristotelian")),
                    genus=(data.get("genus") or "").strip(),
                    differentia=(data.get("differentia") or "").strip(),
                    reason=(data.get("reason") or "").strip(),
                    suggested_genus=(data.get("suggested_genus") or "").strip(),
                    suggested_differentia=(data.get("suggested_differentia") or "").strip(),
                )
            except json.JSONDecodeError as e:
                last_err = f"JSON parse error: {e}"
            except Exception as e:
                last_err = str(e)
            time.sleep(min(2 ** attempt, 10))

        return Classification(error=last_err)


CLASSIFICATION_FIELDS = [
    "is_aristotelian", "genus", "differentia",
    "reason", "suggested_genus", "suggested_differentia",
    "classification_error",
]


def _classification_row(cls: Classification) -> dict:
    if cls.error:
        return {
            "is_aristotelian": "", "genus": "", "differentia": "",
            "reason": "", "suggested_genus": "", "suggested_differentia": "",
            "classification_error": cls.error,
        }
    is_yes = bool(cls.is_aristotelian)
    return {
        "is_aristotelian": "yes" if is_yes else "no",
        "genus": cls.genus if is_yes else "",
        "differentia": cls.differentia if is_yes else "",
        "reason": "" if is_yes else cls.reason,
        "suggested_genus": "" if is_yes else cls.suggested_genus,
        "suggested_differentia": "" if is_yes else cls.suggested_differentia,
        "classification_error": "",
    }


def classify_csv(in_csv: str | Path, out_csv: str | Path,
                  model: str = "gemini-2.5-flash", max_workers: int = 4,
                  api_key: str | None = None) -> int:
    """Read definitions.csv (any schema with at least 'term' and 'meaning'
    columns -- works with both the consolidated and --no-consolidate output
    of definitions.py), classify every row, and write out a new CSV with the
    original columns plus the classification columns appended.
    Returns the number of rows written."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    in_path = Path(in_csv)
    with open(in_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        orig_fieldnames = reader.fieldnames or []

    if "term" not in orig_fieldnames or "meaning" not in orig_fieldnames:
        raise ValueError(f"{in_path} must have 'term' and 'meaning' columns; found {orig_fieldnames}")

    classifier = AristotelianClassifier(model=model, api_key=api_key)
    results: list[Classification | None] = [None] * len(rows)

    def work(i: int, row: dict):
        return i, classifier.classify(row.get("term", ""), row.get("meaning", ""))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(work, i, row): i for i, row in enumerate(rows)}
        done = 0
        for fut in as_completed(futures):
            i, cls = fut.result()
            results[i] = cls
            done += 1
            print(f"[aristotelian_classifier] {done}/{len(rows)} done", end="\r")
    print()

    out_fieldnames = orig_fieldnames + CLASSIFICATION_FIELDS
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row, cls in zip(rows, results):
            out_row = dict(row)
            out_row.update(_classification_row(cls))
            writer.writerow(out_row)

    n_errors = sum(1 for c in results if c and c.error)
    if n_errors:
        print(f"[aristotelian_classifier] WARNING: {n_errors}/{len(rows)} rows failed classification "
              f"(see classification_error column)")

    return len(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Classify textbook definitions as Aristotelian (genus + differentia) or not.")
    ap.add_argument("--input", default="definitions.csv",help="Path to definitions.csv produced by definitions.py.")
    ap.add_argument("--out", default="definitions_classified.csv", help="Output CSV path.")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use.")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent LLM requests.")
    ap.add_argument("--api-key", default=None, help="Gemini API key (else reads GEMINI_API_KEY/GOOGLE_API_KEY env var).")
    args = ap.parse_args()

    n = classify_csv(args.input, args.out, model=args.model, max_workers=args.workers, api_key=args.api_key)
    print(f"Classified {n} definitions -> {args.out}")