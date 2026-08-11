"""
genus_similarity.py
---------------------
Computes pairwise similarity between every distinct "genus" value (from
aristotelian_classifier.py's output) using three methods:

    1. Normalized edit (Levenshtein) similarity: 1 - edit_distance / max_len.
       Sensitive to character-level overlap and word order.
    2. Jaccard similarity on word sets: |intersection| / |union| of each
       phrase's (stopword-stripped) words. Sensitive to shared vocabulary,
       not word order or exact spelling.
    3. (optional, --llm) LLM-judged semantic relationship via Gemini: for
       each pair, asks the model to classify the relationship as exactly one
       of EQUIVALENT / A_SUBSUMES_B / B_SUBSUMES_A / RELATED / DIFFERENT.
       This is the only one of the three that can catch genuinely synonymous
       but differently-worded phrases (e.g. "a financial institution" vs
       "an organization providing financial services" -> EQUIVALENT, despite
       scoring ~0 on both lexical metrics). Costs API calls, roughly
       n*(n-1)/2 pairs, batched to keep the call count down.

Methods 1 and 2 are computed on a NORMALIZED form of each genus string, so
that superficial differences don't count as real dissimilarity:
    1. lowercase
    2. strip leading/trailing whitespace, collapse internal whitespace
    3. remove punctuation that carries no semantic weight (commas, periods,
       parentheses, hyphens treated as word separators, apostrophes dropped)
    4. strip a leading article ("a"/"an"/"the")
    5. cautiously singularize the head noun (the phrase's last word) --
       only for simple, unambiguous regular plurals (see _singularize_word);
       irregular/ambiguous cases like "business", "surplus", "series" are
       deliberately left untouched rather than risk mangling them.

e.g. "A Financial Instruments" and "the financial instrument" both
normalize to "financial instrument" and score a clean 1.0 on both lexical
metrics, instead of a merely-high score for what is really the exact same
phrase. Pairs already identified as identical this way are assigned
EQUIVALENT directly for method 3 too, without spending an API call on them.

Neither lexical metric is truly "semantic" the way embeddings (or the LLM
method) are -- "a monetary metric" and "a financial measure" will score low
on both despite meaning almost the same thing, since they share no words and
little character overlap. That's the tradeoff: methods 1-2 are cheap,
deterministic, need no API calls, and no heavy dependencies (stdlib only) --
good for a fast first pass or for catching near-duplicate/reworded genus
phrases. For genuinely semantic grouping, use --llm here or genus_clustering.py.

Usage:
    python genus_similarity.py definitions_classified.csv --out genus_pairwise_similarity.csv
    python genus_similarity.py definitions_classified.csv --no-singularize   # disable step 5
    python genus_similarity.py definitions_classified.csv --llm             # add the LLM method
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

# Small set of function words stripped when tokenizing for Jaccard -- catches
# non-leading filler words (normalize_genus already strips a LEADING
# a/an/the; this is a supplementary net for mid-phrase occurrences like
# "instrument of a debt").
_STOPWORDS = {"a", "an", "the", "of", "that", "which", "is", "are", "for", "to"}

_LEADING_ARTICLE_RE = re.compile(r"^(a|an|the)\s+")
_APOSTROPHE_RE = re.compile(r"[\u2019\u2018']")   # straight and curly single quotes
_PUNCT_TO_SPACE_RE = re.compile(r"[.,;:!?()\[\]{}\"/\\_-]")
_WHITESPACE_RE = re.compile(r"\s+")

# Valid labels for the LLM relationship method. genus_1 plays "Genus A" and
# genus_2 plays "Genus B" in the prompt, in the same order they're stored in
# PairwiseResult -- so A_SUBSUMES_B means genus_1 subsumes genus_2.
LLM_RELATIONSHIP_LABELS = ("EQUIVALENT", "A_SUBSUMES_B", "B_SUBSUMES_A", "RELATED", "DIFFERENT")

# When mirroring a relationship for the reverse pair order (for the matrix
# CSV), EQUIVALENT/RELATED/DIFFERENT are symmetric, but subsumption flips.
_LLM_MIRROR = {
    "EQUIVALENT": "EQUIVALENT",
    "A_SUBSUMES_B": "B_SUBSUMES_A",
    "B_SUBSUMES_A": "A_SUBSUMES_B",
    "RELATED": "RELATED",
    "DIFFERENT": "DIFFERENT",
}

# Words that look like regular plurals by suffix pattern but aren't (or are
# ambiguous/invariant), so singularization deliberately leaves them alone.
_SINGULARIZE_EXCEPTIONS = {
    "series", "species", "means", "news", "analysis", "basis", "axis",
}


def _singularize_word(word: str) -> str:
    """Cautiously singularize a single word -- only for patterns that are
    unambiguous regular plurals. Deliberately conservative: when in doubt,
    return the word unchanged rather than risk mangling it."""
    if word in _SINGULARIZE_EXCEPTIONS or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"                      # agencies -> agency
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]                             # processes -> process, boxes -> box
    if word.endswith("us") or word.endswith("ss"):
        return word                                  # surplus, business, bonus, class -- not simple plurals
    if word.endswith("s") and not word.endswith("'s"):
        return word[:-1]                             # instruments -> instrument
    return word


def normalize_genus(text: str, singularize: bool = True) -> str:
    """Full normalization pipeline described in the module docstring.
    Returns a cleaned, lowercase, whitespace-collapsed string."""
    t = text.lower().strip()
    t = _APOSTROPHE_RE.sub("", t)          # owner's -> owners (drop, don't split into two tokens)
    t = _PUNCT_TO_SPACE_RE.sub(" ", t)     # other punctuation -> space, so words don't glue together
    t = _WHITESPACE_RE.sub(" ", t).strip()
    t = _LEADING_ARTICLE_RE.sub("", t)     # strip ONLY a leading article, not mid-phrase occurrences
    t = t.strip()

    if singularize and t:
        words = t.split(" ")
        words[-1] = _singularize_word(words[-1])   # only the head noun (last word)
        t = " ".join(words)

    return t


# --------------------------------------------------------------------------
# Load genus values (standalone -- deliberately doesn't import
# genus_clustering.py, so this module has zero external dependencies)
# --------------------------------------------------------------------------

@dataclass
class GenusRow:
    term: str
    genus: str
    source: str  # "extracted" or "suggested"


def load_genus_rows(in_csv: str | Path, include_suggested: bool = False) -> list[GenusRow]:
    """Read definitions_classified.csv (from aristotelian_classifier.py) and
    pull out one GenusRow per term. By default only uses rows where a genus
    was actually extracted (is_aristotelian == yes); pass include_suggested=True
    to also fold in suggested_genus for rows that failed the test."""
    rows: list[GenusRow] = []
    with open(in_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = (row.get("term") or "").strip()
            if not term:
                continue
            is_yes = (row.get("is_aristotelian") or "").strip().lower() == "yes"
            if is_yes:
                genus = (row.get("genus") or "").strip()
                if genus:
                    rows.append(GenusRow(term=term, genus=genus, source="extracted"))
            elif include_suggested:
                genus = (row.get("suggested_genus") or "").strip()
                if genus:
                    rows.append(GenusRow(term=term, genus=genus, source="suggested"))
    return rows


# --------------------------------------------------------------------------
# Metric 1: normalized edit (Levenshtein) similarity
# --------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Classic O(len(s1)*len(s2)) dynamic-programming edit distance.
    No dependency needed -- this is fast enough for the phrase-length
    strings genus values are (a handful of words)."""
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        c1 = s1[i - 1]
        for j in range(1, n + 1):
            cost = 0 if c1 == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[n]


def edit_similarity(s1: str, s2: str, singularize: bool = True) -> float:
    """1 - normalized edit distance, in [0, 1], computed on the NORMALIZED
    form of each string (see normalize_genus). 1.0 means identical after
    normalization -- e.g. "A Financial Instruments" vs "the financial
    instrument" now correctly score 1.0, not just "high"."""
    a, b = normalize_genus(s1, singularize=singularize), normalize_genus(s2, singularize=singularize)
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / max_len


# --------------------------------------------------------------------------
# Metric 2: Jaccard similarity on word sets
# --------------------------------------------------------------------------

def tokenize(text: str, singularize: bool = True) -> set[str]:
    normalized = normalize_genus(text, singularize=singularize)
    words = re.findall(r"[a-z']+", normalized)
    return {w for w in words if w not in _STOPWORDS}


def jaccard_similarity(s1: str, s2: str, singularize: bool = True) -> float:
    """|intersection| / |union| of each phrase's (normalized, stopword-
    stripped) word set, in [0, 1]. 1.0 means identical word sets."""
    a, b = tokenize(s1, singularize=singularize), tokenize(s2, singularize=singularize)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------
# Metric 3 (optional): LLM-judged semantic relationship
# --------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are comparing pairs of "genus" phrases extracted from \
Aristotelian-style definitions in a finance textbook. A genus names the broad \
category a term belongs to (e.g. "a financial institution", "an organization \
providing financial services").

For each numbered pair below (Genus A, Genus B), determine their semantic \
relationship and classify it as EXACTLY one of:
- EQUIVALENT: A and B describe the same category, just worded differently.
- A_SUBSUMES_B: A is a broader category that includes B as a special case.
- B_SUBSUMES_A: B is a broader category that includes A as a special case.
- RELATED: A and B share meaningful conceptual overlap but neither subsumes the other.
- DIFFERENT: A and B describe unrelated or clearly distinct categories.

Output ONLY valid JSON, no markdown fences, no commentary.
Schema: {"relationships": {"<pair_id>": "<EQUIVALENT|A_SUBSUMES_B|B_SUBSUMES_A|RELATED|DIFFERENT>", ...}}
"""

_PAIR_TEMPLATE = "Pair {id}:\nGenus A: {a}\nGenus B: {b}\n"


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _classify_relationship_batch(batch: list[tuple[int, str, str]], model: str,
                                  api_key: str | None, max_retries: int = 3) -> dict[int, str | None]:
    """One LLM call classifying multiple pairs at once. batch is a list of
    (pair_id, genus_a, genus_b). Returns {pair_id: label_or_None}."""
    import json
    import time as _time
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key) if api_key else genai.Client()

    lines = [_PAIR_TEMPLATE.format(id=pid, a=a, b=b) for pid, a, b in batch]
    user_msg = "\n".join(lines) + "\nReturn the JSON now."

    token_budget = max(300, 150 * len(batch))
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=_LLM_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=token_budget,
                    response_mime_type="application/json",
                    # See kg_extractor.py / aristotelian_classifier.py -- Gemini 2.5
                    # models "think" by default and those tokens eat into
                    # max_output_tokens, truncating the JSON. Not needed here.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            candidates = getattr(resp, "candidates", None) or []
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            finish_reason_name = getattr(finish_reason, "name", str(finish_reason))
            raw = resp.text or ""
            if finish_reason_name == "MAX_TOKENS" or not raw:
                token_budget *= 2
                _time.sleep(min(2 ** attempt, 10))
                continue

            data = json.loads(_strip_fences(raw))
            raw_rel = data.get("relationships", {})
            return {
                pid: (raw_rel.get(str(pid)) if raw_rel.get(str(pid)) in LLM_RELATIONSHIP_LABELS else None)
                for pid, _, _ in batch
            }
        except Exception as e:
            last_err = e
            _time.sleep(min(2 ** attempt, 10))

    print(f"[genus_similarity] WARNING: a batch of {len(batch)} pairs failed LLM classification "
          f"after {max_retries} attempts ({last_err}); leaving those unclassified")
    return {pid: None for pid, _, _ in batch}


def classify_relationships(pairs: list[tuple[str, str]], model: str = "gemini-2.5-flash",
                            api_key: str | None = None, batch_size: int = 25,
                            max_workers: int = 5) -> dict[tuple[str, str], str | None]:
    """Classify the semantic relationship of every (genus_a, genus_b) pair,
    batching multiple pairs per LLM call and running batches concurrently."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    indexed = list(enumerate(pairs))
    batches = [indexed[i:i + batch_size] for i in range(0, len(indexed), batch_size)]

    index_to_label: dict[int, str | None] = {}

    def work(batch):
        batch_input = [(idx, a, b) for idx, (a, b) in batch]
        return _classify_relationship_batch(batch_input, model, api_key)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(work, b): b for b in batches}
        done = 0
        for fut in as_completed(futures):
            index_to_label.update(fut.result())
            done += 1
            print(f"[genus_similarity] LLM relationship batches: {done}/{len(batches)} done", end="\r")
    if batches:
        print()

    return {pairs[i]: index_to_label.get(i) for i in range(len(pairs))}


def add_llm_relationships(results: list["PairwiseResult"], model: str = "gemini-2.5-flash",
                           api_key: str | None = None, batch_size: int = 25, max_workers: int = 5,
                           max_pairs: int | None = None, skip_normalization_duplicates: bool = True) -> None:
    """Mutates `results` in place, filling in each PairwiseResult's
    llm_relationship. Pairs already known-identical after normalization
    (genus_1_normalized == genus_2_normalized) are assigned EQUIVALENT
    directly, at zero API cost -- they're provably equivalent already.

    max_pairs caps how many of the REMAINING (non-trivial) pairs get sent to
    the LLM, taken in existing list order, to bound cost/time on a large
    genus list. Pairs beyond the cap are left with llm_relationship=None."""
    pairs_to_query: list[tuple[str, str]] = []
    query_indices: list[int] = []
    for i, r in enumerate(results):
        if skip_normalization_duplicates and r.genus_1_normalized == r.genus_2_normalized:
            r.llm_relationship = "EQUIVALENT"
            continue
        pairs_to_query.append((r.genus_1, r.genus_2))
        query_indices.append(i)

    if max_pairs is not None and len(pairs_to_query) > max_pairs:
        print(f"[genus_similarity] {len(pairs_to_query)} pairs need LLM classification; "
              f"capping to the first {max_pairs} (raise with --llm-max-pairs).")
        pairs_to_query = pairs_to_query[:max_pairs]
        query_indices = query_indices[:max_pairs]

    if not pairs_to_query:
        return

    n_batches = (len(pairs_to_query) + batch_size - 1) // batch_size
    print(f"[genus_similarity] Classifying {len(pairs_to_query)} pairs via LLM "
          f"({n_batches} batched calls, {max_workers} concurrent)...")
    rel_map = classify_relationships(pairs_to_query, model=model, api_key=api_key,
                                      batch_size=batch_size, max_workers=max_workers)
    for i, pair in zip(query_indices, pairs_to_query):
        results[i].llm_relationship = rel_map.get(pair)


# --------------------------------------------------------------------------
# Pairwise computation + output
# --------------------------------------------------------------------------

@dataclass
class PairwiseResult:
    genus_1: str
    genus_2: str
    genus_1_normalized: str
    genus_2_normalized: str
    edit_similarity: float
    jaccard_similarity: float
    llm_relationship: str | None = None  # filled in later by add_llm_relationships, if --llm used


def compute_pairwise_similarity(genus_values: list[str], singularize: bool = True) -> list[PairwiseResult]:
    """All i<j pairs among the given (assumed-distinct) genus strings."""
    normalized = {g: normalize_genus(g, singularize=singularize) for g in genus_values}
    results = []
    for g1, g2 in combinations(genus_values, 2):
        results.append(PairwiseResult(
            genus_1=g1, genus_2=g2,
            genus_1_normalized=normalized[g1], genus_2_normalized=normalized[g2],
            edit_similarity=round(edit_similarity(g1, g2, singularize=singularize), 4),
            jaccard_similarity=round(jaccard_similarity(g1, g2, singularize=singularize), 4),
        ))
    return results


def find_normalization_duplicates(genus_values: list[str], singularize: bool = True) -> dict[str, list[str]]:
    """Groups of DIFFERENT raw genus strings that normalize to the exact
    same form (e.g. "A Financial Instruments" and "the financial
    instrument" both -> "financial instrument"). Only returns groups with
    2+ members. This is the crispest possible signal of redundancy --
    stronger than a merely-high similarity score."""
    groups: dict[str, list[str]] = {}
    for g in genus_values:
        norm = normalize_genus(g, singularize=singularize)
        groups.setdefault(norm, []).append(g)
    return {norm: members for norm, members in groups.items() if len(members) > 1}


def write_pairwise_csv(results: list[PairwiseResult], out_path: str | Path) -> None:
    """genus_1 plays "Genus A" and genus_2 plays "Genus B" for the
    llm_relationship column -- A_SUBSUMES_B means genus_1 subsumes genus_2."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["genus_1", "genus_2", "genus_1_normalized", "genus_2_normalized",
                          "edit_similarity", "jaccard_similarity", "llm_relationship"])
        for r in results:
            writer.writerow([r.genus_1, r.genus_2, r.genus_1_normalized, r.genus_2_normalized,
                              r.edit_similarity, r.jaccard_similarity, r.llm_relationship or ""])


def write_matrix_csv(genus_values: list[str], results: list[PairwiseResult],
                      metric: str, out_path: str | Path) -> None:
    """Square matrix form (genus x genus), useful for a heatmap or for
    eyeballing everything at once in a spreadsheet. metric: "edit",
    "jaccard", or "llm". For "llm", the mirrored (reverse-order) cell is NOT
    just a copy -- A_SUBSUMES_B/B_SUBSUMES_A are swapped for the opposite
    orientation, since subsumption is directional; EQUIVALENT/RELATED/
    DIFFERENT are copied as-is since those are symmetric."""
    lookup = {}
    for r in results:
        if metric == "edit":
            val = r.edit_similarity
        elif metric == "jaccard":
            val = r.jaccard_similarity
        elif metric == "llm":
            val = r.llm_relationship or ""
        else:
            raise ValueError(f"Unknown metric {metric!r}; expected 'edit', 'jaccard', or 'llm'")
        lookup[(r.genus_1, r.genus_2)] = val
        lookup[(r.genus_2, r.genus_1)] = _LLM_MIRROR.get(val, val) if metric == "llm" else val

    diagonal = "EQUIVALENT" if metric == "llm" else 1.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([""] + genus_values)
        for g1 in genus_values:
            row = [g1]
            for g2 in genus_values:
                row.append(diagonal if g1 == g2 else lookup.get((g1, g2), ""))
            writer.writerow(row)


def summarize(results: list[PairwiseResult], high_threshold: float = 0.7) -> dict:
    """Quick numeric summary: are most pairs near-0 (diverse) or are there
    a lot of high-similarity pairs (redundant/near-duplicate genus phrasing)?"""
    n = len(results)
    if n == 0:
        return {"n_pairs": 0}
    edit_vals = [r.edit_similarity for r in results]
    jac_vals = [r.jaccard_similarity for r in results]
    summary = {
        "n_pairs": n,
        "edit_similarity_mean": round(sum(edit_vals) / n, 4),
        "edit_similarity_max": round(max(edit_vals), 4),
        "jaccard_similarity_mean": round(sum(jac_vals) / n, 4),
        "jaccard_similarity_max": round(max(jac_vals), 4),
        f"pairs_above_{high_threshold}_edit": sum(1 for v in edit_vals if v >= high_threshold),
        f"pairs_above_{high_threshold}_jaccard": sum(1 for v in jac_vals if v >= high_threshold),
    }

    llm_labels = [r.llm_relationship for r in results if r.llm_relationship]
    if llm_labels:
        from collections import Counter
        counts = Counter(llm_labels)
        summary["llm_relationship_counts"] = {label: counts.get(label, 0) for label in LLM_RELATIONSHIP_LABELS}
        n_unclassified = sum(1 for r in results if r.llm_relationship is None)
        if n_unclassified:
            summary["llm_relationship_unclassified"] = n_unclassified

    return summary


def analyze_genus_similarity(in_csv: str | Path, out_csv: str | Path,
                              include_suggested: bool = False,
                              edit_matrix_csv: str | Path | None = None,
                              jaccard_matrix_csv: str | Path | None = None,
                              llm_matrix_csv: str | Path | None = None,
                              high_threshold: float = 0.7,
                              singularize: bool = True,
                              use_llm: bool = False,
                              llm_model: str = "gemini-2.5-flash",
                              llm_batch_size: int = 25,
                              llm_workers: int = 5,
                              llm_max_pairs: int | None = None,
                              api_key: str | None = None) -> dict:
    rows = load_genus_rows(in_csv, include_suggested=include_suggested)
    if not rows:
        raise ValueError(f"No genus values found in {in_csv} "
                          f"(need is_aristotelian == 'yes' rows, or pass include_suggested=True)")

    distinct_genus = sorted({r.genus for r in rows}, key=str.lower)
    results = compute_pairwise_similarity(distinct_genus, singularize=singularize)

    if use_llm:
        add_llm_relationships(results, model=llm_model, api_key=api_key,
                               batch_size=llm_batch_size, max_workers=llm_workers,
                               max_pairs=llm_max_pairs)

    write_pairwise_csv(results, out_csv)

    if edit_matrix_csv:
        write_matrix_csv(distinct_genus, results, "edit", edit_matrix_csv)
    if jaccard_matrix_csv:
        write_matrix_csv(distinct_genus, results, "jaccard", jaccard_matrix_csv)
    if llm_matrix_csv and use_llm:
        write_matrix_csv(distinct_genus, results, "llm", llm_matrix_csv)

    dup_groups = find_normalization_duplicates(distinct_genus, singularize=singularize)

    summary = summarize(results, high_threshold=high_threshold)
    summary["n_terms"] = len(rows)
    summary["n_distinct_genus_phrases"] = len(distinct_genus)
    summary["n_normalization_duplicate_groups"] = len(dup_groups)
    summary["normalization_duplicate_groups"] = dup_groups
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Compute pairwise edit-distance, Jaccard, and (optionally) LLM-judged "
                    "semantic relationship between all genus values.")
    ap.add_argument("input", help="Path to definitions_classified.csv (from aristotelian_classifier.py).")
    ap.add_argument("--out", default="genus_pairwise_similarity.csv", help="Long-format pairwise output CSV.")
    ap.add_argument("--edit-matrix", default=None, help="Optional square-matrix CSV for edit similarity.")
    ap.add_argument("--jaccard-matrix", default=None, help="Optional square-matrix CSV for Jaccard similarity.")
    ap.add_argument("--llm-matrix", default=None, help="Optional square-matrix CSV for LLM relationships (requires --llm).")
    ap.add_argument("--include-suggested", action="store_true",
                     help="Also include suggested_genus from non-Aristotelian rows.")
    ap.add_argument("--top", type=int, default=15, help="Print the top N most similar pairs by each metric.")
    ap.add_argument("--high-threshold", type=float, default=0.7,
                     help="Similarity threshold for the 'near-duplicate pair count' summary stat.")
    ap.add_argument("--no-singularize", action="store_true",
                     help="Disable head-noun singularization in the normalization step "
                          "(lowercasing, punctuation stripping, and leading-article removal still apply).")
    ap.add_argument("--llm", action="store_true",
                     help="Also classify each pair's semantic relationship via Gemini "
                          "(EQUIVALENT / A_SUBSUMES_B / B_SUBSUMES_A / RELATED / DIFFERENT). "
                          "Costs API calls -- roughly n*(n-1)/2 pairs, batched to reduce call count. "
                          "Pairs already identical after normalization are classified EQUIVALENT for free.")
    ap.add_argument("--llm-model", default="gemini-2.5-flash", help="Gemini model for the LLM relationship method.")
    ap.add_argument("--llm-batch-size", type=int, default=25,
                     help="Number of pairs bundled into each LLM call (fewer calls, cheaper per pair).")
    ap.add_argument("--llm-workers", type=int, default=5, help="Concurrent LLM batch requests.")
    ap.add_argument("--llm-max-pairs", type=int, default=None,
                     help="Cap the number of (non-trivial) pairs sent to the LLM, to bound cost/time "
                          "on a large genus list. Pairs beyond the cap are left unclassified.")
    ap.add_argument("--api-key", default=None, help="Gemini API key (else reads GEMINI_API_KEY/GOOGLE_API_KEY env var).")
    args = ap.parse_args()
    singularize = not args.no_singularize

    # Compute everything ONCE here (rather than calling analyze_genus_similarity, which
    # would need its own pass) so a costly --llm run never happens twice.
    rows = load_genus_rows(args.input, include_suggested=args.include_suggested)
    if not rows:
        raise SystemExit(f"No genus values found in {args.input} "
                          f"(need is_aristotelian == 'yes' rows, or pass --include-suggested)")
    distinct_genus = sorted({r.genus for r in rows}, key=str.lower)
    results = compute_pairwise_similarity(distinct_genus, singularize=singularize)

    if args.llm:
        add_llm_relationships(results, model=args.llm_model, api_key=args.api_key,
                               batch_size=args.llm_batch_size, max_workers=args.llm_workers,
                               max_pairs=args.llm_max_pairs)

    write_pairwise_csv(results, args.out)
    if args.edit_matrix:
        write_matrix_csv(distinct_genus, results, "edit", args.edit_matrix)
    if args.jaccard_matrix:
        write_matrix_csv(distinct_genus, results, "jaccard", args.jaccard_matrix)
    if args.llm_matrix and args.llm:
        write_matrix_csv(distinct_genus, results, "llm", args.llm_matrix)

    dup_groups = find_normalization_duplicates(distinct_genus, singularize=singularize)
    summary = summarize(results, high_threshold=args.high_threshold)
    summary["n_terms"] = len(rows)
    summary["n_distinct_genus_phrases"] = len(distinct_genus)
    summary["n_normalization_duplicate_groups"] = len(dup_groups)

    print(f"\n{summary['n_terms']} terms -> {summary['n_distinct_genus_phrases']} distinct genus phrases "
          f"-> {summary['n_pairs']} pairs\n")
    for k, v in summary.items():
        if k not in ("n_terms", "n_distinct_genus_phrases", "n_pairs",
                     "llm_relationship_counts", "llm_relationship_unclassified"):
            print(f"  {k}: {v}")

    if "llm_relationship_counts" in summary:
        print("\n  LLM relationship counts:")
        for label, count in summary["llm_relationship_counts"].items():
            print(f"    {label}: {count}")
        if "llm_relationship_unclassified" in summary:
            print(f"    (unclassified: {summary['llm_relationship_unclassified']})")

    if dup_groups:
        print(f"\n{len(dup_groups)} group(s) of genus phrases that are IDENTICAL after normalization "
              f"(these score exactly 1.0/1.0 on both lexical metrics -- the strongest possible "
              f"redundancy signal):")
        for norm, members in dup_groups.items():
            print(f"  {norm!r}  <-  {members}")

    if args.top:
        print(f"\nTop {args.top} pairs by edit similarity:")
        for r in sorted(results, key=lambda r: -r.edit_similarity)[:args.top]:
            extra = f"  [{r.llm_relationship}]" if r.llm_relationship else ""
            print(f"  {r.edit_similarity:.3f}  {r.genus_1!r} <-> {r.genus_2!r}{extra}")

        print(f"\nTop {args.top} pairs by Jaccard similarity:")
        for r in sorted(results, key=lambda r: -r.jaccard_similarity)[:args.top]:
            extra = f"  [{r.llm_relationship}]" if r.llm_relationship else ""
            print(f"  {r.jaccard_similarity:.3f}  {r.genus_1!r} <-> {r.genus_2!r}{extra}")

        if args.llm:
            # The whole point of the LLM method is catching pairs the lexical
            # metrics miss -- so also show pairs the LLM found meaningfully
            # related despite low lexical similarity.
            interesting = [r for r in results
                           if r.llm_relationship in ("EQUIVALENT", "A_SUBSUMES_B", "B_SUBSUMES_A")
                           and r.edit_similarity < 0.4 and r.jaccard_similarity < 0.4]
            if interesting:
                print(f"\nPairs the LLM flagged as related despite LOW lexical similarity "
                      f"(this is what the lexical-only metrics miss):")
                for r in interesting[:args.top]:
                    print(f"  [{r.llm_relationship}]  {r.genus_1!r} <-> {r.genus_2!r}  "
                          f"(edit={r.edit_similarity:.2f}, jaccard={r.jaccard_similarity:.2f})")

    written = [args.out]
    for path in (args.edit_matrix, args.jaccard_matrix, args.llm_matrix if args.llm else None):
        if path:
            written.append(path)
    print(f"\nWrote {', '.join(written)}")
