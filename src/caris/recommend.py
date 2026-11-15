"""The three recommenders of the previous system, with two fixes.

Each compares an image's predicted codes with the codes of every artwork in a collection,
and returns its best-scoring artwork, or None when no artwork shares anything with the
query. Codes are compared without their bracketed text: 25F23(LION) is read as 25F23, and
11H(PAUL)6 as 11H6. Each distinct query code counts once.

The previous system cut every code at its first bracket, so 11H(PAUL)6, the martyrdom of
Paul, became 11H, any male saint. It also counted a query code as often as it occurred
after that cut, so several saints' codes added 11H several times.
"""

import math
import re

BRACKETED = re.compile(r"\([^()]*\)")


def recommend(collection: dict[str, list[str]], codes: list[str], idf_impact: float = 1.0) -> dict[str, str | None]:
    """Each recommender's pick from the collection, which maps an artwork to its codes."""
    return {
        "prefix": by_prefix(collection, codes),
        "idf": by_idf(collection, codes, idf_impact),
        "jaccard": by_jaccard(collection, codes),
    }


def by_prefix(collection, codes):
    """Each query code adds its best similarity to the artwork's codes."""
    query_codes = _query_codes(codes)
    scores = {}
    for artwork, artwork_codes in collection.items():
        artwork_codes = [_base_code(code) for code in artwork_codes]
        scores[artwork] = sum(
            max((_code_similarity(query, code) for code in artwork_codes), default=0)
            for query in query_codes
        )
    return _best(scores)


def idf_scores(collection) -> dict[str, float]:
    """ln((N + 1) / (df + 1)) + 1 for each code, where df counts the artworks that have it."""
    artworks = len(collection)
    document_counts = {}
    for artwork_codes in collection.values():
        for code in {_base_code(code) for code in artwork_codes}:
            document_counts[code] = document_counts.get(code, 0) + 1
    return {code: math.log((artworks + 1) / (count + 1)) + 1 for code, count in document_counts.items()}


def by_idf(collection, codes, idf_impact=1.0):
    """Each query code that the artwork has adds its IDF, raised to idf_impact."""
    idf = idf_scores(collection)
    query_codes = _query_codes(codes)
    scores = {}
    for artwork, artwork_codes in collection.items():
        artwork_codes = {_base_code(code) for code in artwork_codes}
        scores[artwork] = sum(idf.get(code, 0) ** idf_impact for code in query_codes if code in artwork_codes)
    return _best(scores)


def by_jaccard(collection, codes):
    """Shared codes divided by all the codes in either set."""
    query_codes = {_base_code(code) for code in codes}
    scores = {}
    for artwork, artwork_codes in collection.items():
        artwork_codes = {_base_code(code) for code in artwork_codes}
        shared = query_codes & artwork_codes
        scores[artwork] = len(shared) / len(query_codes | artwork_codes) if shared else 0
    return _best(scores)


def _query_codes(codes):
    """The query's distinct base codes, in the order they first occur. Summing in one fixed order
    gives artworks that share the same codes exactly equal scores, so ties go to collection order."""
    return list(dict.fromkeys(map(_base_code, codes)))


def _base_code(code):
    """The code without its bracketed text: 25F23(LION) gives 25F23, and 11H(PAUL)6 gives 11H6."""
    while (shorter := BRACKETED.sub("", code)) != code:  # repeated, for brackets inside brackets
        code = shorter
    return code.split("(")[0].strip()  # a bracket that is never closed runs to the end


def _code_similarity(query, code):
    """1 for the same code, lower for codes that only match after trimming trailing characters."""
    if query == code:
        return 1
    for trim, score in ((1, 0.5), (2, 0.25)):
        query_prefix, code_prefix = query[:-trim], code[:-trim]
        # A code too short to trim leaves an empty prefix, which says nothing about relatedness.
        if query_prefix and query_prefix in (code_prefix, code):
            return score
        if code_prefix and code_prefix == query:
            return score
    return 0


def _best(scores):
    best = max(scores, key=scores.get, default=None)
    return best if best is not None and scores[best] > 0 else None
