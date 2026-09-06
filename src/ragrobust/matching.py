"""Answer-string matching, shared by dataset construction and scoring.

One primitive, in one place, because the same question -- "does this text assert
this answer?" -- is asked when flagging a passage answer-bearing, when guarding
against answer leaks, and when grading a model's response. Those had drifted
into three separate bare-substring tests, and a bare substring is wrong in the
same way in all three.

The failure that motivated this: a TriviaQA seed with the gold answer
"King George I" was flagged answer-bearing because its passage contained
"King George III". The passage was about George III surrendering Crown revenues
and did not answer who succeeded Queen Anne. The model refused, correctly, and
would have been scored wrong. The same defect in `contains_answer` graded a
prediction of "King George III" as CORRECT against that gold -- inflating
accuracy rather than deflating it.

Word boundaries fix both directions while preserving the short-answer
convention P1 Section 5.4.3 relies on: "it was completed in 1889" still matches
a gold of "1889", because that match sits on word boundaries.
"""

from __future__ import annotations

import re
import unicodedata


def strip_accents(text: str) -> str:
    """Fold accents so 'Bogota' matches 'Bogotá'.

    The seed loaders store some answers with accents removed, so an accented
    passage and its unaccented gold answer would otherwise never match. Folding
    both sides is the conservative fix; it cannot create a false match between
    two genuinely different words.
    """
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def contains_on_word_boundary(needle: str, haystack: str) -> bool:
    """Whether `needle` occurs in `haystack` delimited by non-word characters.

    Case-insensitive and accent-folded. Lookarounds rather than `\\b` because
    `\\b` is defined against the pattern's own edge characters: a needle ending
    in punctuation, such as "St." or "Ph.D.", has a non-word character at the
    boundary and `\\b` would fail to anchor where a reader expects.
    """
    n = strip_accents(needle.strip().lower())
    if not n:
        return False
    h = strip_accents(haystack.lower())
    return re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", h) is not None
