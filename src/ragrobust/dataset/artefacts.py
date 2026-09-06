"""Generation-artefact detection for the conflict testbed (P1 Section 5.9).

P1 5.9 lists three mitigations for the circularity threat. The first two are
implemented elsewhere -- disjoint model families (`providers/factory.py`) and
judge-versus-human agreement (`judge.py`). This module is the third:

    "The manual-validation pass on a stratified sample checks that generated
    cases do not contain detectable generation artefacts (such as templated
    phrasing that signals the perturbation)."

DEVIATION, and it must be stated in Chapter 6. P1 places this check inside the
human annotator's checklist. It is not there -- `dataset/validation.CHECKLISTS`
asks about contradiction, plausibility and topicality, but never about
templated phrasing -- and the annotation cannot be redone before submission.
What follows is an automated substitute measuring the same property. It is
weaker than a human read in one respect (it cannot notice an artefact nobody
thought to count) and stronger in another (it inspects all 198 generated
passages rather than a stratified sample of them).

Why the property matters. Each conflict case carries the original passage and a
model-written contradiction of it. If the generated passage is recognisable
from surface form alone -- a stock opening, a giveaway connective, a different
length -- then a pipeline could in principle learn to pick the human-written
side without doing any of the reasoning the Conflict Resolution Score is meant
to measure, and RQ2 would be measuring style detection.

Three measurements, each comparing generated passages against the ORIGINALS
they were derived from. The comparison is the measurement: Wikipedia prose
repeats itself too, so an absolute count of repeated phrasing says nothing
without the human-written baseline beside it.

  1. Edit localisation. The generator is asked to rewrite a passage changing
     only the answering claim, so a small, localised edit is the DESIRED
     outcome -- it means the two sides of the conflict differ in the claim and
     nothing else. A large edit fraction means a wholesale rewrite, which is
     where style artefacts enter. This doubles as the standing verification of
     the locked decision that conflict generation stays passage-level.
  2. Cross-case repeated n-grams. Templated phrasing shows up as the same
     n-gram recurring across otherwise unrelated cases.
  3. Generated-only vocabulary. Tokens the model reaches for that the source
     corpus does not use.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

# Contraction-preserving word tokenizer. Deliberately simple: the measurement is
# a RATIO between two corpora processed identically, so tokenizer sophistication
# cannot change the comparison, only the units it is expressed in.
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

# Wikipedia markup survives into the passages (evidence is prepared, not
# filtered), and it is present in BOTH sides of every pair. Left in rather than
# stripped: removing it would change the denominator for the original passages
# and the generated ones by different amounts, since a rewrite does not
# reproduce markup at the same rate.


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def token_edit_fraction(original: str, generated: str) -> float:
    """How much of the passage the generator changed, in [0, 1].

    ``1 - ratio`` of `difflib.SequenceMatcher` over token sequences, where the
    ratio is ``2 * matched / (len(a) + len(b))``. 0.0 means the passages are
    token-identical; 1.0 means they share nothing.

    Token-level rather than character-level so that a one-word substitution in
    a long passage scores as one token changed, not as the character span it
    happens to occupy.
    """
    a, b = tokenize(original), tokenize(generated)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """The DISTINCT n-grams of a token sequence.

    A set, not a list: a phrase repeated twice inside one passage is one
    passage's worth of evidence about templating, not two.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def cross_document_ngrams(
    documents: list[str], n: int, *, min_documents: int = 2
) -> dict[tuple[str, ...], int]:
    """n-grams that appear in at least `min_documents` different documents.

    Document frequency, never total frequency. Templating is a claim about
    phrasing shared BETWEEN unrelated cases; a passage that repeats a phrase
    internally is just prose.
    """
    counts: Counter[tuple[str, ...]] = Counter()
    for doc in documents:
        counts.update(ngrams(tokenize(doc), n))
    return {g: c for g, c in counts.items() if c >= min_documents}


def over_represented_tokens(
    generated: list[str],
    original: list[str],
    *,
    min_count: int = 5,
    top_k: int = 25,
) -> tuple[list[tuple[str, float, int]], list[tuple[str, int]]]:
    """Tokens the generator uses far more often than the source corpus does.

    Returns `(over_represented, generated_only)`.

    `over_represented` holds `(token, rate_ratio, generated_count)` where the
    ratio is the token's rate in generated text divided by its rate in original
    text. No smoothing: a token absent from the originals has an undefined
    ratio, not a large one, so it is reported separately in `generated_only`
    instead of being given a fabricated denominator. Generated-only tokens are
    the stronger artefact signal of the two and deserve to be read on their own.
    """
    gen_counts, orig_counts = Counter(generated), Counter(original)
    gen_total, orig_total = len(generated) or 1, len(original) or 1

    over: list[tuple[str, float, int]] = []
    only: list[tuple[str, int]] = []
    for token, count in gen_counts.items():
        if count < min_count:
            continue
        if orig_counts[token] == 0:
            only.append((token, count))
            continue
        ratio = (count / gen_total) / (orig_counts[token] / orig_total)
        if ratio > 1.0:
            over.append((token, ratio, count))

    over.sort(key=lambda x: (-x[1], x[0]))
    only.sort(key=lambda x: (-x[1], x[0]))
    return over[:top_k], only[:top_k]


def generated_only_document_share(
    originals: list[str], generated: list[str], *, min_documents: int = 2
) -> list[tuple[str, float, int]]:
    """Tokens absent from every original, ranked by how many cases use them.

    `(token, share_of_generated_documents, document_count)`.

    Separate from `over_represented_tokens`, and gated on DOCUMENT frequency
    rather than total count, because the two answer different questions. A
    token used forty times inside one passage is that passage's subject matter.
    A token used once each in forty different passages is a template. An
    absolute-count filter cannot tell those apart, and the first version of this
    module used one -- a stock phrase spread thinly across every case slipped
    through it entirely.

    `min_documents` defaults to 2, and dropping below it would break the check
    on its own design. Every contradiction replaces a claim, so the replacement
    value -- "1771" where the original said "1769" -- is a generated-only token
    in exactly one case, always, by construction. Those are the perturbation
    working, not a template. Templating is a claim about phrasing shared BETWEEN
    cases, so a token confined to one case cannot be evidence of it however
    small the corpus is.
    """
    original_vocab = {t for doc in originals for t in tokenize(doc)}
    if not generated:
        return []
    doc_counts: Counter[str] = Counter()
    for doc in generated:
        doc_counts.update(set(tokenize(doc)) - original_vocab)
    n = len(generated)
    out = [(t, c / n, c) for t, c in doc_counts.items() if c >= min_documents]
    out.sort(key=lambda x: (-x[2], x[0]))
    return out


@dataclass
class ArtefactReport:
    """The generation-artefact measurement over all conflict pairs."""

    n_pairs: int
    # 1. Edit localisation
    mean_edit_fraction: float
    median_edit_fraction: float
    max_edit_fraction: float
    n_pairs_over_half_rewritten: int
    # 2. Cross-case repeated phrasing
    ngram_sizes: tuple[int, ...]
    cross_case_ngrams_generated: dict[int, int]
    cross_case_ngrams_original: dict[int, int]
    # 3. Vocabulary
    over_represented: list[tuple[str, float, int]]
    generated_only: list[tuple[str, int]]
    generated_only_by_case_share: list[tuple[str, float, int]]
    generated_only_max_case_share: float
    # Length
    mean_length_ratio: float
    verdict: str
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_pairs": self.n_pairs,
            "edit_localisation": {
                "mean_edit_fraction": round(self.mean_edit_fraction, 4),
                "median_edit_fraction": round(self.median_edit_fraction, 4),
                "max_edit_fraction": round(self.max_edit_fraction, 4),
                "n_pairs_over_half_rewritten": self.n_pairs_over_half_rewritten,
                "note": (
                    "A SMALL edit fraction is the desired outcome: the generator "
                    "is asked to change only the answering claim, so a localised "
                    "edit means the two sides of the conflict differ in that "
                    "claim and nothing else."
                ),
            },
            "cross_case_repeated_phrasing": {
                "ngram_sizes": list(self.ngram_sizes),
                "generated": self.cross_case_ngrams_generated,
                "original": self.cross_case_ngrams_original,
                "note": (
                    "n-grams shared by two or more different cases. The original "
                    "column is the baseline: human-written Wikipedia prose "
                    "repeats itself too, so only the comparison is meaningful."
                ),
            },
            "vocabulary": {
                "over_represented": [
                    {"token": t, "rate_ratio": round(r, 3), "generated_count": c}
                    for t, r, c in self.over_represented
                ],
                "generated_only": [
                    {"token": t, "generated_count": c} for t, c in self.generated_only
                ],
                "generated_only_by_case_share": [
                    {"token": t, "share_of_cases": share, "n_cases": c}
                    for t, share, c in self.generated_only_by_case_share
                ],
                "generated_only_max_case_share": round(self.generated_only_max_case_share, 4),
            },
            "mean_length_ratio_generated_over_original": round(self.mean_length_ratio, 4),
            "verdict": self.verdict,
            "flags": list(self.flags),
        }


def analyse_generation_artefacts(
    pairs: list[tuple[str, str]],
    *,
    ngram_sizes: tuple[int, ...] = (4, 5, 6, 7, 8),
    top_k_spread: int = 25,
    template_case_share: float = 0.25,
    wholesale_rewrite_fraction: float = 0.5,
    ngram_excess_ratio: float = 2.0,
) -> ArtefactReport:
    """Run all three measurements over `(original_text, generated_text)` pairs.

    The thresholds are mechanical and stated in the signature so the verdict is
    reproducible rather than a judgement call:

      * a token absent from every original but present in more than
        `template_case_share` of generated passages is a template marker;
      * a mean edit fraction above `wholesale_rewrite_fraction` means the
        generator rewrote rather than edited;
      * cross-case n-gram counts more than `ngram_excess_ratio` times the
        original baseline mean repeated phrasing beyond what the source corpus
        already exhibits.

    A clean verdict is a claim about these three things and nothing else. It is
    not a claim that no artefact exists -- an automated check cannot make that
    claim, and Chapter 6 should not either.
    """
    if not pairs:
        raise ValueError("no conflict pairs supplied")

    edits = [token_edit_fraction(o, g) for o, g in pairs]
    ordered = sorted(edits)
    mid = len(ordered) // 2
    median = (ordered[mid] if len(ordered) % 2
              else (ordered[mid - 1] + ordered[mid]) / 2.0)

    originals = [o for o, _ in pairs]
    generated = [g for _, g in pairs]
    gen_tokens = [t for g in generated for t in tokenize(g)]
    orig_tokens = [t for o in originals for t in tokenize(o)]

    cross_gen = {n: len(cross_document_ngrams(generated, n)) for n in ngram_sizes}
    cross_orig = {n: len(cross_document_ngrams(originals, n)) for n in ngram_sizes}

    over, only = over_represented_tokens(gen_tokens, orig_tokens)

    # How widely is the most common generated-only token spread across cases?
    # One passage using an odd word is prose; a quarter of them using it is a
    # template. Measured over document frequency, independently of the
    # `min_count` that gates the vocabulary table -- see
    # `generated_only_document_share` for why the two must not share a filter.
    spread = generated_only_document_share(originals, generated)
    max_share = max((share for _, share, _ in spread), default=0.0)
    top_spread = [(t, round(share, 4), c) for t, share, c in spread[:top_k_spread]]

    lengths = [
        (len(tokenize(g)) / len(tokenize(o))) if tokenize(o) else 1.0
        for o, g in pairs
    ]

    flags: list[str] = []
    if max_share > template_case_share:
        flags.append(
            f"a token absent from every original passage appears in "
            f"{max_share:.1%} of generated passages"
        )
    mean_edit = sum(edits) / len(edits)
    if mean_edit > wholesale_rewrite_fraction:
        flags.append(
            f"mean edit fraction {mean_edit:.3f} indicates wholesale rewriting "
            f"rather than a localised claim change"
        )
    for n in ngram_sizes:
        if cross_orig[n] and cross_gen[n] > ngram_excess_ratio * cross_orig[n]:
            flags.append(
                f"{n}-grams shared across cases: {cross_gen[n]} generated vs "
                f"{cross_orig[n]} original"
            )

    verdict = (
        "no generation artefact detected by these three measurements"
        if not flags else
        "possible generation artefact -- see flags"
    )

    return ArtefactReport(
        n_pairs=len(pairs),
        mean_edit_fraction=mean_edit,
        median_edit_fraction=median,
        max_edit_fraction=max(edits),
        n_pairs_over_half_rewritten=sum(1 for e in edits if e > wholesale_rewrite_fraction),
        ngram_sizes=ngram_sizes,
        cross_case_ngrams_generated=cross_gen,
        cross_case_ngrams_original=cross_orig,
        over_represented=over,
        generated_only=only,
        generated_only_by_case_share=top_spread,
        generated_only_max_case_share=max_share,
        mean_length_ratio=sum(lengths) / len(lengths),
        verdict=verdict,
        flags=flags,
    )
