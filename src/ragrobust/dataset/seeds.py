"""Seed sampling and filtering.

P1 Section 5.5.1 samples roughly 300 examples from each of Natural Questions and
TriviaQA, after filtering out ambiguous answers, multi-answer fields, and
evidence documents that are extremely long or short. Two seed datasets are used
so that the benchmark does not over-fit the characteristics of a single source.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..matching import contains_on_word_boundary
from ..schema import Passage, SeedSource


@dataclass
class Seed:
    """A filtered seed question-answer pair with its evidence passages."""

    seed_id: str
    source: SeedSource
    query: str
    answer: str
    passages: list[Passage]
    # Same-document windows that are NOT part of this seed's evidence. They exist
    # only to stock the shared distractor corpus (see `extract_context_windows`)
    # and are deliberately kept out of `passages` so they never count as signal
    # tokens, never satisfy the answerable flag, and never widen the refusal
    # perturbation's removal target.
    extra_passages: list[Passage] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def answer_passages(self) -> list[Passage]:
        return [p for p in self.passages if p.is_answer_bearing]

    @property
    def non_answer_passages(self) -> list[Passage]:
        return [p for p in self.passages if not p.is_answer_bearing]


class SeedFilterStats(dict):
    """Counts of why seeds were rejected. Reported in Chapter 6."""


def filter_seeds(
    seeds: list[Seed],
    *,
    drop_ambiguous_answers: bool = True,
    drop_multi_answer: bool = True,
    min_evidence_tokens: int = 50,
    max_evidence_tokens: int = 2000,
) -> tuple[list[Seed], SeedFilterStats]:
    """Apply the P1 Section 5.5.1 exclusions.

    Returns the surviving seeds and a rejection breakdown, because the acceptance
    rate is a reportable property of the benchmark rather than an internal detail.
    """
    stats = SeedFilterStats(
        total=len(seeds),
        no_answer_passage=0,
        ambiguous_answer=0,
        multi_answer=0,
        evidence_too_short=0,
        evidence_too_long=0,
        kept=0,
    )
    out: list[Seed] = []

    for s in seeds:
        # A seed with no answer-bearing passage cannot anchor any perturbation.
        if not s.answer_passages:
            stats["no_answer_passage"] += 1
            continue

        ans = s.answer.strip()
        if drop_ambiguous_answers and (not ans or len(ans) < 2):
            stats["ambiguous_answer"] += 1
            continue

        # Multi-answer fields make correctness ill-defined, and make the conflict
        # perturbation ambiguous because there is no single claim to contradict.
        if drop_multi_answer and any(sep in ans for sep in (";", " | ", "||")):
            stats["multi_answer"] += 1
            continue

        n_tokens = sum(p.token_estimate() for p in s.passages)
        if n_tokens < min_evidence_tokens:
            stats["evidence_too_short"] += 1
            continue
        if n_tokens > max_evidence_tokens:
            stats["evidence_too_long"] += 1
            continue

        out.append(s)
        stats["kept"] += 1

    return out, stats


def sample_seeds(seeds: list[Seed], n: int, seed: int) -> list[Seed]:
    """Deterministic sample. The RNG seed is recorded in gen_params."""
    rng = random.Random(seed)
    if n >= len(seeds):
        return list(seeds)
    return rng.sample(seeds, n)



# --------------------------------------------------------------------------
# Evidence preparation
#
# configs/dataset.yaml caps evidence at 800 whitespace tokens, because a noise
# case at ratio 0.90 carries ten times its signal and must still fit the
# max_context_tokens=8192 control. Raw NQ and TriviaQA documents are far longer,
# so evidence is PREPARED to fit rather than filtered out: filtering would
# reject nearly every TriviaQA seed and collapse the yield.
# --------------------------------------------------------------------------

DEFAULT_EVIDENCE_TOKENS = 700

# Namespaced dataset repositories. The bare canonical names ("natural_questions",
# "trivia_qa") no longer resolve: huggingface_hub requires 'namespace/name'.
NQ_DATASET = "google-research-datasets/natural_questions"
TRIVIAQA_DATASET = "mandarjoshi/trivia_qa"


def extract_answer_window(text: str, answer: str, max_tokens: int) -> str | None:
    """Return at most `max_tokens` words of `text`, centred on the answer.

    Returns None when the answer does not occur in the text at all, which is the
    signal to drop the seed rather than store a passage that does not support
    the answer it claims to.

    Centring matters. Taking a document's opening `max_tokens` words is what the
    previous loader did, and it silently produced "answerable" cases whose
    answer passage did not contain the answer -- the flag was set from the
    document while the text stored was a prefix of it. Every perturbation
    downstream trusts that flag, so the corruption would have surfaced only as
    inexplicably poor accuracy on the answerable controls.
    """
    needle = answer.strip().lower()
    if not needle:
        return None
    lowered = text.lower()
    position = lowered.find(needle)
    if position < 0:
        return None

    words = text.split()
    if len(words) <= max_tokens:
        return " ".join(words)

    # Convert the character offset to a word index, then centre a window on it.
    answer_start_word = len(text[:position].split())
    answer_len_words = len(needle.split())
    start = max(0, answer_start_word - (max_tokens - answer_len_words) // 2)
    window = " ".join(words[start : start + max_tokens])

    if needle in window.lower():
        return window
    # The answer straddled the window edge; anchor the window at it instead.
    window = " ".join(words[answer_start_word : answer_start_word + max_tokens])
    return window if needle in window.lower() else None


# A tag name must follow the bracket, so a comparison written in prose
# ("a < b and c > d") is not mistaken for markup and deleted. Same shape as the
# review sheet's _TAG_RE in dataset/validation.py.
_MARKUP_RE = re.compile(r"</?[A-Za-z][^<>]{0,80}>")


def strip_markup_for_embedding(text: str) -> str:
    """Return `text` with Wikipedia markup removed, for EMBEDDING ONLY.

    Seed evidence is prepared, not filtered (HANDOFF locked decision), so NQ
    passages keep their raw `<P> <Table> <Tr> <Td>` token stream. Every pipeline
    therefore sees identical text and nothing here changes what is stored or
    served -- this transform exists solely to compute similarity, exactly as the
    review sheet's rendering exists solely to be read.

    It has to exist because the markup dominates the embedding. Measured on the
    built corpus, the "nearest neighbours" of a Virgin Australia passage were a
    music chart, a poker results table and an Olympic medal table, all scoring
    0.65-0.69 -- they were near-identical runs of `<Td></Tr>`, not near in topic.
    Stripping the tags puts all six genuine same-article neighbours in ranks 1-6
    at 0.78-0.83 and drops those three to ~0.53.

    Selecting distractors on unstripped similarity is what produced a noise
    testbed whose distractors failed P1 5.5.3's "topically related" review on
    every sampled case.
    """
    return re.sub(r"\s+", " ", _MARKUP_RE.sub(" ", text)).strip()


def _normalise_for_answer_match(text: str) -> str:
    """Markup-free, punctuation-free, single-spaced lower case."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ",
                                      strip_markup_for_embedding(text).lower())).strip()


def distractor_states_answer(text: str, answer: str) -> bool:
    """True if a candidate distractor asserts the gold answer.

    P1 5.5.3 asks the reviewer to confirm that "the distractor passages ... do
    not contain the answer", so a distractor that states it is not a distractor
    at all -- it makes the case trivially answerable and flattens the Noise
    Degradation Curve it was meant to bend.

    A raw `answer.lower() in passage.lower()` misses two whole classes, both
    found in the built testbed:

      * markup and spacing. "The Double" survives as "The <I>Double</I>" and
        NQ's tokenised text spaces punctuation out, so the substring never
        matches. Eighteen instances leaked this way.
      * inverted names. Wikipedia bibliographies render "Nicholas Sparks" as
        "Sparks , Nicholas", which shares no substring with the gold answer.

    The inversion test is restricted to two alphabetic tokens: reversing a
    numeric answer like "in 1978" matches the ordinary prose "1978 in".
    """
    hay = _normalise_for_answer_match(text)
    needle = _normalise_for_answer_match(answer)
    if not needle:
        return False
    if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay):
        return True

    tokens = answer.strip().split()
    if len(tokens) == 2 and all(t.isalpha() for t in tokens):
        inverted = _normalise_for_answer_match(" ".join(reversed(tokens)))
        if inverted != needle and re.search(rf"(?<!\w){re.escape(inverted)}(?!\w)", hay):
            return True

    # Third class: a DISTINCTIVE COMPONENT of a multi-word answer. Matching only
    # the full string let "Anakin" stand in for "Anakin Skywalker", "Leonardo"
    # for "Leonardo da Vinci" and "Wembley" for "Wembley Stadium" -- three leaks
    # found by hand in a 29-case sample of the built noise testbed, each of them
    # sitting in a passage about the query's own subject, so the answer was
    # plainly recoverable.
    #
    # Deliberately over-rejects. A distractor discarded for containing
    # "Phosphorus" when the answer is "Phosphorus pentoxide" costs one candidate
    # from a pool; a distractor kept because only part of the answer appears
    # makes the case trivially answerable and flattens the very curve the noise
    # dimension exists to bend. build_distractor_pool() takes the same stance.
    for part in _distinctive_components(answer):
        if re.search(rf"(?<!\w){re.escape(part)}(?!\w)", hay):
            return True
    return False


# Tokens too common to identify an answer on their own: a distractor mentioning
# "November" or "French" is not thereby stating "November 1999" or "Old French".
_COMMON_ANSWER_TOKENS = frozenset("""
january february march april may june july august september october november december
north south east west northern southern eastern western united great new old first last
saint mount lake river city county state island house world national international
john james david michael robert william mary anne peter paul george charles
""".split())


def _distinctive_components(answer: str) -> list[str]:
    """Tokens of a multi-word answer that could identify it on their own.

    Restricted to capitalised alphabetic tokens of five characters or more --
    surnames, place names, distinctive nouns -- with common given names, months,
    compass points and geographic fillers removed. A single-word answer has no
    components: the full-string test above already covers it.
    """
    tokens = answer.strip().split()
    if len(tokens) < 2:
        return []
    out: list[str] = []
    for raw in tokens:
        tok = re.sub(r"[^A-Za-z']", "", raw)
        if len(tok) < 5 or not tok.isalpha():
            continue
        # Capitalised in the source, or the whole answer is upper case
        # (TriviaQA renders golds like "ROBERT LOUIS STEVENSON").
        if not (raw[:1].isupper() or answer.isupper()):
            continue
        if tok.lower() in _COMMON_ANSWER_TOKENS:
            continue
        out.append(_normalise_for_answer_match(tok))
    return [o for o in out if o]


def extract_context_windows(
    text: str,
    answer: str,
    max_tokens: int,
    *,
    n_windows: int = 6,
    min_tokens: int = 50,
) -> list[str]:
    """Return answer-free windows from elsewhere in the same document.

    P1 Section 5.5.2 requires noise distractors to be "selected from the same
    corpus" and "topically related, in order to avoid the trivial case where
    distractors are obviously off-topic and easily ignored", and Section 5.5.2
    selects the refusal replacement by "retrieving topically nearest neighbours".
    Nearest-neighbour selection can only work if neighbours exist in the corpus.

    They did not. The corpus was one answer-centred window per seed -- 674
    passages across 591 seeds, 508 of them holding a single passage -- so each
    topic appeared exactly once and no seed had a topical neighbour to retrieve.
    Measured against the cached embeddings, the *best* available cosine was a
    median of 0.590 and 286 of 500 seeds had no candidate at all above 0.60, so
    raising the similarity floor emptied the pools instead of improving them.

    Other windows of the seed's own document are, by construction, the genuine
    topical neighbours, and they are what a real retriever returns alongside the
    answering chunk. Windows overlapping the answer are excluded so a distractor
    can never carry the answer it is meant to distract from; the caller applies
    the lexical answer check as well, and the embedding ceiling still drops
    near-duplicates of the answer passage.
    """
    words = text.split()
    if len(words) < min_tokens:
        return []

    needle = answer.strip().lower()
    out: list[str] = []
    for start in range(0, len(words), max_tokens):
        if len(out) >= n_windows:
            break
        window = " ".join(words[start : start + max_tokens])
        # Short tail fragments read as generation artefacts during manual
        # validation, which is exactly what P1 5.5.3 asks the annotator to catch.
        if len(window.split()) < min_tokens:
            continue
        # Bare substring, not word-boundary: for a distractor the conservative
        # test is the right one. Excluding "King George I" because the window
        # says "King George III" costs one candidate; admitting a window that
        # states the answer breaks the noise testbed's second review criterion.
        if needle and needle in window.lower():
            continue
        out.append(window)
    return out


def verify_seed_integrity(seed: Seed) -> list[str]:
    """Check that every answer-bearing flag matches the text actually stored.

    Cheap to run and worth running on every loaded seed: the schema validates
    that an answerable case HAS an answer-bearing passage, but it trusts the
    flag rather than reading the text, so a mislabelled passage passes silently.
    """
    problems: list[str] = []
    needle = seed.answer.strip().lower()
    if not needle:
        problems.append(f"{seed.seed_id}: empty answer")
        return problems
    for p in seed.passages:
        # Word-boundary, not bare substring: a gold answer of "King George I"
        # occurs inside "King George III", and the integrity gate exists
        # precisely to stop a flag that the text does not support.
        present = contains_on_word_boundary(seed.answer, p.text)
        if p.is_answer_bearing and not present:
            problems.append(
                f"{seed.seed_id}/{p.passage_id}: flagged answer-bearing but the "
                f"answer {seed.answer!r} does not occur in the stored text"
            )
    if not any(p.is_answer_bearing for p in seed.passages):
        problems.append(f"{seed.seed_id}: no answer-bearing passage")
    return problems


def load_natural_questions(
    limit: int = 2000, *, evidence_tokens: int = DEFAULT_EVIDENCE_TOKENS
) -> tuple[list[Seed], SeedFilterStats]:
    """Load Natural Questions via the datasets library.

    Deferred import so that the schema and perturbation logic can be imported
    and unit tested without the heavy dependency present.

    Evidence is a window centred on the short answer rather than the document's
    opening tokens, and `is_answer_bearing` is decided by reading the text that
    is actually stored. Rejections are counted and returned because P1 Section
    5.5.1 makes the acceptance rate a reportable property of the benchmark.
    """
    from datasets import load_dataset  # noqa: PLC0415

    # Streamed. Natural Questions is enormous and a materialised download would
    # cost hours of wall clock for the few thousand rows actually needed.
    ds = load_dataset(NQ_DATASET, split="validation", streaming=True)
    stats = SeedFilterStats(source="natural_questions", seen=0, no_short_answer=0,
                            answer_not_in_document=0, kept=0)
    out: list[Seed] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        stats["seen"] += 1
        ann = row.get("annotations") or {}
        short = (ann.get("short_answers") or [{}])[0]
        text = (short.get("text") or [None])[0] if isinstance(short, dict) else None
        if not text:
            stats["no_short_answer"] += 1
            continue

        document = " ".join(row["document"]["tokens"]["token"])
        window = extract_answer_window(document, text, evidence_tokens)
        if window is None:
            # The short answer is not recoverable from the document text, so
            # there is no honest way to mark a passage as answer-bearing.
            stats["answer_not_in_document"] += 1
            continue

        out.append(
            Seed(
                seed_id=f"nq-{i}",
                source=SeedSource.NATURAL_QUESTIONS,
                query=row["question"]["text"],
                answer=text,
                passages=[
                    Passage(
                        passage_id=f"nq-{i}-p0",
                        text=window,
                        is_answer_bearing=contains_on_word_boundary(text, window),
                    )
                ],
                extra_passages=[
                    Passage(
                        passage_id=f"nq-{i}-x{k}",
                        text=w,
                        is_answer_bearing=False,
                    )
                    for k, w in enumerate(
                        # A noise case at r=0.90 carries nine distractor tokens
                        # per signal token -- about 6,300 -- and the seed's own
                        # article is where its genuine topical neighbours live.
                        # Sixteen windows of 700 tokens can fund that ratio
                        # without falling back on weaker cross-document matches.
                        extract_context_windows(
                            document, text, evidence_tokens, n_windows=16
                        )
                    )
                ],
            )
        )
        stats["kept"] += 1
    return out, stats


def load_trivia_qa(
    limit: int = 2000, *, evidence_tokens: int = DEFAULT_EVIDENCE_TOKENS
) -> tuple[list[Seed], SeedFilterStats]:
    """Load TriviaQA via the datasets library.

    One answer-bearing window plus at most one answer-free companion passage, so
    a seed carries the multi-passage structure the retrieval step needs while
    staying inside the evidence budget. The previous loader stored three 2,500
    character contexts (~1,290 tokens), which no longer fits, and it decided
    `is_answer_bearing` from the untruncated context while storing the truncated
    one -- so a passage could be flagged answer-bearing after truncation had
    removed the answer.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(TRIVIAQA_DATASET, "rc.wikipedia", split="validation", streaming=True)
    stats = SeedFilterStats(source="trivia_qa", seen=0, no_answer_or_context=0,
                            answer_not_in_context=0, kept=0)
    out: list[Seed] = []

    answer_budget = int(evidence_tokens * 0.7)
    companion_budget = evidence_tokens - answer_budget

    for i, row in enumerate(ds):
        if i >= limit:
            break
        stats["seen"] += 1
        answer = row["answer"]["value"]
        contexts = row["entity_pages"]["wiki_context"]
        if not answer or not contexts:
            stats["no_answer_or_context"] += 1
            continue

        window = None
        used_index = -1
        for j, ctx in enumerate(contexts):
            window = extract_answer_window(ctx, answer, answer_budget)
            if window is not None:
                used_index = j
                break
        if window is None:
            stats["answer_not_in_context"] += 1
            continue

        passages = [
            Passage(
                passage_id=f"tqa-{i}-p0",
                text=window,
                is_answer_bearing=contains_on_word_boundary(answer, window),
            )
        ]
        # A companion passage from a different context, only if it is genuinely
        # answer-free; otherwise the refusal perturbation could not remove the
        # evidence by dropping the flagged passage alone.
        for j, ctx in enumerate(contexts):
            if j == used_index:
                continue
            companion = " ".join(ctx.split()[:companion_budget])
            if companion and not distractor_states_answer(companion, answer):
                passages.append(
                    Passage(
                        passage_id=f"tqa-{i}-p1",
                        text=companion,
                        is_answer_bearing=False,
                    )
                )
                break

        # Every wiki context for this entity is a same-topic neighbour, so the
        # whole entity page stocks the distractor corpus, not just the context
        # the answer window came from.
        extras: list[Passage] = []
        for j, ctx in enumerate(contexts):
            for k, w in enumerate(
                # Sized like a retrieval chunk, not like the companion evidence
                # slot: these are distractors, so `companion_budget` (210 tokens)
                # would make them a third the size of an NQ distractor and force
                # three times as many passages to fund the same noise ratio.
                extract_context_windows(ctx, answer, evidence_tokens, n_windows=8)
            ):
                extras.append(
                    Passage(
                        passage_id=f"tqa-{i}-x{j}_{k}",
                        text=w,
                        is_answer_bearing=False,
                    )
                )

        out.append(
            Seed(
                seed_id=f"tqa-{i}",
                source=SeedSource.TRIVIA_QA,
                query=row["question"],
                answer=answer,
                passages=passages,
                extra_passages=extras,
            )
        )
        stats["kept"] += 1
    return out, stats
