"""Prompt templates and final-answer extraction.

P1 Section 5.6.5 lists "the prompt template prefix that frames the task" among
the variables held constant across configurations. That constant prefix lives
here as `TASK_PREFIX` and is shared verbatim by all three pipeline classes, so
the only prompt-level difference between classes is the reasoning-elicitation
block. Without that discipline a prompt difference would ride along with the
pipeline class and confound the comparison the study is built to make.

P1 Section 5.6.4 requires that "each pipeline is prompted to terminate its output
with a delimited final-answer field, and only the content of that field is
scored". The delimiters are defined here and the extraction is deliberately
strict: a response that does not close the field is reported as unextractable
rather than salvaged, because silently scoring a partial answer would make an
output-length failure look like a reasoning failure.

The refusal instruction is not optional decoration. P1 Section 5.4.1 defines the
correct response to an unanswerable case as "an explicit refusal of the form
'the available evidence is insufficient' or equivalent". A model never told that
abstention is permitted cannot be scored fairly on refusal recall.
"""

from __future__ import annotations

import re

ANSWER_OPEN = "<final_answer>"
ANSWER_CLOSE = "</final_answer>"

# The exact string a pipeline is told to emit when the evidence is insufficient.
# The response parser (build item 3) classifies against a wider pattern set;
# this is what the prompt asks for, not the only form accepted.
REFUSAL_SENTINEL = "INSUFFICIENT EVIDENCE"

# Held constant across all three pipeline classes (P1 Section 5.6.5).
TASK_PREFIX = """\
You answer questions using only the passages provided below. Follow these rules:

1. Use only the provided passages. Do not use outside knowledge.
2. If the passages do not contain enough information to answer, say so instead of
   guessing.
3. If the passages disagree with each other, say that they disagree, state each
   position, and explain which you follow and why.
4. End your response with the answer enclosed in {open}{close} tags.
   Put the answer there and nothing else.
   If the evidence is insufficient, put exactly {sentinel} inside the tags."""

# The only block that varies by pipeline class.
REASONING_BLOCK = """\

Before answering, work through the passages step by step: identify what each one
claims about the question, note any disagreement or missing information, then
decide. Put that reasoning before the {open} tag."""

DECOMPOSE_BLOCK = """\
You are planning how to answer a question from a fixed set of passages.

Question: {query}

Write between one and {max_subqueries} short search queries that would help find
the evidence needed. Put each on its own line. Write nothing else."""

SYNTHESISE_BLOCK = """\

You gathered the evidence below over several search steps. Synthesise it into a
single answer to the question."""


def _format_passages(passages: list[dict[str, str]]) -> str:
    """Number passages so a response can refer to them and a human can audit them."""
    return "\n\n".join(
        f"[{i + 1}] {p['text'].strip()}" for i, p in enumerate(passages)
    )


def build_prompt(
    query: str,
    passages: list[dict[str, str]],
    *,
    reasoning: bool = False,
    extra_block: str = "",
) -> str:
    """Assemble a generation prompt.

    `reasoning` appends the chain-of-thought elicitation block of P1 Section
    5.6.2. It is the prompted form of reasoning; the reasoning-trained form is
    selected at the provider layer through the thinking flag instead, so a
    reasoning configuration uses exactly one of the two and never both.
    """
    prefix = TASK_PREFIX.format(
        open=ANSWER_OPEN, close=ANSWER_CLOSE, sentinel=REFUSAL_SENTINEL
    )
    parts = [prefix]
    if reasoning:
        parts.append(REASONING_BLOCK.format(open=ANSWER_OPEN))
    if extra_block:
        parts.append(extra_block)
    parts.append(f"\nPassages:\n{_format_passages(passages)}")
    parts.append(f"\nQuestion: {query}")
    return "\n".join(parts)


def build_decompose_prompt(query: str, *, max_subqueries: int = 3) -> str:
    return DECOMPOSE_BLOCK.format(query=query, max_subqueries=max_subqueries)


_ANSWER_RE = re.compile(
    re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.DOTALL | re.IGNORECASE
)


def extract_final_answer(text: str) -> str | None:
    """Return the content of the delimited final-answer field, or None.

    None means the field was absent or never closed. The caller decides what
    that signifies: the agentic loop reads it as "not finished yet", and the
    response parser reads it as unparseable, which P1 Section 5.6.4 scores as a
    non-refusal and as incorrect.

    When a response contains several fields the last one wins, because a model
    that restates its answer has committed to the final statement.
    """
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def parse_subqueries(text: str, *, max_subqueries: int = 3) -> list[str]:
    """Read the decomposition step's output into a list of sub-queries.

    Tolerant of the numbering and bullet styles models add unprompted, because a
    decomposition step that yields nothing usable would silently turn the
    agentic pipeline into a single-retrieval pipeline and erase the very
    difference the study is measuring.
    """
    out: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        # Strip "1.", "1)", "-", "*", and surrounding quotes.
        cleaned = re.sub(r"^\s*(?:\d+\s*[.)]|[-*•])\s*", "", cleaned)
        cleaned = cleaned.strip().strip('"').strip("'").strip()
        if cleaned:
            out.append(cleaned)
        if len(out) >= max_subqueries:
            break
    return out


# --------------------------------------------------------------------------
# Agentic ReAct blocks (P1 Section 5.6.3)
# --------------------------------------------------------------------------

NEXT_QUERY_MARKER = "NEXT_QUERY:"

REACT_STEP_BLOCK = """\

You are searching a fixed set of passages to answer the question, one search at a
time. The evidence gathered so far appears below.

Decide what to do next:
  - If the evidence is enough to answer, or is clearly never going to be enough,
    answer now using the {open}{close} tags described above.
  - Otherwise write one more search query on a single line beginning with
    {marker} and write nothing else.

You have {remaining} search step(s) left."""

_NEXT_QUERY_RE = re.compile(
    re.escape(NEXT_QUERY_MARKER) + r"[ \t]*(.+)", re.IGNORECASE
)


def format_evidence(evidence: list[tuple[str, list[dict[str, str]]]]) -> str:
    """Render accumulated evidence, de-duplicated across search steps.

    A passage retrieved by two sub-queries is shown once. Repeating it would
    weight it twice in the model's context purely as an artefact of the search
    path, which would make the agentic class's effective noise ratio depend on
    how often it happened to re-retrieve the same passage.
    """
    seen: set[str] = set()
    lines: list[str] = []
    n = 0
    for subquery, passages in evidence:
        fresh = [p for p in passages if p["passage_id"] not in seen]
        seen.update(p["passage_id"] for p in fresh)
        lines.append(f'Search: "{subquery}"')
        if not fresh:
            lines.append("  (no passages beyond those already gathered)")
        for p in fresh:
            n += 1
            lines.append(f"  [{n}] {p['text'].strip()}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_react_step_prompt(
    query: str,
    evidence: list[tuple[str, list[dict[str, str]]]],
    *,
    remaining: int,
) -> str:
    prefix = TASK_PREFIX.format(
        open=ANSWER_OPEN, close=ANSWER_CLOSE, sentinel=REFUSAL_SENTINEL
    )
    step = REACT_STEP_BLOCK.format(
        open=ANSWER_OPEN, close=ANSWER_CLOSE, marker=NEXT_QUERY_MARKER, remaining=remaining
    )
    return "\n".join(
        [prefix, step, f"\nEvidence so far:\n{format_evidence(evidence)}", f"\nQuestion: {query}"]
    )


def build_synthesise_prompt(
    query: str, evidence: list[tuple[str, list[dict[str, str]]]]
) -> str:
    prefix = TASK_PREFIX.format(
        open=ANSWER_OPEN, close=ANSWER_CLOSE, sentinel=REFUSAL_SENTINEL
    )
    return "\n".join(
        [
            prefix,
            SYNTHESISE_BLOCK,
            f"\nEvidence gathered:\n{format_evidence(evidence)}",
            f"\nQuestion: {query}",
        ]
    )


def parse_next_query(text: str) -> str | None:
    """Extract the agent's requested next search, or None if it did not ask.

    Checked only after `extract_final_answer` returns None, so a response that
    both answers and speculates about further searching is treated as an answer.
    A committed answer ends the loop (P1 Section 5.6.3).
    """
    match = _NEXT_QUERY_RE.search(text)
    if not match:
        return None
    return match.group(1).strip().strip('"').strip("'").strip() or None
