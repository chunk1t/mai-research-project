"""End-to-end smoke build at P1's target proportions, using mock seeds."""
import sys, random, numpy as np
sys.path.insert(0, "src")
from ragrobust.schema import Benchmark, Passage, SeedSource, Dimension
from ragrobust.dataset.seeds import Seed, filter_seeds
from ragrobust.dataset.refusal import build_refusal_testbed
from ragrobust.dataset.noise import build_noise_testbed, achieved_ratio
from ragrobust.dataset.conflict import build_conflict_case
from ragrobust.corpus import SandboxedCorpus

R = random.Random(20260721)
def mk(i):
    ans = str(1800 + (i % 150))
    ps = [Passage(passage_id=f"s{i}-p0",
                  text=f"The monument in region {i} was finished in {ans} following a long build.",
                  is_answer_bearing=True)]
    for j in range(3):
        ps.append(Passage(passage_id=f"s{i}-p{j+1}",
                          text=f"Region {i} is known for its architecture and hosts events in season {j}. " * 3))
    return Seed(seed_id=f"s{i}",
                source=SeedSource.NATURAL_QUESTIONS if i % 2 else SeedSource.TRIVIA_QA,
                query=f"In what year was the monument in region {i} finished?",
                answer=ans, passages=ps)

seeds = [mk(i) for i in range(900)]
kept, fstats = filter_seeds(seeds, min_evidence_tokens=20, max_evidence_tokens=5000)
print("seed filter:", dict(fstats))

emb = {}
for s in kept:
    base = np.array([R.gauss(0,1) for _ in range(16)])
    for k,p in enumerate(s.passages):
        emb[p.passage_id] = base + np.array([R.gauss(0,0.4) for _ in range(16)])*(1+0.1*k)

# Refusal: 100 unanswerable + 100 answerable = 200
refusal, rstats = build_refusal_testbed(kept[:200], emb, n_unanswerable=100,
                                        n_answerable=100, similarity_floor=-1.0,
                                        similarity_ceiling=1.0)
print("refusal:", rstats)

# Conflict: 200 (contradiction text mocked; real run uses Gemini + NLI filter)
conflict = []
for s in kept[200:400]:
    wrong = str(int(s.answer) + R.choice([-5,-3,3,5]))
    gen = s.answer_passages[0].text.replace(s.answer, wrong)
    conflict.append(build_conflict_case(s, gen, 0.93, gen_params={"mock": True}))
print("conflict:", len(conflict))

# Noise: 500 seeds x 5 ratios = 2500
pool = [Passage(passage_id=f"d{j}", text=" ".join(["contextual filler text"]*25)) for j in range(120)]
for j,pp in enumerate(pool):
    emb[pp.passage_id] = np.array([R.gauss(0,1) for _ in range(16)])
noise, nstats = build_noise_testbed(kept[400:], pool, emb, n_cases=500, similarity_floor=-1.0, similarity_ceiling=1.0)
print("noise:", {k:v for k,v in nstats.items() if k != "warnings"})

bench = Benchmark(version="0.1.0-smoke", cases=refusal + conflict + noise)
m = bench.manifest()
print("\n=== MANIFEST ===")
for k,v in m.items(): print(f"  {k}: {v}")

inst = bench.evaluation_instances()
print(f"\ninstances/config: {inst}   (P1 target ~2900)")
print(f"x12 configs:      {inst*12}   (P1 target ~34800)")

# Ratio fidelity
from collections import defaultdict
byr = defaultdict(list)
for c in noise: byr[c.noise_ratio].append(achieved_ratio(c.retrieved_passages))
print("\nnoise ratio fidelity (target -> mean achieved):")
for r in sorted(byr): print(f"  {r:.2f} -> {np.mean(byr[r]):.3f}  (n={len(byr[r])})")

# Sandbox holds on every unanswerable case
bad = 0
for c in refusal:
    if not c.is_answerable:
        sc = SandboxedCorpus(c, distractor_pool=[])
        sc.assert_perturbation_holds()
        if any(p.is_answer_bearing for p in sc.retrieve(c.query, 999, "bm25").passages): bad += 1
print(f"\nsandbox leaks on unanswerable cases: {bad}")

bench.to_jsonl("data/cases/smoke.jsonl")
rt = Benchmark.from_jsonl("data/cases/smoke.jsonl", "0.1.0-smoke")
print(f"roundtrip: {len(rt.cases)} cases, hash match: {rt.manifest()['content_hash']==m['content_hash']}")
