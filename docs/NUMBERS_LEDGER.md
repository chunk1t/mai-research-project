# Numbers ledger

Every quantitative claim in the report, mapped to the artefact it was read from
and the exact JSON path within it. `python scripts/verify_pack.py` parses the
table below and asserts each value against its artefact, so a number cannot
drift from the data that produced it. Run it from the repository root; it takes
no arguments.

Extracted from the writing pack that backs Chapters 6 to 9; only the
machine-checkable rows are reproduced here, since the rest is drafting material
rather than an artefact of the run.

## Ledger

Machine-checkable. `python scripts/verify_pack.py` parses this table and asserts
every value against its artefact, so the pack cannot drift from the data.

| id | value | artefact | json path |
|---|---|---|---|
| naive.refusal_f1 | 0.8733 | runs/analysis.json | descriptive.naive.refusal.refusal_f1 |
| reasoning.refusal_f1 | 0.8387 | runs/analysis.json | descriptive.reasoning.refusal.refusal_f1 |
| agentic.refusal_f1 | 0.8645 | runs/analysis.json | descriptive.agentic.refusal.refusal_f1 |
| naive.refusal_precision | 0.7975 | runs/analysis.json | descriptive.naive.refusal.refusal_precision |
| reasoning.refusal_precision | 0.7555 | runs/analysis.json | descriptive.reasoning.refusal.refusal_precision |
| agentic.refusal_precision | 0.8114 | runs/analysis.json | descriptive.agentic.refusal.refusal_precision |
| naive.refusal_recall | 0.9650 | runs/analysis.json | descriptive.naive.refusal.refusal_recall |
| reasoning.refusal_recall | 0.9425 | runs/analysis.json | descriptive.reasoning.refusal.refusal_recall |
| agentic.refusal_recall | 0.9250 | runs/analysis.json | descriptive.agentic.refusal.refusal_recall |
| naive.plain_accuracy | 0.2700 | runs/analysis.json | descriptive.naive.refusal.plain_accuracy |
| naive.crs | 0.0227 | runs/analysis.json | descriptive.naive.conflict.crs_mean |
| reasoning.crs | 0.2008 | runs/analysis.json | descriptive.reasoning.conflict.crs_mean |
| agentic.crs | 0.0570 | runs/analysis.json | descriptive.agentic.conflict.crs_mean |
| naive.binary_correct_side | 0.3813 | runs/analysis.json | descriptive.naive.conflict.binary_correct_side_accuracy |
| reasoning.binary_correct_side | 0.1477 | runs/analysis.json | descriptive.reasoning.conflict.binary_correct_side_accuracy |
| agentic.binary_correct_side | 0.3291 | runs/analysis.json | descriptive.agentic.conflict.binary_correct_side_accuracy |
| naive.binary_committed | 0.5957 | runs/analysis.json | descriptive.naive.conflict.binary_correct_side_accuracy_committed_only |
| reasoning.binary_committed | 0.4875 | runs/analysis.json | descriptive.reasoning.conflict.binary_correct_side_accuracy_committed_only |
| agentic.binary_committed | 0.6265 | runs/analysis.json | descriptive.agentic.conflict.binary_correct_side_accuracy_committed_only |
| naive.ragas_faithfulness | 0.6218 | runs/analysis.json | descriptive.naive.conflict.ragas_faithfulness |
| reasoning.ragas_faithfulness | 0.8718 | runs/analysis.json | descriptive.reasoning.conflict.ragas_faithfulness |
| agentic.ragas_faithfulness | 0.6731 | runs/analysis.json | descriptive.agentic.conflict.ragas_faithfulness |
| reasoning.non_termination_rate | 0.2614 | runs/analysis.json | descriptive.reasoning.conflict.non_terminating_rate |
| agentic.non_termination_rate | 0.0569 | runs/analysis.json | descriptive.agentic.conflict.non_terminating_rate |
| naive.non_termination_rate | 0.0000 | runs/analysis.json | descriptive.naive.conflict.non_terminating_rate |
| naive.ndc_auc | 0.5481 | runs/analysis.json | descriptive.naive.noise.ndc_auc |
| reasoning.ndc_auc | 0.5552 | runs/analysis.json | descriptive.reasoning.noise.ndc_auc |
| agentic.ndc_auc | 0.5933 | runs/analysis.json | descriptive.agentic.noise.ndc_auc |
| naive.ndc50 | 0.8663 | runs/analysis.json | descriptive.naive.noise.ndc50 |
| naive.ndc_slope | 0.0221 | runs/analysis.json | descriptive.naive.noise.ndc_slope |
| reasoning.ndc_slope | 0.0078 | runs/analysis.json | descriptive.reasoning.noise.ndc_slope |
| agentic.ndc_slope | 0.0149 | runs/analysis.json | descriptive.agentic.noise.ndc_slope |
| judge.kappa | 0.8258 | runs/validation/crs_judge_agreement.json | cohens_kappa |
| judge.quadratic_kappa | 0.9565 | runs/validation/crs_judge_agreement.json | quadratic_weighted_kappa |
| judge.ac1 | 0.9792 | runs/validation/crs_judge_agreement.json | paradox_diagnostics.gwet_ac1 |
| judge.observed_agreement | 0.9800 | runs/validation/crs_judge_agreement.json | observed_agreement |
| judge.kappa_prefix | 0.0805 | runs/validation/prefix_judge/crs_judge_agreement.json | cohens_kappa |
| labels.kappa | -0.0338 | runs/validation/label_agreement.json | paradox_diagnostics.cohens_kappa |
| labels.observed_agreement | 0.8900 | runs/validation/label_agreement.json | paradox_diagnostics.observed_agreement |
| labels.prevalence_index | 0.8900 | runs/validation/label_agreement.json | paradox_diagnostics.prevalence_index |
| labels.bias_index | 0.0700 | runs/validation/label_agreement.json | paradox_diagnostics.bias_index |
| labels.pabak | 0.7800 | runs/validation/label_agreement.json | paradox_diagnostics.pabak |
| labels.ac1 | 0.8772 | runs/validation/label_agreement.json | paradox_diagnostics.gwet_ac1 |
| validation.fraction | 0.1753 | runs/validation/sample_stats.json | fraction_achieved |
| validation.base_cases | 890 | runs/validation/sample_stats.json | base_cases_total |
| validation.primary_n | 156 | runs/validation/sample_stats.json | primary_n |
| benchmark.n_cases | 2858 | runs/benchmark_manifest.json | n_cases |
| benchmark.content_hash | 903392537ebef4de | runs/benchmark_manifest.json | content_hash |
| run.n_rows | 34283 | runs/scores.json | n_rows |
| run.n_errored | 13 | runs/scores.json | n_errored |
| run.recovered_total | 1983 | runs/scores.json | undelimited_answer_recovery.recovered_total |
| run.unrecoverable | 459 | runs/scores.json | undelimited_answer_recovery.unrecoverable_nonempty |
| ragas.n_instances | 450 | runs/ragas_analysis.json | n_instances |
| ragas.conflict.miss_rate | 0.7000 | runs/ragas_analysis.json | coverage_gap.conflict.faithfulness.miss_rate |
| ragas.refusal.miss_rate | 0.7143 | runs/ragas_analysis.json | coverage_gap.refusal.faithfulness.miss_rate |
| ragas.noise.miss_rate | 0.6897 | runs/ragas_analysis.json | coverage_gap.noise.faithfulness.miss_rate |
| ragas.abstained_n | 188 | runs/ragas_analysis.json | abstention.abstained.n |
| ragas.abstained_faithfulness_defined | 1 | runs/ragas_analysis.json | abstention.abstained.faithfulness.n_defined |
| ragas.committed_faithfulness_defined | 234 | runs/ragas_analysis.json | abstention.committed.faithfulness.n_defined |
| ragas.abstained_answer_relevance | 0.3436 | runs/ragas_analysis.json | abstention.abstained.answer_relevance.mean_where_defined |
| ragas.refusal.ctx_rho | -0.3114 | runs/ragas_analysis.json | correlations.refusal.context_relevance.spearman_rho |
| artefacts.n_pairs | 198 | runs/generation_artefacts.json | n_pairs |
| artefacts.mean_edit | 0.0092 | runs/generation_artefacts.json | edit_localisation.mean_edit_fraction |
| artefacts.max_edit | 0.0604 | runs/generation_artefacts.json | edit_localisation.max_edit_fraction |
| artefacts.over_half_rewritten | 0 | runs/generation_artefacts.json | edit_localisation.n_pairs_over_half_rewritten |
| artefacts.max_case_share | 0.0101 | runs/generation_artefacts.json | vocabulary.generated_only_max_case_share |

## Inferential claims

Checked by the same script against `runs/analysis.json`, independently of the
table above.

**Six of nine differences are statistically significant** at the 95% bootstrap
level. None reaches its pre-declared practical threshold.

**Every significant difference is small.** Six of nine comparisons exclude zero;
none of them clears the effect size that was declared to matter in advance.
