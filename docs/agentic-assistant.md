# Agentic Financial Research Assistant

This document describes the agentic enhancement of FinEvidence (the PDC project): what it adds to the existing pipeline, how it works, how it is evaluated, and the observed results.

## Goal

The existing FinEvidence pipeline answers a financial question in one fixed pass: retrieve the top segments, generate a program, execute it. It always answers, and it cannot notice when its evidence is missing, inconsistent or misread.

The agent wraps the same components in a bounded research loop. It plans the question, retrieves evidence iteratively, calculates only with controlled tools, verifies every candidate answer, and abstains when the evidence is insufficient, ambiguous or conflicting. Every answer comes with its source citations, its calculation trace and the log of tool calls that produced it.

## What it reuses and what it adds

| Existing component (unchanged) | Used by the agent as |
| --- | --- |
| Allowlisted inference input (`build_structured_input`, `assert_inference_safe`) | The only view of a record the toolbox can see |
| Evidence segmentation (`build_evidence_segments`, `segment_retrieval_text`) | The unit of retrieval and citation (`table_n`, `text_n`) |
| Financial-value normalization (`parse_numeric_candidate`) | Parsing every number the agent reads |
| Program parser and safe executor (`parse_finqa_program`, `execute_program`) | The `calculate` tool: no `eval`, official FinQA semantics |
| Operand grounding (`operand_grounding`) | A hard verification check |
| Execution accuracy (`compare_execution`) | The headline metric |
| Conservative arithmetic (`reason_over_candidates`) | The single-pass pipeline baseline |
| Retriever rankings and generator beams (Notebooks 10, 14) | Optional backends (`ranking-file`, `generator-beams`) |

| New (`finevidence.agent`, unless noted) | Role |
| --- | --- |
| `planner` | Task planning: question → task type, entity terms, periods, ratio part/whole, ordered steps |
| `tools` | Controlled, logged tools: `search_evidence`, `extract_values`, `calculate` |
| `evidence` | Value mentions with row label, column and period; conflict detection |
| `proposers` | Candidate programs: `TemplateProposer` (offline) or `CandidateFileProposer` (T5 beams) |
| `verifier` | Hard and soft checks on every candidate |
| `controller` | The research loop, answer selection and the abstention policy |
| `report` | Markdown research report: answer, plan, evidence, calculation, checks, tool log |
| `baselines`, `benchmark` | The single-pass pipeline wrapper and the system comparison |
| `cli` | `finevidence-agent ask` / `evaluate` |
| `finevidence.retrieval.lexical` | BM25 within-record retriever; replay of saved rankings |
| `finevidence.evaluation.agentic` | Coverage, selective accuracy, wrong-answer rate, evidence completeness, efficiency, risk-coverage |

## Research loop

```
question + report
      │
      ▼
 1. Plan ─────────── planner        task (percent change, change, ratio, sum, average,
      │                             row max/min, comparison), entity terms, periods, steps
      ▼
 2. Retrieve ─────── search_evidence round 1: the question; later rounds: entity + each
      │      ▲                      period (and "total" for ratios), with a larger k
      ▼      │
 3. Read values ──── extract_values  labelled numbers from retrieved rows and sentences
      │      │
      ▼      │
 4. Propose ──────── proposer        candidate programs that fill the plan's slots
      │      │
      ▼      │
 5. Verify ───────── calculate +     executes · cites retrieved evidence · literals grounded
      │      │       verifier        · plausible for the task · periods match · no conflict
      ▼      │
 6. Decide ──────────────────────────  confident and unambiguous → answer
             └── not yet, budget left ─┘  budget spent → abstain with the reason
```

The budget is `max_rounds` retrieval rounds (3 by default), starting at `initial_k = 5` segments and adding `k_step = 5` per round. A question whose shape the proposer cannot calculate is abstained on immediately, without spending retrieval budget.

### Verification

| Check | Kind | Fails when |
| --- | --- | --- |
| `executes` | hard | the program does not parse or execute on the safe executor |
| `cites_evidence` | hard | the program cites no segment, or one that was never retrieved |
| `operands_grounded` | hard | a numeric literal does not occur in the cited segments |
| `plausible` | hard | e.g. a percent change beyond ±1000%, a share of a whole outside [0, 1], a change of exactly zero, a number for a yes/no question |
| `periods_match` | soft | no operand comes from a period the question names (support × 0.6) |
| `no_conflicting_evidence` | soft, blocking | the evidence holds two different values for the same label, column and period |

### Abstention

The agent abstains, and says why, when:

- **unsupported**: no supported calculation shape was recognised;
- **insufficient evidence**: no values matching the plan were found, or no candidate passed the hard checks;
- **low confidence**: the best candidate's support is below `min_confidence` (0.60);
- **ambiguous**: a candidate with a different value is within `ambiguity_margin` (0.01) of the best;
- **conflicting evidence**: the best candidate uses a value that the evidence contradicts.

An abstention keeps the withheld best candidate (`withheld_value`), so risk-coverage curves can be computed without rerunning.

## Running it

```bash
python -m pip install -e '.[dev]'
python -m finevidence.data.acquisition          # pinned train/dev, SHA-256 verified

finevidence-agent ask --split dev --index 0      # Markdown research report
finevidence-agent ask --split dev --id "V/2008/page_17.pdf-1" --json
finevidence-agent ask --record my_report.json    # {"question", "table", "pre_text", "post_text"}
finevidence-agent evaluate --split dev --results data/processed/agent/dev_comparison.json
```

`python -m finevidence.agent` is equivalent to `finevidence-agent`. To run the agent on the trained models' outputs instead of the offline backends, point it at the saved artifacts:

```bash
finevidence-agent evaluate --config configs/agent_generator.toml \
  --rankings data/processed/retriever_bert_base_dev_rankings.jsonl \
  --beams data/processed/generator-t5-base-dev-v1_dev_iteration1_candidates.jsonl
```

The test split is not reachable from the CLI. It stays reserved for the single final M4 run.

## Evaluation protocol

- **Systems**, all scored identically by `finevidence.agent.benchmark` on the same records:
  1. `pipeline-conservative-arithmetic`: the existing single-pass baseline (Notebook 08 procedure, BM25 top 3 in place of TF-IDF);
  2. `agent-single-pass-v1`: planner and template proposer, one retrieval round, no verification, no abstention (`configs/agent_single_pass.toml`);
  3. `agent-no-abstention-v1`: adds iterative retrieval and verification but always answers (`configs/agent_no_abstention.toml`);
  4. `agent-offline-v1`: the full agent (`configs/agent_offline.toml`).
- **Design data**: planner rules and template logic were written from train questions, and the abstention policy was chosen on train records 1000–1999. Dev was run once for the results below. Test was not loaded.
- **Metrics** (`finevidence.evaluation.agentic`):
  - answer accuracy: official execution accuracy over all records, with abstentions counted as incorrect, plus a bootstrap 95% CI;
  - reliability: coverage, selective accuracy (accuracy of answered questions) and wrong-answer rate (answered and wrong, over all records);
  - evidence completeness: share of answered questions whose citations include every gold segment, gold recall and citation precision;
  - efficiency: tool calls, retrieval rounds, latency.

## Results (dev, 883 questions)

| System | Exec acc [95% CI] | Coverage | Selective acc | Wrong-answer rate | Cites all gold | Gold recall | Tool calls | ms / q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Existing pipeline: conservative arithmetic | 3.96% [2.83, 5.32] | 32.16% | 12.32% | 28.20% | 39.08% | 54.98% | 1.0 | 0.5 |
| Agent, single pass | 22.54% [19.82, 25.25] | 70.55% | 31.94% | 48.02% | 51.69% | 65.52% | 5.5 | 0.9 |
| Agent, iterative + verified, no abstention | **24.92%** [22.20, 27.75] | 76.22% | 32.69% | 51.30% | 50.07% | 64.04% | 6.5 | 1.4 |
| **Agent (full)** | 20.50% [17.89, 23.33] | 50.74% | **40.40%** | **30.24%** | **59.60%** | **72.01%** | 8.8 | 2.2 |

Paired comparisons on the same 883 records (exact McNemar):

- **Iterative retrieval and verification vs single pass**: 23 questions fixed, 2 broken (p ≈ 2e-5).
- **Full agent vs single pass, wrong answers**: 166 wrong answers avoided, 9 new ones (p ≈ 2e-38).
- **Full agent vs single pass, correct answers**: 14 gained, 32 lost to abstention (p = 0.011). Abstention costs about 2 points of accuracy in exchange for a 37% relative cut in wrong answers.
- **Every agent variant vs the existing conservative pipeline**: 164–202 questions fixed, 17–18 broken (p < 1e-29).

Abstentions of the full agent (435 of 883): low confidence 149, insufficient evidence 109, unsupported question shape 96, ambiguous 76, failed verification 5, conflicting evidence 0. Citation precision of answered questions is 77.9%. Latency is 2.2 ms mean and 6.2 ms p95 per question on an Apple M2 CPU.

Risk-coverage for the full agent: raising the support threshold from 0 to 0.8 moves selective accuracy over scored questions from 33.7% to 45.0%, while coverage falls to 56%. The threshold is therefore a direct reliability dial.

| Task (planner) | Dev questions | Answered | Correct |
| --- | ---: | ---: | ---: |
| ratio | 318 | 168 | 56 |
| change | 169 | 105 | 38 |
| percent change | 157 | 107 | 67 |
| unsupported | 96 | 0 | 0 |
| sum | 70 | 20 | 3 |
| average | 55 | 37 | 9 |
| row min / max | 13 | 8 | 8 |
| comparison | 5 | 3 | 0 |

## Interpretation

- **Agentic control helps both accuracy and reliability, but they trade off.** Iteration and verification alone add 2.4 points of accuracy over a single pass. Abstention then removes most wrong answers: 30% of questions answered wrongly instead of 48%, and when the agent does answer it is right 40% of the time instead of 32%. This is the core claim of the enhancement, and it is significant on paired dev records.
- **Evidence completeness improves.** When the full agent answers, its citations contain every gold segment 60% of the time, against 52% for the single pass and 39% for the existing pipeline.
- **The offline proposer bounds absolute accuracy.** Template slot filling reaches about 25% execution accuracy, far below the trained T5 generator's 61.95% (Notebook 14). The agent's verifier, abstention and reporting do not depend on the template proposer: `CandidateFileProposer` runs the same loop over the T5 beams (`configs/agent_generator.toml`). That comparison, the existing T5 pipeline against the agent over T5 beams, is the most important next run. It needs the saved Notebook 10 rankings and Notebook 14 candidates, which are not on this machine, so it has not been executed.

## Limitations

- **Design and tuning data.** Planner rules and thresholds were set on train; results are dev only, from one deterministic run. No test-split result exists.
- **Planner coverage.** The rule-based planner covers seven calculation shapes; 96 dev questions (11%) fall outside them and are always abstained on. Multi-step questions that combine shapes, such as the share of a change, are not planned.
- **Weak sentence evidence.** Values in sentences are bound from nearby words and the nearest year, which is much weaker than table binding; sums and averages are the weakest tasks.
- **Heuristic confidence.** The support score is a heuristic, not a calibrated probability.
- **Conflict check not exercised on dev.** The conflicting-evidence check fired on no dev question; it is covered by unit tests only.
- **One dev-prompted fix.** After the first dev run, inspecting dev record 0 showed an average taken across unlike columns. Sums and averages without named periods were then restricted to dated, like-for-like columns, re-checked on train and dev re-run once. The table above is that second run; the first is in the experiment log.
- **Baseline retriever.** The pipeline baseline uses BM25 rather than Notebook 08's TF-IDF. It reproduces Notebook 08's 35/883 execution-correct count, but the two retrievers are not identical record by record.
