# Architecture

FinEvidence is a pipeline of tested modules with an agentic research layer on top. The pipeline (below) answers in one fixed pass. The agent (`finevidence.agent`, [agentic-assistant.md](agentic-assistant.md)) reuses the same modules as controlled tools inside a bounded plan–retrieve–verify loop that can abstain. Each stage reads only what it needs, and every model input is built from the allowlisted raw fields (`id`, `question`, `table`, `pre_text`, `post_text`).

## Pipeline (implemented in M3)

```
question + report (table, pre_text, post_text)
        │
        ▼
1. Load and verify ── finevidence.data          pinned splits, SHA-256 check, schema validation
        │
        ▼
2. Segment ───────── finevidence.retrieval      one segment per table row and sentence, keyed like gold_inds
        │                                       (table_n, text_n), rendered with column headers
        ▼
3. Retrieve ──────── Notebook 10                bert-base-uncased cross-encoder scores every
        │            (finevidence.retrieval)     (question, segment) pair; keep the top 5
        ▼
4. Generate ──────── Notebook 11                t5-base writes a FinQA program, e.g.
        │            (finevidence.generation)    subtract(5829, 5735), divide(#0, 5735)
        │                                       Notebook 14: keep the 4 beam candidates and take the
        │                                       first that executes and uses only numbers in the facts
        ▼
5. Execute ───────── finevidence.reasoning      safe executor (no eval), official FinQA value rules
        │
        ▼
answer + cited facts + program + executed value
        │
        ▼
6. Evaluate ──────── finevidence.evaluation     execution and program accuracy, retrieval recall,
                     finevidence.generation      invalid rate, operand grounding
```

## Modules

| Package | Responsibility | Key public functions |
| --- | --- | --- |
| `finevidence.data` | Load, validate and preprocess raw records; parse programs without executing them; check evidence keys; download pinned data | `load_finqa_split`, `build_structured_input`, `parse_finqa_program`, `validate_evidence_key`, `normalize_financial_value`, `acquisition.download_finqa_splits` |
| `finevidence.retrieval` | Evidence segments, retriever training and ranking pairs, retriever config | `build_evidence_segments`, `build_training_pairs`, `build_ranking_pairs`, `sample_epoch_pairs`, `load_retriever_config` |
| `finevidence.generation` | Generator inputs (training, retrieved, gold-facts oracle), config, per-record scoring | `build_training_examples`, `build_retrieved_examples`, `build_oracle_examples`, `score_program_prediction`, `load_generator_config`, `select_grounded_candidate` |
| `finevidence.reasoning` | Safe program executor; the conservative arithmetic baseline | `execute_program`, `reason_over_candidates` |
| `finevidence.evaluation` | Metrics | `compare_execution`, `compare_programs`, `gold_recall_at_k`, `operand_grounding`, `compare_display_answers` |

Training loops live in Notebooks 10–11, because they need a GPU and `torch`. Everything they call that runs without a GPU is in `src/` and unit-tested (`pytest`, 53 tests).

## Agentic research layer (PDC enhancement)

```
question + report
        │
        ▼
1. Plan ──────────── planner          task, entity terms, periods, ratio part/whole, steps
        │
        ▼
2. Search ────────── search_evidence  BM25 (or saved Notebook 10 rankings); round 1 uses the
        │      ▲                      question, later rounds entity + period sub-queries, wider k
        ▼      │
3. Read values ───── extract_values   labelled numbers per table row and sentence
        │      │
        ▼      │
4. Propose ───────── proposer         template slot filling, or saved T5 beams (Notebook 14)
        │      │
        ▼      │
5. Verify ────────── calculate +      safe executor, citations, grounding, plausibility,
        │      │     verifier         periods, conflicts
        ▼      │
6. Decide ───────────┘ refine while budget and doubt remain
        │
        ▼
answer + citations + calculation trace + checks + tool log, or abstention with its reason
```

| Package | Responsibility | Key public functions |
| --- | --- | --- |
| `finevidence.agent` | Planner, controlled toolbox, value mentions, proposers, verifier, research loop, report, CLI | `plan_question`, `build_agent`, `FinancialResearchAgent.run`, `render_markdown`, `finevidence-agent` |
| `finevidence.retrieval.lexical` | Within-record BM25 retriever; replay of saved Notebook 10 rankings | `Bm25Retriever`, `RankingFileRetriever` |
| `finevidence.evaluation.agentic` | Coverage, selective accuracy, wrong-answer rate, evidence completeness, efficiency, risk-coverage | `score_agent_answer`, `summarize_agent_scores`, `risk_coverage` |

The agent never reads annotation fields: its toolbox is built from `build_structured_input` and checked with `assert_inference_safe`. All arithmetic goes through `execute_program`.

## Baselines kept for comparison

Notebooks 06–08 use TF-IDF retrieval with extractive value selection or one-operation conservative arithmetic. They are transparent lower bounds: 0.6–4.0% dev execution accuracy, against 59.2% for the pipeline above.

## Design decisions

- **Program generation, not answer generation.** The model writes an executable program, so every answer carries an auditable calculation and the arithmetic is exact.
- **Shared rendering for training and inference.** Positives and negatives (retriever), and gold and distractor facts (generator), are all rendered from the raw context by key, never from annotated `gold_inds` text, so no class can be told apart by its formatting.
- **No publisher-derived inputs.** `table_retrieved`, `text_retrieved`, `model_input` and `tfidftopn` are excluded, because the official repository documents an earlier label leak in its table-to-text conversion.
- **Abstention and presentation:** implemented by the agent layer: a support threshold, ambiguity and conflict checks decide whether to answer, and `render_markdown` presents answer, evidence, calculation, checks and tool log together.
