# Latest eval results

_Generated 2026-07-01T23:10:42+00:00 · dataset `eval/dataset_large.jsonl` · model `grounded-mock (keyless)` · judge `lexical`_

| Metric | Value |
|--------|-------|
| Corpus chunks | 927 |
| Questions | 250 |
| Faithfulness (RAG) | 98.3% |
| Faithfulness (baseline) | 64.8% |
| Lift | 1.52x |
| Latency mean / p95 | 11ms / 13ms |

**Résumé line**
> Multi-agent RAG (LangGraph, hybrid RRF retrieval, ReAct tool-calling): 98% faithfulness vs 65% no-retrieval baseline on a 250-question benchmark over 927 chunks; p95 latency 13ms.

> ⚠️ **Not résumé-safe** — keyless mock (extractive) + lexical judge; this validates the pipeline, not answer quality. Re-run with `OPENAI_API_KEY` set.
