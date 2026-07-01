# AgentDesk Benchmark

Reproducible evaluation of the multi-agent RAG pipeline on a real corpus.

## Setup

- **Corpus:** Python standard-library documentation — 2,402 documented objects across
  86 modules, chunked into **927** retrievable chunks. Built offline by
  `scripts/build_corpus.py` (no network, no API key).
- **Eval set:** **250** auto-generated definition questions ("What does `X` do?") with
  the object's first docstring sentence as the reference. Built by `scripts/build_eval.py`
  (seeded, reproducible).
- **Metric:** grounded faithfulness — fraction of the answer supported by the retrieved
  context. Lexical judge without an API key; **LLM-as-judge** when a provider is set.

## Reproduce

```bash
python scripts/build_corpus.py corpus
python scripts/build_eval.py 250 eval/dataset_large.jsonl
python -m agentdesk.rag.ingest ./corpus
python -m eval.run_eval eval/dataset_large.jsonl
```

## Results (keyless run — lexical judge, grounded-mock LLM)

| Metric | Value |
|--------|-------|
| Corpus chunks indexed | 927 |
| Questions | 250 |
| Faithfulness (RAG) | 98.3% |
| Faithfulness (no-retrieval baseline) | 64.8% |
| Retrieval latency (p95) | ~12 ms |

### How to read these numbers — honestly

The keyless run **validates the pipeline, not answer quality**:

- The grounded-mock LLM answers **extractively** (it copies the best-matching context
  sentence), so its answers are near-trivially "faithful" → the 98.3% mostly measures
  that retrieval surfaces the right chunk, not that a model reasoned well.
- The no-retrieval baseline uses a fixed fallback string whose words happen to overlap
  the corpus vocabulary, which **inflates** it to 64.8% — treat it as a floor artifact,
  not a real comparison.
- The **~12 ms** latency is real and meaningful: it's retrieval + agent overhead with no
  model call. With a real LLM in the loop, end-to-end latency is dominated by generation
  (typically ~1–2 s).

### Real quality numbers (with a provider)

Set `OPENAI_API_KEY` (or configure Bedrock) and re-run — the harness switches to a real
generative model **and** an LLM-as-judge:

```bash
export OPENAI_API_KEY=sk-...       # or configure Bedrock
python -m eval.run_eval eval/dataset_large.jsonl
# [eval] judge: LLM-as-judge
```

Record whatever you measure here — those are the numbers worth putting on a résumé.
