# oligogym-repro — production reproduction pipeline

Nextflow DSL2 pipeline reproducing the benchmark of *OligoGym: Curated Datasets
and Benchmarks for Oligonucleotide Drug Discovery* (Roche), against
`github.com/Roche/oligogym` at commit `97f5b9f58d9e445a8ecb0218037af7465c3708c0`.

Two arms, run separately:

| arm | configs | fold-fits | what it answers |
|---|---|---|---|
| `primary` | 9,188 (+2 skipped) | 45,940 | do we recover the published numbers? |
| `corrective` | 2,334 | 11,670 | what changes once the verified data defects are fixed? |

## Launch

```bash
# primary arm, both queues
nextflow run StevenFroelichBMRN/oligogym-repro -profile batch \
  --arm primary \
  --chunks      assets/chunks_primary.csv \
  --assignments assets/chunk_assignments_primary.csv \
  --outdir s3://r6333-pep-nppc-oi-bmn333-dev/oligogym-repro --version v1 \
  -resume
```

`--arm corrective` with the corrective chunk manifests runs the other arm into a
separate output prefix.  `--only_chunks` / `--max_chunks` / `--compute_class`
restrict a run for smoke testing.  Concurrency is the `OLIGOGYM_QUEUE_SIZE`
environment variable (default 32), because the g4dn vCPU quota is unconfirmed.

## Determinism

Unseeded by default, matching the published methodology (upstream seeds nothing).
`--seed N` makes a run repeatable while preserving the published resampling
scheme; `--proper_kfold` additionally replaces the per-fold reshuffle with a true
disjoint partition and is a documented methodological divergence, for sensitivity
analysis only.  Every output row records `seed_used` and `proper_kfold`.

See `docs/pipeline_notes.md` for the chunking arithmetic and every encoded
constraint with its measured justification.
