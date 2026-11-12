# log-anomaly

Host-based log anomaly detection: parse authentication, process and network logs,
engineer time-windowed features, score events with three unsupervised models, and
evaluate the result with time-based splits and an explicit base-rate discussion.

**Status:** work in progress. See `docs/design.md` for the plan and `REPORT.md`
for results.

## Quick start

```bash
uv sync
uv run log-anomaly download --dataset all      # fetch + checksum the public datasets
uv run log-anomaly synth --out data/synthetic   # labelled synthetic corpus
uv run log-anomaly train  --log <logfile> --model iforest --out artifacts/model.joblib
uv run log-anomaly score  <logfile> --model artifacts/model.joblib --top-k 20
uv run log-anomaly evaluate --log <logfile> --labels <labels.csv> --out reports/
```

## Data

See `docs/design.md` and `README.md` sections below for sources, licences and
citations. Raw data is never committed.
