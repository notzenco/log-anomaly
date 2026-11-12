# log-anomaly

Host-based log anomaly detection: parse authentication, process and network
logs into one event schema, engineer causal time-windowed features, score
events with three unsupervised models, and evaluate the result with
time-based splits, explicit base rates and alert volumes an analyst could
actually live with.

**Headline finding** (SYNTHETIC evaluation corpus — see below): at a single
event-level threshold, a statistical rarity baseline and Isolation Forest
detect brute-force and credential-stuffing bursts but are blind to singleton
attacks (rare process execution, data exfiltration, new-host and off-hours
logins), while a linear autoencoder (PCA reconstruction error) detects 53% of
attack *episodes* against their 2–4%, at a slightly worse PR-AUC (0.47 vs
0.50). Aggregate precision/recall hides this; per-attack-type and episode
metrics expose it. Full results, charts and a false-positive analysis:
[REPORT.md](REPORT.md). The protocol was fixed before implementation:
[docs/design.md](docs/design.md).

## Quick start

```bash
uv sync
uv run log-anomaly synth --out data/synthetic        # labelled SYNTHETIC corpus
uv run log-anomaly evaluate \
    --log data/synthetic/synth.log --labels data/synthetic/labels.csv \
    --out reports --corpus synthetic
uv run log-anomaly download --dataset all            # real Loghub logs, checksum-verified
uv run log-anomaly train --log data/raw/logs/Linux.log --model rarity \
    --out artifacts/linux.joblib --anchor-year 2004
uv run log-anomaly score data/raw/logs/Linux.log \
    --model artifacts/linux.joblib --top-k 20 --anchor-year 2004
```

`score` prints ranked alerts, each with the features that drove its score:

```
  1  score=62.16  2024-05-19 02:29:02+00:00  host05  auth_failure  user=dave.clarke  src=194.26.40.222
     reason: hour_profile_distance=7.93, off_hours=7.01, src_user_entropy_1h=6.43
     log: May 19 02:29:02 host05 sshd(pam_unix)[185773]: authentication failure; ...
```

## Commands

| Command | Does |
| --- | --- |
| `log-anomaly download [--dataset linux\|openssh\|proxifier\|all]` | Fetch a public dataset and verify its MD5 against the pinned value |
| `log-anomaly synth [--out DIR] [--days N] [--seed N]` | Generate the labelled SYNTHETIC corpus |
| `log-anomaly train --log FILE --model rarity\|iforest\|pca-ae --out MODEL.joblib` | Fit a model on a whole log file |
| `log-anomaly score FILE --model MODEL.joblib [--top-k N] [--threshold X]` | Ranked alerts with per-alert reasons (top contributing features) |
| `log-anomaly evaluate --log FILE --labels LABELS.csv --out DIR` | Time-split evaluation: PR-AUC, precision/recall at F1/F5/20-alerts-per-day points, per-attack-type recall, episode detection |

Syslog timestamps carry no year; pass `--anchor-year` (the December→January
rollover is handled). Synthetic inputs are detected from their file header,
announced on stderr, and recorded in model/evaluation metadata.

Note: `score` computes features from the start of the file it is given, so
score whole files, not short slices — every user looks new at a file boundary.

## Data

**Raw data is never committed** (see `.gitignore`). Fetch it with
`log-anomaly download` or `scripts/download_data.sh`; both verify MD5
checksums pinned in `src/log_anomaly/data/datasets.py`.

### Real data: Loghub

Source: the [Loghub](https://github.com/logpai/loghub) collection (Zenodo
DOI [10.5281/zenodo.8196385](https://doi.org/10.5281/zenodo.8196385)).

Licence: free for research or academic work, provided the Loghub repository
is referenced and the paper is cited:

> Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. *Loghub: A
> Large Collection of System Log Datasets for AI-driven Log Analytics.* IEEE
> International Symposium on Software Reliability Engineering (ISSRE), 2023.

| Dataset | Archive MD5 | Content used |
| --- | --- | --- |
| Linux | `6d1802d7778126f21c001c6aa7b6b106` | `/var/log/messages`, one server, ~264 days: SSH/PAM auth, sessions, services |
| OpenSSH | `a44e40b4348697dabf2bba56885e0a38` | OpenSSH server log, 28 days, 655k lines: 182 successes vs 505k failures |
| Proxifier | `2612fce12cc3d16599ddb3db8e9c477a` | Windows proxy log, 21k lines: application → destination, bytes, duration |

**How to use the real data.** These corpora have **no ground-truth labels**,
so no honest precision/recall exists for them. Use them with `train` +
`score`, then audit the top alerts by hand, as REPORT.md §6 does for the
Linux log (top 30 `rarity` alerts: 30/30 remote password guessing, from 8
distinct sources). `evaluate` requires a labels CSV (`event_index,label,attack_type`)
aligned to parsed events — the synthetic corpus provides one.

Candidates considered and rejected (LANL Unified/Comprehensive datasets
behind registration, CICIDS2017 network flows, multi-GB HDFS/BGL system
logs) are discussed in [docs/design.md](docs/design.md) §2.

### Synthetic data

`log-anomaly synth` generates a 120-day, 6-host, 120-user corpus (273k
events, base rate 1.36%) in BSD syslog shape, with nine labelled attack
families: background scanning, brute force, credential stuffing, new-host
and off-hours logins, privilege escalation, rare process execution, data
exfiltration and lateral movement. **It is synthetic**: the file begins with
a `# SYNTHETIC LOG` header, its metadata says so, and every result derived
from it is labelled SYNTHETIC in this README, the CLI output and REPORT.md.
It exists because the real corpora are unlabelled; its numbers are a
regression test of the pipeline, not a capability claim.

## Method in one paragraph

Each log line becomes a normalised event; 31 features are computed per event
from strictly trailing windows (failure bursts per user/source, source user
entropy, new user–host pairs, learned working-hours profiles, rare
process-parent pairs, new destinations and byte-volume z-scores), so offline
scoring reproduces what an online scorer would have seen. Three models with
different inductive biases score the same features: a sum-of-surprise
statistical baseline, Isolation Forest, and PCA reconstruction error (chosen
over a deep autoencoder to keep dependencies light — no torch — while
keeping exact per-feature attribution). Evaluation splits by time duration,
fits on train, picks thresholds on validation, and reports on test with the
base rate next to every precision number.

## Development

```bash
uv sync
uv run --locked pytest      # includes a tiny committed fixture; no downloads
uv run --locked ruff check .
uv run python scripts/make_report.py   # regenerates REPORT.md's numbers/charts
```

Layout: `src/log_anomaly/{parsing,features,models,evaluation,data}` plus the
CLI; `tests/` mirrors it; `docs/design.md` is the spec; `REPORT.md` the
results. CI (GitHub Actions) runs ruff + pytest on Python 3.11/3.12 against
the fixture only.

## License

Code: MIT — see [LICENSE](LICENSE). The Loghub datasets are *not* covered by
this license and are not distributed here; they remain under Loghub's
research/academic terms above. The synthetic corpus is generated by this
project's code and carries the same MIT license.
