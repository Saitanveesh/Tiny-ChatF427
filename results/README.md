# Recorded experiment results

This directory contains the logs and machine-readable histories retained from the TinyChat-F427 training run.

Key files:

- `logs/full_training.log` — 30M-token baseline training output
- `training_history.json` — baseline validation history
- `training_summary.json` — baseline summary
- `training_qualification.json` — learning-rate qualification results
- `logs/stage_c_v2_training.log` — Stage-C v2 specialization
- `logs/stage_d_v2_training.log` and `stage_d_v2_history.json` — Stage-D v2 factual/refusal specialization
- `logs/stage_e_training.log` and `stage_e_history.json` — Stage-E behavior repair
- `logs/stage_f_training.log` — final Stage-F repair

Final reported Stage-F candidate:

```text
Factual accuracy      : 84.1%
Conversation semantic : 75.0%
Conversation exact    : 75.0%
Unknown/refusal       : 100.0%
```

These are controlled held-out results, not open-ended general-QA accuracy.
