# Evaluation

The evaluation scripts separate controlled benchmark behavior from unrestricted free-form behavior.

- `evaluate_nonmath.py` measures non-math controlled factual QA by category.
- `analyze_stage_d_results.py` inspects the composition of the Stage-D paraphrase set.
- `inspect_stage_e_conversation.py` prints held-out conversational intent responses for manual inspection.

The compact held-out JSON sets are stored under `data/eval/`.

Important: controlled benchmark accuracy is not presented as general conversational accuracy. The final project explicitly records that free-form questioning exposed severe intent and knowledge failures despite strong bounded-test scores.
