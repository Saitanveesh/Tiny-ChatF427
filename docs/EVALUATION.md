# Evaluation

TinyChat-F427 is evaluated as a bounded experimental model, not as a general-purpose assistant.

## Controlled results

The final Stage-F candidate reported:

| Evaluation | Result |
|---|---:|
| Factual accuracy | **84.1%** |
| Conversation semantic accuracy | **75.0%** |
| Conversation exact accuracy | **75.0%** |
| Unknown/refusal | **100.0%** |

Earlier Stage-D v2 true holdout testing reported:

| Evaluation | Result |
|---|---:|
| Known-fact accuracy | 43.2% |
| Arithmetic accuracy | 14.3% |
| Unknown/refusal | 100.0% |

That 43.2% aggregate included 91 arithmetic-style examples among 179 paraphrase examples. When arithmetic and conversational examples were removed, the non-math factual subset contained 82 examples and achieved approximately **86.6%** before later conversation repair.

## Why the controlled score is not enough

A manual free-form interactive test exposed severe failures outside the curated distributions. Representative observed outputs included:

```text
> are you gpt
Gravity.

> why is sky black
Blood transports oxygen, nutrients, hormones, and waste products through the body.

> define speed
Goodbye.

> what is speed of light
The heart pumps blood through the body.
```

This is an important project result. A small model can score well on bounded paraphrase/intent tests while remaining unreliable in unrestricted interaction.

Accordingly, this repository does **not** claim 84.1% accuracy on arbitrary general-knowledge questions.

## Arithmetic

The Transformer did not learn a reliable arithmetic algorithm. The PC chat demo therefore includes a deterministic router for simple binary integer arithmetic. Router accuracy must not be reported as neural-model arithmetic accuracy.

## Re-running evaluations

The compact holdout JSON files are kept under `data/eval/`. The original evaluation scripts are under `evaluation/`. Some scripts reference the original `data/stage_d_v2` paths; when reproducing historical runs, either regenerate the data through the training builders or mirror the provided evaluation JSONs to those paths.
