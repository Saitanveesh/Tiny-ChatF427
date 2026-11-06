# Contributing to TinyChat-F427

Thanks for your interest in compact language models and embedded inference. Contributions are welcome from ML, embedded, systems and documentation developers.

## Before starting

Read [the README](README.md), [project status](docs/PROJECT_STATUS.md) and [roadmap](docs/ROADMAP.md). This is a **research prototype**: the final Transformer has not yet been demonstrated running end-to-end on an STM32F427.

For nontrivial changes, [open an issue](https://github.com/Saitanveesh/Tiny-ChatF427/issues) describing the goal and proposed approach before writing a large patch.

## Good places to start

- **Documentation:** correct broken links, clarify CPU vs MCU results, add platform-specific setup notes.
- **Tests:** add model shape, causality, tokenization, and serialization regression tests.
- **PC performance:** measure PyTorch inference cost with clear hardware and software details.
- **Embedded feasibility:** document numerical precision, quantization error, Flash and SRAM costs, and timing.
- **Reproducibility:** identify historical scripts or artifacts that are missing from the published snapshot and propose recoverable alternatives.

See [the roadmap](docs/ROADMAP.md) for acceptance criteria.

## Local development

```bash
git clone https://github.com/Saitanveesh/Tiny-ChatF427.git
cd Tiny-ChatF427
python -m venv .venv
# Activate the environment for your shell (see docs/GETTING_STARTED.md).
python -m pip install -e .
python -m unittest discover -s tests -v
```

The tests do not require the historical trained checkpoint.

## Pull request checklist

- Keep the scope focused; describe the technical problem and the change.
- Include reproducible commands, environment details and observed output.
- Add or update tests when behavior changes; explain any test that cannot be run.
- Distinguish **measured**, **calculated**, and **planned** metrics.
- Do not claim that the final Transformer runs on STM32F427 until an on-device end-to-end demonstration is available.
- Do not add large generated datasets, model weights or third-party media without discussing provenance, licenses and distribution first.
- For any MCU measurements, include board revision, MCU part number, toolchain, clock, quantization scheme, Flash/SRAM accounting and token latency methodology.

## Reporting issues

Use the [issue tracker](https://github.com/Saitanveesh/Tiny-ChatF427/issues). For code bugs, provide the command, expected and actual result, traceback, OS, Python and package versions. For hardware issues, include target details and a safe reproduction procedure.

**Security issues:** avoid publishing details that could enable abuse of deployed hardware. Open a minimal non-sensitive issue requesting a private reporting channel; do not post secrets or an active exploitation chain in a public issue.

## Community expectations

Be respectful, evidence-led and constructive. Disagreement about accuracy, feasibility or methodology is welcome; personal attacks and harassment are not. Maintainers may close issues or PRs that violate these expectations.

## License

By submitting a contribution, you agree it may be distributed under this project's [MIT License](LICENSE).
