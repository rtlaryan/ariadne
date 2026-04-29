# Ariadne

Ariadne is the Python side of the `auxila` calculator-agent project. It trains and evaluates agents that operate the adjacent `icalc` browser calculator through canonical key actions.

## Project layout

- `agents/` — HTTP-serving workers for supervised generation, DAgger rollouts, RL rollouts, and oracle task generation.
- `core/` — tokenizer, dataset/state serialization, action simulation, calculator spec, Python calculator evaluator, and model code.
- `eval/` — diagnostic evaluation suite generation, live workers, scoring, reporting, and CLI orchestration.
- `trainers/` — supervised pretraining, DAgger training, and PPO/RL training entry points.
- `configs/` — experiment, DAgger, RL, and tokenizer configuration.
- `web_ui/` — FastAPI app for interactive checkpoint testing.
- `docs/plans/` — implementation plans and architecture decisions.

Generated datasets, checkpoints, logs, evaluation outputs, and Python caches should stay out of version control.

## Environment

Use the workspace venv from the parent project root:

```bash
cd /home/aryan/projects/auxila
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests -q
```

The `PYTHONPATH` value is important because tests and CLIs import both `ariadne` and the adjacent `icalc` project from the shared `auxila` workspace.

## Canonical calculator contract

Ariadne uses canonical model-facing keys rather than button display text:

- Operators: `+`, `-`, `*`, `/`, `%`, `^`, `!`
- Controls: `Enter`, `Backspace`, `Escape`, `m`
- Scientific keys: `sin`, `cos`, `tan`, `log`, `ln`, `sqrt`, `inv`, `pi`, `e`, `deg`
- State includes `angleMode: "deg" | "rad"`

`ariadne.core.calculator_spec` is the Python source of truth for canonical actions and display conversion. `icalcState.availableInteractions` is expected to expose these same canonical keys.

## Oracle and task generation

`ariadne.agents.oracle.Oracle.generate_task(...)` returns a typed `CalculatorTask` with:

- `expression`
- canonical `plan`
- `expected` evaluator result
- `angle_mode`
- `category`
- deterministic seed metadata
- feature/depth/plan-length metadata

The rich task profile covers arithmetic, decimals, constants, powers, factorials, unary functions, and degree/radian trigonometry. Generated training tasks avoid invalid math; diagnostic evaluation suites include explicit error-state cases.

## Evaluation harness

The diagnostic evaluation CLI is:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m ariadne.eval.run \
  --config /home/aryan/projects/auxila/ariadne/configs/experiment.yaml \
  --suite smoke \
  --dry-run-suite \
  --output-dir /tmp/ariadne_eval_smoke
```

A dry run writes:

- `suite.jsonl`
- `manifest.json`

A live run also writes:

- `episodes.ndjson`
- `summary.json`
- `summary.md`

Evaluation reports separate entry correctness, result correctness, expected-error correctness, and overall success, then aggregate by bucket, stratum, feature, and failure reason.

## Common commands

Run targeted rich-oracle/eval tests:

```bash
cd /home/aryan/projects/auxila
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest \
  tests/test_calculator_spec.py \
  tests/test_action_sim.py \
  tests/test_icalc_state_contract.py \
  tests/test_calculator_eval.py \
  tests/test_oracle_rich_generation.py \
  tests/test_tokenizer_rich_actions.py \
  tests/test_eval_suite.py \
  tests/test_eval_scoring.py \
  tests/test_eval_reporting.py \
  tests/test_orchestrate_evaluation.py \
  tests/test_orchestrate_guards.py \
  -q
```

Run orchestration:

```bash
cd /home/aryan/projects/auxila
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m ariadne.orchestrate \
  --config ariadne/configs/experiment.yaml
```

Run the checkpoint web UI:

```bash
cd /home/aryan/projects/auxila
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/uvicorn ariadne.web_ui.app:app --reload --port 7000
```
