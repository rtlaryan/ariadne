# Rich Oracle and Evaluation Rework Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Replace Ariadne's limited calculator oracle/data generation and current evaluation harness with a clean, typed, feature-complete task-generation system and a robust diagnostic evaluation harness.

**Architecture:** Introduce a central calculator capability spec, canonical calculator state/actions, a typed `CalculatorTask` API, Python-side calculator evaluation for expected metadata, and a modular evaluation package with suite generation, scoring, worker orchestration, and reporting separated. This is a clean replacement: deprecated tuple-style oracle APIs, display-label data compatibility, and old eval schemas should be removed rather than preserved.

**Tech Stack:** Python 3 via `/home/aryan/projects/auxila/.venv`, PyTorch, FastAPI, Selenium bridge, pytest, YAML configs, JavaScript calculator frontend.

---

## Implementation Status

Completed locally through non-browser verification on 2026-04-28.

- Milestone A implementation is complete: canonical iCalc state/actions, typed `CalculatorTask` generation, rich task metadata, action simulation, data-generation metadata logging, and tokenizer/dataset support are in place.
- Milestone B implementation is complete up to the network/browser boundary: diagnostic suite generation, manifest writing, eval schemas, scoring, reporting, worker record shape, orchestrator wiring, and dry-run suite generation are in place.
- Verification completed locally with the project venv:
  - `PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests -q`
  - Result: `53 passed, 1 skipped`.
  - `PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m ariadne.eval.run --config /home/aryan/projects/auxila/ariadne/configs/experiment.yaml --suite smoke --dry-run-suite --output-dir /tmp/ariadne_eval_smoke`
  - Result: dry-run artifacts written under `/tmp/ariadne_eval_smoke/smoke` with 50 suite cases.
- Browser-backed end-to-end eval runs are intentionally excluded from local completion because those runs are expected to execute on a separate network machine and will be tested manually.

---

## Locked Product/Architecture Decisions

1. **Milestone A scope:** rich expression generation plus deg/rad angle-mode tasks; memory keys/stateful memory tasks are deferred.
2. **Canonical actions:** `window.icalcState.availableInteractions` exposes canonical keys, not button display text.
3. **Angle state:** expose `angleMode: "deg" | "rad"` in `icalcState`.
4. **Oracle API:** use a typed `CalculatorTask` API. Do not preserve the old `(expr, plan)` API as a compatibility goal.
5. **Generation profiles:** built-in named profiles with lightweight YAML overrides.
6. **Invalid math:** training/data generation avoids invalid expressions; evaluation includes explicit error-state buckets.
7. **Eval architecture:** modular rewrite inside `ariadne/eval/`, not `eval2`.
8. **Eval buckets:** feature-first diagnostic strata.
9. **Unseen guarantee:** exact canonical unseen + reserved eval seed ranges now; design toward future structural holdout presets.
10. **Eval success:** report entry correctness and result correctness separately; normal-task overall success requires both.
11. **Expected values:** Python evaluator generates expected metadata; browser terminal state is used for live scoring.
12. **Reporting:** file-based rich diagnostics: `suite.jsonl`, `manifest.json`, `episodes.ndjson`, `summary.json`, `summary.md`.
13. **Migration:** clean replacement; remove deprecated compatibility bloat.
14. **Workflow:** strict TDD with small commits.

---

## Commands and Environment

Use the project venv for all Python commands:

```bash
cd /home/aryan/projects/auxila
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests -q
```

Targeted commands used throughout this plan:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_calculator_spec.py -q
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_action_sim.py -q
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_calculator_eval.py -q
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_oracle_rich_generation.py -q
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_suite.py -q
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_scoring.py -q
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_reporting.py -q
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_orchestrate_evaluation.py tests/test_orchestrate_guards.py -q
```

---

# Milestone A: Rich Oracle and Data Generation

## Task A1: Add central calculator capability spec

**Objective:** Create a single source of truth for canonical calculator keys, features, and display conversion.

**Files:**
- Create: `ariadne/core/calculator_spec.py`
- Create: `tests/test_calculator_spec.py`

**Step 1: Write failing tests**

Create `tests/test_calculator_spec.py` with tests for canonical key sets and display conversion:

```python
import unittest

from ariadne.core.calculator_spec import (
    BASIC_ACTIONS,
    SCIENTIFIC_ACTIONS,
    canonicalize_key,
    display_for_key,
    is_function_key,
    text_for_action,
)


class CalculatorSpecTests(unittest.TestCase):
    def test_display_symbols_canonicalize_to_model_keys(self) -> None:
        self.assertEqual(canonicalize_key("÷"), "/")
        self.assertEqual(canonicalize_key("×"), "*")
        self.assertEqual(canonicalize_key("⌫"), "Backspace")
        self.assertEqual(canonicalize_key("AC"), "Escape")
        self.assertEqual(canonicalize_key("="), "Enter")
        self.assertEqual(canonicalize_key("√"), "sqrt")
        self.assertEqual(canonicalize_key("π"), "pi")

    def test_actions_include_rich_expression_keys_but_not_memory(self) -> None:
        for key in [".", "%", "^", "!", "pi", "e", "sqrt", "inv", "deg"]:
            self.assertIn(key, BASIC_ACTIONS | SCIENTIFIC_ACTIONS)
        for key in ["mc", "m+", "m-", "mr"]:
            self.assertNotIn(key, BASIC_ACTIONS | SCIENTIFIC_ACTIONS)

    def test_function_and_constant_display_text(self) -> None:
        self.assertTrue(is_function_key("sqrt"))
        self.assertEqual(text_for_action("sqrt"), "sqrt(")
        self.assertEqual(text_for_action("inv"), "inv(")
        self.assertEqual(text_for_action("pi"), "π")
        self.assertEqual(display_for_key("sqrt"), "√")
        self.assertEqual(display_for_key("pi"), "π")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_calculator_spec.py -q
```

Expected: FAIL because `ariadne.core.calculator_spec` does not exist.

**Step 3: Implement `ariadne/core/calculator_spec.py`**

Add:

```python
"""Canonical calculator capability specification for Ariadne and icalc."""

from __future__ import annotations

DIGITS = frozenset("0123456789")
BASIC_OPERATORS = frozenset({"+", "-", "*", "/", "%"})
SCIENTIFIC_FUNCTIONS = frozenset({"sin", "cos", "tan", "log", "ln", "sqrt", "inv"})
CONSTANTS = frozenset({"pi", "e"})
POSTFIX_OPERATORS = frozenset({"!"})
INFIX_SCIENTIFIC_OPERATORS = frozenset({"^"})
GROUPING = frozenset({"(", ")"})
CONTROL_ACTIONS = frozenset({"Enter", "Backspace", "Escape", "m"})
ANGLE_ACTIONS = frozenset({"deg"})

BASIC_ACTIONS = DIGITS | BASIC_OPERATORS | frozenset({".", "Enter", "Backspace", "Escape", "m"})
SCIENTIFIC_ACTIONS = GROUPING | SCIENTIFIC_FUNCTIONS | CONSTANTS | POSTFIX_OPERATORS | INFIX_SCIENTIFIC_OPERATORS | ANGLE_ACTIONS
ALL_ACTIONS = BASIC_ACTIONS | SCIENTIFIC_ACTIONS

DISPLAY_TO_CANONICAL = {
    "÷": "/",
    "×": "*",
    "⌫": "Backspace",
    "AC": "Escape",
    "=": "Enter",
    "√": "sqrt",
    "π": "pi",
}

CANONICAL_TO_DISPLAY = {
    "/": "÷",
    "*": "×",
    "Backspace": "⌫",
    "Escape": "AC",
    "Enter": "=",
    "sqrt": "√",
    "pi": "π",
}


def canonicalize_key(key: str) -> str:
    return DISPLAY_TO_CANONICAL.get(str(key), str(key))


def display_for_key(key: str) -> str:
    return CANONICAL_TO_DISPLAY.get(str(key), str(key))


def is_function_key(key: str) -> bool:
    return canonicalize_key(key) in SCIENTIFIC_FUNCTIONS


def text_for_action(key: str) -> str:
    key = canonicalize_key(key)
    if key in SCIENTIFIC_FUNCTIONS:
        return f"{key}("
    if key == "pi":
        return "π"
    return key
```

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_calculator_spec.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add core/calculator_spec.py ../tests/test_calculator_spec.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: add calculator capability spec"
```

---

## Task A2: Update action simulation to use canonical spec

**Objective:** Make `apply_key_to_readout()` mirror canonical expression-building behavior for rich actions.

**Files:**
- Modify: `ariadne/core/action_sim.py`
- Modify: `tests/test_action_sim.py`

**Step 1: Write failing tests**

Extend `tests/test_action_sim.py` with:

```python
from ariadne.core.action_sim import apply_key_to_readout, simulate_plan


def test_rich_expression_actions_update_readout():
    assert apply_key_to_readout("", "pi") == "π"
    assert apply_key_to_readout("", "e") == "e"
    assert apply_key_to_readout("2", "^") == "2^"
    assert apply_key_to_readout("5", "!") == "5!"
    assert apply_key_to_readout("", "inv") == "inv("
    assert apply_key_to_readout("", "sqrt") == "sqrt("


def test_angle_toggle_does_not_change_readout():
    assert apply_key_to_readout("sin(30)", "deg") == "sin(30)"


def test_simulate_plan_reconstructs_expression_before_enter():
    plan = ["m", "sqrt", "9", ")", "+", "2", ".", "5", "Enter"]
    assert simulate_plan(plan) == "sqrt(9)+2.5"
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_action_sim.py -q
```

Expected: FAIL because `simulate_plan` does not exist and action sim may not cover all keys.

**Step 3: Implement**

Update `ariadne/core/action_sim.py` to import from `calculator_spec` and add `simulate_plan()`.

Key behavior:

```python
from ariadne.core.calculator_spec import SCIENTIFIC_FUNCTIONS, canonicalize_key, text_for_action

SMART_BACKSPACE_SUFFIXES = tuple(f"{name}(" for name in sorted(SCIENTIFIC_FUNCTIONS, key=len, reverse=True))


def apply_key_to_readout(current: str, key: str) -> str:
    key = canonicalize_key(key)
    if key in {"m", "deg", "Enter"}:
        return current
    if key == "Backspace":
        for suffix in SMART_BACKSPACE_SUFFIXES:
            if current.endswith(suffix):
                return current[: -len(suffix)]
        return current[:-1] if current else ""
    if key == "Escape":
        return ""
    return current + text_for_action(key)


def simulate_plan(plan: list[str]) -> str:
    readout = ""
    for key in plan:
        readout = apply_key_to_readout(readout, key)
    return readout
```

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_action_sim.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add core/action_sim.py ../tests/test_action_sim.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: simulate rich calculator actions"
```

---

## Task A3: Expose canonical actions and angleMode from icalc

**Objective:** Make browser state canonical and expose degree/radian mode.

**Files:**
- Modify: `icalc/script.js`
- Modify: `icalc/README.md`
- Create: `tests/test_icalc_state_contract.py`

**Step 1: Write failing contract test**

Create `tests/test_icalc_state_contract.py` to statically check the frontend contract:

```python
from pathlib import Path

ICALC_SCRIPT = Path(__file__).resolve().parents[1] / "icalc" / "script.js"


def test_icalc_exposes_angle_mode_and_canonical_action_mapping():
    src = ICALC_SCRIPT.read_text()
    assert "angleMode" in src
    assert "canonical" in src or "canonicalize" in src
    assert "data-value" in src or "dataset.value" in src
    assert "actionToKey" in src
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_icalc_state_contract.py -q
```

Expected: FAIL because `angleMode` is not exposed yet.

**Step 3: Modify `icalc/script.js`**

In `updateExposedState()`, replace button text extraction with canonical action extraction.

Implementation shape:

```js
const actionToKey = {
    'calculate': 'Enter',
    'delete': 'Backspace',
    'all-clear': 'Escape',
};

const valueToKey = {
    'pi': 'pi',
    'sqrt': 'sqrt',
};

const canonicalizeButton = (btn) => {
    const action = btn.dataset.action;
    const value = btn.dataset.value;
    if (action && actionToKey[action]) return actionToKey[action];
    if (value) return valueToKey[value] || value;
    return null;
};

const interactions = Array.from(document.querySelectorAll('.btn'))
    .filter(btn => {
        if (this.mode === 'basic' && btn.closest('.scientific-pad')) return false;
        return btn.offsetParent !== null;
    })
    .map(canonicalizeButton)
    .filter(Boolean);

interactions.push('m');

const state = {
    readout: this.currentValue,
    history: this.history,
    mode: this.mode,
    angleMode: this.isDegree ? 'deg' : 'rad',
    lastAction: this.lastAction,
    availableInteractions: interactions,
    error: this.error,
    memory: this.memory
};
```

Do not include memory actions in Milestone A generated tasks, but if buttons remain visible in scientific mode, decide whether to expose them. Preferred for this milestone: remove memory buttons from available interactions by omitting `memory-*` mappings until stateful tasks are implemented.

**Step 4: Update README**

Update `icalc/README.md` protocol examples to show canonical keys and `angleMode`.

**Step 5: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_icalc_state_contract.py -q
```

Expected: PASS.

**Step 6: Commit in `icalc` and `ariadne` as needed**

Because `icalc` and `ariadne` are separate repos, commit frontend changes in `icalc`, test changes in `ariadne` if tests are tracked there.

```bash
git -C /home/aryan/projects/auxila/icalc add script.js README.md
git -C /home/aryan/projects/auxila/icalc commit -m "feat: expose canonical calculator state"

git -C /home/aryan/projects/auxila/ariadne add ../tests/test_icalc_state_contract.py
git -C /home/aryan/projects/auxila/ariadne commit -m "test: document icalc state contract"
```

---

## Task A4: Add Python calculator evaluator

**Objective:** Evaluate calculator expressions in Python for suite metadata, safe generation, and result scoring expectations.

**Files:**
- Create: `ariadne/core/calculator_eval.py`
- Create: `tests/test_calculator_eval.py`

**Step 1: Write failing tests**

Create tests covering basic, decimal, constants, functions, angle modes, and errors:

```python
import math
import unittest

from ariadne.core.calculator_eval import evaluate_expression


class CalculatorEvalTests(unittest.TestCase):
    def test_evaluates_decimal_arithmetic(self) -> None:
        result = evaluate_expression("1.5+2.25", angle_mode="deg")
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.value, 3.75)

    def test_evaluates_constants_and_power(self) -> None:
        result = evaluate_expression("pi^2", angle_mode="deg")
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.value, math.pi ** 2, places=10)

    def test_evaluates_trig_in_degrees_and_radians(self) -> None:
        self.assertAlmostEqual(evaluate_expression("sin(30)", angle_mode="deg").value, 0.5, places=10)
        self.assertAlmostEqual(evaluate_expression("sin(pi/2)", angle_mode="rad").value, 1.0, places=10)

    def test_evaluates_calculator_functions(self) -> None:
        self.assertAlmostEqual(evaluate_expression("sqrt(9)", angle_mode="deg").value, 3.0)
        self.assertAlmostEqual(evaluate_expression("log(100)", angle_mode="deg").value, 2.0)
        self.assertAlmostEqual(evaluate_expression("ln(e)", angle_mode="deg").value, 1.0)
        self.assertAlmostEqual(evaluate_expression("inv(4)", angle_mode="deg").value, 0.25)
        self.assertAlmostEqual(evaluate_expression("5!", angle_mode="deg").value, 120.0)

    def test_reports_expected_errors(self) -> None:
        self.assertFalse(evaluate_expression("sqrt(-1)", angle_mode="deg").ok)
        self.assertFalse(evaluate_expression("log(-5)", angle_mode="deg").ok)
        self.assertFalse(evaluate_expression("1/0", angle_mode="deg").ok)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_calculator_eval.py -q
```

Expected: FAIL because module does not exist.

**Step 3: Implement evaluator**

Create `ariadne/core/calculator_eval.py` with:

- `CalculatorEvalResult` dataclass
- `evaluate_expression(expression: str, angle_mode: str = "deg")`
- safe AST-based evaluator or tightly controlled parser

Do **not** use raw `eval()`.

Supported semantics:

- `π` and `pi` => `math.pi`
- `e` => `math.e`
- `^` => exponentiation
- `!` => factorial for non-negative integers up to constraint
- `%` => modulo
- `sin/cos/tan` deg/rad aware
- `log` base 10
- `ln` natural log
- `sqrt`
- `inv(x)` => `1 / x`
- result rounded/displayed similarly to JS `toPrecision(12)` where practical

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_calculator_eval.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add core/calculator_eval.py ../tests/test_calculator_eval.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: add calculator expression evaluator"
```

---

## Task A5: Replace oracle with typed CalculatorTask generation

**Objective:** Replace the old tuple-style oracle with typed rich tasks and feature metadata.

**Files:**
- Modify: `ariadne/agents/oracle.py`
- Create: `tests/test_oracle_rich_generation.py`

**Step 1: Write failing tests**

Create tests:

```python
import unittest

from ariadne.agents.oracle import CalculatorTask, Oracle
from ariadne.core.action_sim import simulate_plan
from ariadne.core.calculator_eval import evaluate_expression


class RichOracleTests(unittest.TestCase):
    def test_generate_task_returns_calculator_task(self) -> None:
        task = Oracle(profile="rich_v1").generate_task(seed=123, category="decimal_arithmetic")
        self.assertIsInstance(task, CalculatorTask)
        self.assertIn("decimal", task.features)
        self.assertEqual(simulate_plan(task.plan), task.expression)

    def test_task_generation_is_deterministic_without_global_random(self) -> None:
        oracle = Oracle(profile="rich_v1")
        a = oracle.generate_task(seed=42, category="constants")
        b = oracle.generate_task(seed=42, category="constants")
        self.assertEqual(a, b)

    def test_trig_tasks_include_angle_mode_metadata(self) -> None:
        task = Oracle(profile="rich_v1").generate_task(seed=7, category="angle_trig")
        self.assertIn(task.angle_mode, {"deg", "rad"})
        self.assertIn("trig", task.features)

    def test_safe_training_tasks_evaluate_successfully(self) -> None:
        oracle = Oracle(profile="rich_v1")
        for category in ["decimal_arithmetic", "constants", "power", "factorial", "unary_function", "angle_trig", "mixed_scientific"]:
            task = oracle.generate_task(seed=100, category=category)
            result = evaluate_expression(task.expression, angle_mode=task.angle_mode or "deg")
            self.assertTrue(result.ok, (category, task))
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_oracle_rich_generation.py -q
```

Expected: FAIL because the old oracle lacks typed API.

**Step 3: Implement typed oracle**

In `ariadne/agents/oracle.py`:

- Add `CalculatorTask` dataclass.
- Make `Oracle.generate_task(...) -> CalculatorTask`.
- Use `random.Random(seed)`, never global `random.seed()`.
- Support categories:
  - `basic_integer`
  - `decimal_arithmetic`
  - `parentheses`
  - `modulo`
  - `power`
  - `factorial`
  - `constants_pi`
  - `constants_e`
  - `sqrt`
  - `inv`
  - `trig_deg`
  - `trig_rad`
  - `log`
  - `ln`
  - `mixed_scientific`
  - `angle_trig`
- Generate plans from expressions using canonical keys and mode/angle toggles.

Suggested task fields:

```python
@dataclass(frozen=True)
class CalculatorTask:
    expression: str
    canonical_expression: str
    plan: list[str]
    features: frozenset[str]
    depth: int
    token_length: int
    scientific: bool
    angle_mode: str | None
    category: str
    seed: int | None
    goal_type: str = "expression"
    expected_value: float | None = None
    expected_display: str | None = None
    expected_error: str | None = None
```

**Step 4: Remove old compatibility paths**

Update tests and direct callers later; do not keep old tuple API as final behavior.

**Step 5: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_oracle_rich_generation.py tests/test_action_sim.py tests/test_calculator_eval.py -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add agents/oracle.py ../tests/test_oracle_rich_generation.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: replace oracle with typed rich task generation"
```

---

## Task A6: Update tokenizer/dataset serialization for canonical rich state

**Objective:** Serialize new canonical calculator state, including angle mode, with no legacy display-label compatibility goal.

**Files:**
- Modify: `ariadne/core/tokenizer.py`
- Modify: `ariadne/core/dataset.py`
- Modify: `ariadne/configs/tokenizer.json`
- Modify/Create tests as needed: `tests/test_dataset_metrics.py`, `tests/test_tokenizer_rich_actions.py`

**Step 1: Write failing tests**

Add `tests/test_tokenizer_rich_actions.py`:

```python
from ariadne.core.dataset import StateSerializer
from ariadne.core.tokenizer import TokenMap


def test_tokenizer_seed_contains_rich_canonical_actions():
    tok = TokenMap.load("/home/aryan/projects/auxila/ariadne/configs/tokenizer.json")
    for key in [".", "%", "^", "!", "pi", "e", "sqrt", "inv", "deg", "angleMode:", "rad"]:
        assert key in tok.token_to_id


def test_state_serializer_includes_angle_mode_and_canonical_keys():
    tok = TokenMap()
    for key in ["mode:", "angleMode:", "scientific", "rad", "keys:", "sqrt", "pi"]:
        tok.add_token(key)
    serializer = StateSerializer(tok)
    tokens = serializer.serialize({
        "mode": "scientific",
        "angleMode": "rad",
        "availableInteractions": ["sqrt", "pi"],
    })
    assert "angleMode:" in tokens
    assert "rad" in tokens
    assert "sqrt" in tokens
    assert "pi" in tokens
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_tokenizer_rich_actions.py -q
```

Expected: FAIL until tokenizer/serializer are updated.

**Step 3: Implement**

- Add angle mode serialization in `StateSerializer.serialize()`.
- Remove old `_KEY_NORMALIZE` dependency or replace with central canonical spec only at direct system boundaries.
- Update tokenizer seed with rich keys and state labels.

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_tokenizer_rich_actions.py tests/test_dataset_metrics.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add core/tokenizer.py core/dataset.py configs/tokenizer.json ../tests/test_tokenizer_rich_actions.py ../tests/test_dataset_metrics.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: serialize canonical rich calculator state"
```

---

## Task A7: Integrate typed tasks into data-generation agents

**Objective:** Update supervised, DAgger, and RL agents to consume `CalculatorTask` directly.

**Files:**
- Modify: `ariadne/agents/gen_agent.py`
- Modify: `ariadne/agents/dagger_agent.py`
- Modify: `ariadne/agents/rl_agent.py`
- Modify: existing tests for those agents

**Step 1: Write/adjust failing tests**

Update tests to assert task objects are handled and logged with metadata:

- `tests/test_dagger_agent.py`
- `tests/test_rl_ppo.py`

Add assertions that logged rows include:

```text
task
task_canonical
features
angle_mode
goal_type
expected_value
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_dagger_agent.py tests/test_rl_ppo.py -q
```

Expected: FAIL due to old tuple handling.

**Step 3: Implement**

Update agent internals:

- `current_task` stores a `CalculatorTask`.
- `expr` becomes `task.expression`.
- `plan` becomes `task.plan`.
- DAgger trajectory uses `task.plan`.
- RL reward uses `task.expression`, `task.angle_mode`.
- JSONL rows include task metadata.
- No old tuple compatibility.

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_dagger_agent.py tests/test_rl_ppo.py tests/test_oracle_rich_generation.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add agents/gen_agent.py agents/dagger_agent.py agents/rl_agent.py ../tests/test_dagger_agent.py ../tests/test_rl_ppo.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: use typed calculator tasks in agents"
```

---

## Task A8: Add task-generation config profiles and orchestrator plumbing

**Objective:** Add built-in `rich_v1` profile and lightweight YAML overrides, then pass config into agents.

**Files:**
- Modify: `ariadne/agents/oracle.py` or create `ariadne/core/task_profiles.py`
- Modify: `ariadne/orchestrate.py`
- Modify: `ariadne/configs/experiment.yaml`
- Modify: `ariadne/configs/dagger.yaml`
- Modify: `ariadne/configs/rl.yaml`
- Modify: `tests/test_orchestrate_guards.py`

**Step 1: Write failing tests**

Add tests that orchestrator command construction passes `--task-profile` / `--task-config` or equivalent into agent subprocesses.

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_orchestrate_guards.py -q
```

Expected: FAIL until plumbing exists.

**Step 3: Implement config shape**

Add YAML:

```yaml
task_generation:
  profile: "rich_v1"
  seed_offset: 0
  max_depth: 6
  weights:
    basic_integer: 0.10
    decimal_arithmetic: 0.15
    parentheses: 0.10
    modulo: 0.05
    power: 0.10
    factorial: 0.05
    constants: 0.10
    unary_function: 0.15
    angle_trig: 0.10
    mixed_scientific: 0.10
  constraints:
    max_integer: 100
    decimal_places: [1, 2]
    max_factorial_arg: 8
    max_abs_value: 1000000
```

Pass config to gen/DAgger/RL agent CLIs. Prefer a JSON string/path argument over many individual CLI flags.

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_orchestrate_guards.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add orchestrate.py configs/experiment.yaml configs/dagger.yaml configs/rl.yaml agents/oracle.py ../tests/test_orchestrate_guards.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: configure rich task generation"
```

---

# Milestone B: Diagnostic Evaluation Harness

## Task B1: Add eval schema dataclasses

**Objective:** Define stable typed records for eval cases, manifests, episodes, and step diagnostics.

**Files:**
- Create: `ariadne/eval/schema.py`
- Create: `tests/test_eval_schema.py`

**Step 1: Write failing tests**

Test JSON round-tripping for `EvalCase` and manifest.

**Step 2: Implement**

Dataclasses:

```python
@dataclass(frozen=True)
class EvalCase:
    task: str
    task_canonical: str
    goal_type: str
    bucket: str
    stratum: str
    features: tuple[str, ...]
    depth: int
    token_length: int
    oracle_plan: list[str]
    oracle_plan_length: int
    scientific: bool
    angle_mode: str | None
    seed: int
    expected_value: float | None
    expected_display: str | None
    expected_error: str | None

@dataclass(frozen=True)
class EvalManifest:
    preset: str
    generator_profile: str
    seed: int
    counts_by_bucket: dict[str, int]
    counts_by_stratum: dict[str, int]
    counts_by_feature: dict[str, int]
    seen_task_count: int
    max_training_depth: int
```

**Step 3: Verify and commit**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_schema.py -q
```

Commit:

```bash
git -C /home/aryan/projects/auxila/ariadne add eval/schema.py ../tests/test_eval_schema.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: define evaluation schemas"
```

---

## Task B2: Rewrite eval suite generation with feature-first strata

**Objective:** Generate deterministic diagnostic suites with metadata, error buckets, and unseen guarantees.

**Files:**
- Replace/Modify: `ariadne/eval/suite.py`
- Modify: `ariadne/eval/common.py`
- Modify: `tests/test_eval_suite.py`

**Step 1: Write failing tests**

Update tests to assert:

- deterministic generation
- exact canonical unseen exclusion
- reserved seed ranges
- all expected buckets/strata present in smoke preset
- expected values present for normal tasks
- expected errors present for error tasks

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_suite.py -q
```

Expected: FAIL against old suite.

**Step 3: Implement strata**

Buckets/strata:

```text
core_unseen/basic_integer
core_unseen/decimal_arithmetic
core_unseen/parentheses
core_unseen/mixed_basic
feature_unseen/modulo
feature_unseen/power
feature_unseen/factorial
feature_unseen/constants_pi
feature_unseen/constants_e
feature_unseen/sqrt
feature_unseen/inv
feature_unseen/trig_deg
feature_unseen/trig_rad
feature_unseen/log
feature_unseen/ln
composition_unseen/decimal_plus_function
composition_unseen/constants_plus_power
composition_unseen/nested_functions
composition_unseen/mixed_scientific
composition_unseen/angle_mode_switching
composition_unseen/long_expression
ood_depth/deeper_than_training
ood_length/longer_plan_than_training
error_states/division_by_zero
error_states/invalid_sqrt
error_states/invalid_log
error_states/invalid_ln
error_states/tan_singularity_deg
error_states/invalid_factorial
```

Presets:

```python
SUITE_PRESETS = {
    "smoke": {...},
    "standard": {...},
    "full": {...},
}
```

Use reserved seeds by bucket, e.g.:

```python
SEED_RANGES = {
    "core_unseen": 1_000_000,
    "feature_unseen": 2_000_000,
    "composition_unseen": 3_000_000,
    "ood_depth": 4_000_000,
    "ood_length": 5_000_000,
    "error_states": 9_000_000,
}
```

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_suite.py tests/test_oracle_rich_generation.py tests/test_calculator_eval.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add eval/suite.py eval/common.py ../tests/test_eval_suite.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: generate diagnostic evaluation suites"
```

---

## Task B3: Add eval scoring module

**Objective:** Centralize entry correctness, result correctness, error correctness, divergence, and failure classification.

**Files:**
- Create: `ariadne/eval/scoring.py`
- Create: `tests/test_eval_scoring.py`
- Modify: `tests/test_eval_worker.py` later to use scoring helpers

**Step 1: Write failing tests**

Create tests for:

- entry success when history matches canonical task
- result success with numeric tolerance
- entry success but result failure
- result success but entry mismatch
- expected error success
- first divergence step
- failure classification

Example:

```python
from ariadne.eval.scoring import score_terminal_state


def test_scores_entry_and_result_success_separately():
    case = {"task": "sqrt(9)+2", "task_canonical": "sqrt(9)+2", "goal_type": "expression", "expected_value": 5.0}
    state = {"history": ["sqrt(9)+2"], "readout": "5", "error": None}
    score = score_terminal_state(case, state)
    assert score.entry_success is True
    assert score.result_success is True
    assert score.overall_success is True


def test_scores_expected_error_success():
    case = {"task": "sqrt(-1)", "task_canonical": "sqrt(-1)", "goal_type": "expected_error", "expected_error": "Calculation Error"}
    state = {"history": [], "readout": "Error", "error": "Calculation Error"}
    score = score_terminal_state(case, state)
    assert score.error_success is True
    assert score.overall_success is True
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_scoring.py -q
```

Expected: FAIL because module does not exist.

**Step 3: Implement**

`ariadne/eval/scoring.py` should provide:

```python
@dataclass(frozen=True)
class TerminalScore:
    entry_success: bool
    result_success: bool | None
    error_success: bool | None
    overall_success: bool
    failure_reason: str | None


def score_terminal_state(case: dict, state: dict, tolerance: float = 1e-9) -> TerminalScore: ...
def first_divergence_step(goal: str, steps: list[dict]) -> int | None: ...
def classify_failure(case: dict, state: dict, steps: list[dict], score: TerminalScore) -> str | None: ...
```

Failure reasons:

```text
success
timeout
early_enter
wrong_expression_entered
wrong_result
invalid_prefix
mode_error
angle_mode_error
unavailable_action
model_exception
bridge_error
calculator_error
expected_error_not_reached
```

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_scoring.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add eval/scoring.py ../tests/test_eval_scoring.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: score evaluation outcomes"
```

---

## Task B4: Refactor eval worker to use schema and scoring

**Objective:** Make `worker.py` mostly handle HTTP bridge lifecycle and model inference; delegate scoring to `scoring.py`.

**Files:**
- Replace/Modify: `ariadne/eval/worker.py`
- Modify: `tests/test_eval_worker.py`

**Step 1: Write failing tests**

Update `tests/test_eval_worker.py` to import scoring helpers from `ariadne.eval.scoring`, not worker-private functions.

Add tests that worker episode records include:

```text
entry_success
result_success
overall_success
failure_reason
expected_value
features
angle_mode
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_worker.py -q
```

Expected: FAIL until refactor.

**Step 3: Implement**

- Load `EvalCase` dicts from `suite.jsonl`.
- Predict actions as before, but include angle mode in serialized state via updated dataset serializer.
- Record step diagnostics.
- On terminal/timeout, call `score_terminal_state()`.
- Write one episode JSON object per case to `episodes.ndjson`.
- Remove old duplicate scoring helpers from `worker.py`.

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_worker.py tests/test_eval_scoring.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add eval/worker.py ../tests/test_eval_worker.py
git -C /home/aryan/projects/auxila/ariadne commit -m "refactor: modularize evaluation worker scoring"
```

---

## Task B5: Rewrite eval reporting with rich diagnostics

**Objective:** Produce detailed JSON and Markdown summaries from episode records.

**Files:**
- Replace/Modify: `ariadne/eval/reporting.py`
- Create: `tests/test_eval_reporting.py`

**Step 1: Write failing tests**

Create synthetic episodes and assert summary includes:

- overall metrics
- by bucket
- by stratum
- by feature
- by depth bin
- by plan-length bin
- by angle mode
- by goal type
- by failure reason
- top failure examples
- confusion pairs

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_reporting.py -q
```

Expected: FAIL until reporting is expanded.

**Step 3: Implement**

Output fields:

```json
{
  "overall": {
    "episodes": 0,
    "entry_success_rate": 0.0,
    "result_success_rate": 0.0,
    "overall_success_rate": 0.0,
    "error_success_rate": 0.0
  },
  "by_bucket": {},
  "by_stratum": {},
  "by_feature": {},
  "by_depth_bin": {},
  "by_plan_length_bin": {},
  "by_angle_mode": {},
  "by_goal_type": {},
  "by_failure_reason": {},
  "confusion_pairs": [],
  "top_failures": []
}
```

Write `summary.md` with sections:

```text
Overall
Worst Strata
By Feature
By Failure Reason
Entry/Result Mismatches
Top Failure Examples
```

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_eval_reporting.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add eval/reporting.py ../tests/test_eval_reporting.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: report rich evaluation diagnostics"
```

---

## Task B6: Update eval run CLI and orchestrator integration

**Objective:** Wire suite generation, manifest, worker outputs, and reporting into `python -m ariadne.eval.run` and orchestrator Phase 4.

**Files:**
- Modify: `ariadne/eval/run.py`
- Modify: `ariadne/orchestrate.py`
- Modify: `tests/test_orchestrate_evaluation.py`
- Modify: `tests/test_eval_smoke.py` if needed

**Step 1: Write failing tests**

Update orchestrator evaluation tests to assert output structure:

```text
suite.jsonl
manifest.json
episodes.ndjson
summary.json
summary.md
```

**Step 2: Verify RED**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_orchestrate_evaluation.py -q
```

Expected: FAIL until integration updated.

**Step 3: Implement**

`eval/run.py` should:

1. Resolve checkpoint/tokenizer/config context.
2. Build seen-task set.
3. Generate suite + manifest.
4. Run/evaluate workers.
5. Combine episode records.
6. Write reports.

Orchestrator Phase 4 should call the CLI with config-driven:

```yaml
evaluation:
  enabled: true
  checkpoint: latest
  suite: standard
  workers: 8
  base_port: 9000
  remote_clients: true
  headless: true
  decode_mode: greedy
  output_dir: evaluation
  max_steps_multiplier: 2.0
```

**Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests/test_orchestrate_evaluation.py tests/test_orchestrate_guards.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git -C /home/aryan/projects/auxila/ariadne add eval/run.py orchestrate.py ../tests/test_orchestrate_evaluation.py ../tests/test_eval_smoke.py
git -C /home/aryan/projects/auxila/ariadne commit -m "feat: integrate diagnostic evaluation pipeline"
```

---

# Milestone Verification

## After Milestone A

Run:

```bash
cd /home/aryan/projects/auxila
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest \
  tests/test_calculator_spec.py \
  tests/test_action_sim.py \
  tests/test_icalc_state_contract.py \
  tests/test_calculator_eval.py \
  tests/test_oracle_rich_generation.py \
  tests/test_tokenizer_rich_actions.py \
  tests/test_dagger_agent.py \
  tests/test_rl_ppo.py \
  -q
```

Expected: all pass.

Manually inspect sample generated tasks:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python - <<'PY'
from ariadne.agents.oracle import Oracle
oracle = Oracle(profile='rich_v1')
for category in ['decimal_arithmetic', 'constants_pi', 'power', 'factorial', 'trig_deg', 'trig_rad', 'mixed_scientific']:
    task = oracle.generate_task(seed=123, category=category)
    print(category, task)
PY
```

Expected: tasks cover the requested features, have canonical plans, and evaluate successfully.

## After Milestone B

Run:

```bash
cd /home/aryan/projects/auxila
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m pytest tests -q
```

Expected: full suite passes.

Generate a smoke suite without live browser/model if supported by CLI:

```bash
PYTHONPATH=/home/aryan/projects/auxila .venv/bin/python -m ariadne.eval.run \
  --suite smoke \
  --dry-run-suite \
  --output-dir /tmp/ariadne_eval_smoke
```

Expected files:

```text
/tmp/ariadne_eval_smoke/suite.jsonl
/tmp/ariadne_eval_smoke/manifest.json
```

When model/bridge are available, run live smoke eval and verify:

```text
suite.jsonl
manifest.json
episodes.ndjson
summary.json
summary.md
```

---

# Known Risks and Mitigations

## Risk: Python evaluator diverges from icalc JavaScript

Mitigation:
- Unit tests for known expressions.
- Optional browser parity smoke test later.
- Keep evaluator semantics intentionally close to `icalc/script.js`.

## Risk: Factorial/power/log generation creates unstable values

Mitigation:
- Use constraints: `max_factorial_arg`, `max_abs_value`, limited exponent sizes.
- Reject generated tasks whose Python evaluator returns error for safe categories.

## Risk: Angle mode is hidden/inconsistent

Mitigation:
- Expose `angleMode` in `icalcState`.
- Include it in `StateSerializer`.
- Include it in task metadata and eval reports.

## Risk: Too many eval strata dilute counts

Mitigation:
- Smoke/standard/full presets have explicit per-stratum counts.
- Reports aggregate by bucket and feature as well as stratum.

## Risk: Clean replacement breaks existing old runs/datasets

Mitigation:
- Accepted by product decision.
- Regenerate datasets with new canonical schema.
- Keep code simpler and avoid legacy compatibility bloat.

---

# Final Acceptance Criteria

Milestone A is complete when:

- `icalcState.availableInteractions` is canonical.
- `icalcState.angleMode` exists.
- `Oracle.generate_task(...)` returns `CalculatorTask`.
- Rich categories generate deterministic safe tasks with metadata.
- Action simulation reconstructs task expressions from plans.
- Data-generation agents log task metadata.
- Tokenizer/dataset understand canonical rich state.
- Targeted Milestone A tests pass.

Milestone B is complete when:

- Eval suite generation produces feature-first diagnostic suites.
- Suite metadata includes expected values/errors and reserved seeds.
- Eval scoring reports entry/result/error success separately.
- Eval worker writes rich episode records.
- Reporting writes `summary.json` and `summary.md` with diagnostic breakdowns.
- Orchestrator Phase 4 writes `suite.jsonl`, `manifest.json`, `episodes.ndjson`, `summary.json`, `summary.md`.
- Full tests pass.
