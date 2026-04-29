"""
agents/oracle.py — Typed calculator task generator and key-sequence planner.

The Oracle is deterministic by seed/index and returns CalculatorTask objects
with expression, canonical action plan, expected result metadata, and features.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from ariadne.core.calculator_eval import EvaluationResult, evaluate_expression, normalize_expression
from ariadne.core.calculator_spec import SCIENTIFIC_FUNCTIONS, canonicalize_key

FUNCTIONS = ["sin", "cos", "tan", "log", "ln", "sqrt", "inv"]
OPERATIONS = ["+", "-", "*", "/"]
RICH_OPERATIONS = ["+", "-", "*", "/", "%", "^"]
CONSTANTS = ["pi", "e"]
SCIENTIFIC_PLAN_KEYS = SCIENTIFIC_FUNCTIONS | frozenset(CONSTANTS) | {"^", "!"}


@dataclass(frozen=True)
class CalculatorTask:
    expression: str
    plan: list[str]
    expected: EvaluationResult
    angle_mode: str = "deg"
    category: str = "mixed"
    seed: int | None = None
    metadata: dict | None = None

    @property
    def task_canonical(self) -> str:
        return normalize_expression(self.expression)

    @property
    def features(self) -> frozenset[str]:
        values = (self.metadata or {}).get("features", [])
        return frozenset(str(value) for value in values)

    @property
    def depth(self) -> int:
        return int((self.metadata or {}).get("depth", 0))

    @property
    def token_length(self) -> int:
        return len(self.plan)

    @property
    def scientific(self) -> bool:
        return "scientific_mode" in self.features or any(key in self.plan for key in SCIENTIFIC_PLAN_KEYS)

    @property
    def goal_type(self) -> str:
        return "expected_error" if self.expected.error else "expression"

    @property
    def expected_value(self) -> float | None:
        return self.expected.value

    @property
    def expected_error(self) -> str | None:
        return self.expected.error

    def to_dict(self) -> dict:
        data = asdict(self)
        data["task_canonical"] = self.task_canonical
        data["features"] = sorted(self.features)
        data["depth"] = self.depth
        data["token_length"] = self.token_length
        data["scientific"] = self.scientific
        data["goal_type"] = self.goal_type
        data["expected_value"] = self.expected.value
        data["expected_error"] = self.expected.error
        return data


class Oracle:
    def __init__(self, profile: str = "rich_v1") -> None:
        self.profile = profile
        self.functions = FUNCTIONS
        self.operations = OPERATIONS

    # ------------------------------------------------------------------
    # Expression generation
    # ------------------------------------------------------------------

    def _number(self, rng: random.Random, allow_decimal: bool = True) -> str:
        if allow_decimal and rng.random() < 0.3:
            return f"{rng.randint(1, 99)}.{rng.randint(1, 9)}"
        return str(rng.randint(1, 100))

    def _atom(self, rng: random.Random, rich: bool) -> str:
        choices = ["number"] * 6
        if rich:
            choices += ["constant", "factorial"]
        choice = rng.choice(choices)
        if choice == "constant":
            return rng.choice(CONSTANTS)
        if choice == "factorial":
            return f"{rng.randint(0, 7)}!"
        return self._number(rng, allow_decimal=rich)

    def _expression(self, rng: random.Random, depth: int = 2, rich: bool = True, category: str = "mixed") -> str:
        if depth <= 0:
            return self._atom(rng, rich)

        if category in {"basic_integer", "mixed_basic"}:
            return f"{rng.randint(1, 100)} {rng.choice(OPERATIONS)} {rng.randint(1, 100)}"
        if category == "parentheses":
            return f"({rng.randint(1, 20)} {rng.choice(['+', '-'])} {rng.randint(1, 20)}) * {rng.randint(2, 9)}"
        if category == "modulo":
            return f"{rng.randint(10, 100)} % {rng.randint(2, 12)}"
        if category == "power":
            return f"{rng.randint(2, 9)} ^ {rng.randint(2, 4)}"
        if category == "factorial":
            return f"{rng.randint(0, 7)}!"
        if category in {"constants", "constants_pi"}:
            return f"pi {rng.choice(['+', '*'])} {self._number(rng)}"
        if category == "constants_e":
            return f"e {rng.choice(['+', '*'])} {self._number(rng)}"
        if category in {"sqrt", "inv", "log", "ln"}:
            return f"{category}({rng.randint(1, 100)})"
        if category == "unary_function":
            func = rng.choice(FUNCTIONS)
            inner = str(rng.randint(1, 100)) if func in {"log", "ln", "sqrt", "inv"} else rng.choice(["30", "45", "60"])
            return f"{func}({inner})"
        if category in {"trig_deg", "trig_rad", "angle_trig"}:
            func = rng.choice(["sin", "cos", "tan"])
            if category == "trig_rad":
                arg = rng.choice(["pi/6", "pi/4", "pi/3", "pi/2"])
            else:
                arg = str(rng.randint(1, 89) if func == "tan" else rng.randint(0, 360))
            return f"{func}({arg})"
        if category == "decimal_arithmetic":
            return f"{self._number(rng, True)} {rng.choice(OPERATIONS)} {self._number(rng, True)}"
        if category in {"mixed_scientific", "decimal_plus_function"}:
            return f"{rng.randint(1, 20)} + sqrt({rng.randint(1, 100)})"
        if category == "constants_plus_power":
            return f"pi + {rng.randint(2, 6)} ^ 2"
        if category == "nested_functions":
            return f"sqrt(inv({rng.randint(1, 20)}))"

        choices = ["atom", "binary", "parens"]
        if rich:
            choices += ["function", "power"]
        choice = rng.choice(choices)

        if choice == "atom":
            return self._atom(rng, rich)
        if choice == "function":
            func = rng.choice(FUNCTIONS)
            # Keep generated training data valid by constraining domains.
            if func in {"log", "ln", "sqrt", "inv"}:
                inner = str(rng.randint(1, 100))
            else:
                inner = self._expression(rng, depth - 1, rich, category)
            return f"{func}({inner})"
        if choice == "parens":
            return f"({self._expression(rng, depth - 1, rich, category)})"
        if choice == "power":
            return f"{rng.randint(2, 9)} ^ {rng.randint(2, 4)}"

        op = rng.choice(RICH_OPERATIONS if rich else OPERATIONS)
        left = self._expression(rng, depth - 1, rich, category)
        right = self._expression(rng, depth - 1, rich, category)
        if op in {"/", "%"} and right.strip() in {"0", "0.0"}:
            right = "1"
        return f"{left} {op} {right}"

    # ------------------------------------------------------------------
    # Key-sequence planning
    # ------------------------------------------------------------------

    def plan(self, expression: str, current_mode: str = "basic", angle_mode: str = "deg") -> list[str]:
        """Convert a math expression string to canonical keystroke actions."""
        steps: list[str] = []
        expr = normalize_expression(expression)
        is_scientific = any(f"{func}(" in expr for func in SCIENTIFIC_FUNCTIONS) or any(c in expr for c in CONSTANTS) or any(k in expr for k in ["^", "!"])
        if is_scientific and current_mode != "scientific":
            steps.append("m")
        if is_scientific and angle_mode == "rad":
            steps.append("deg")

        i = 0
        tokens = sorted(list(SCIENTIFIC_FUNCTIONS) + CONSTANTS, key=len, reverse=True)
        while i < len(expr):
            ch = expr[i]
            if ch == " ":
                i += 1
                continue
            matched = None
            for tok in tokens:
                if expr.startswith(tok, i):
                    matched = tok
                    break
            if matched:
                steps.append(canonicalize_key(matched))
                i += len(matched)
                if matched in SCIENTIFIC_FUNCTIONS and i < len(expr) and expr[i] == "(":
                    i += 1
            else:
                steps.append(canonicalize_key(ch))
                i += 1
        steps.append("Enter")
        return steps

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    def _task_from_expr(
        self,
        expression: str,
        *,
        seed: int | None,
        category: str,
        current_mode: str,
        angle_mode: str,
        depth: int,
    ) -> CalculatorTask:
        plan = self.plan(expression, current_mode=current_mode, angle_mode=angle_mode)
        expected = evaluate_expression(expression, angle_mode=angle_mode)
        features = self._features(expression, plan)
        return CalculatorTask(
            expression=expression,
            plan=plan,
            expected=expected,
            angle_mode=angle_mode,
            category=category,
            seed=seed,
            metadata={"features": features, "depth": depth, "plan_length": len(plan), "profile": self.profile},
        )

    def generate_task(
        self,
        *,
        seed: int | None = None,
        category: str = "mixed",
        current_mode: str = "basic",
        basic_only: bool = False,
        min_depth: int = 1,
        max_depth: int = 3,
        profile: str | None = None,
    ) -> CalculatorTask:
        rng = random.Random(seed)
        profile = profile or self.profile
        rich = (profile != "basic_v1") and not basic_only
        if category == "trig_rad":
            angle_mode = "rad"
        elif category == "trig_deg":
            angle_mode = "deg"
        else:
            angle_mode = "rad" if rich and rng.random() < 0.25 else "deg"
        for _ in range(200):
            depth = rng.randint(min_depth, max_depth)
            expr = self._expression(rng, depth, rich=rich, category=category)
            task = self._task_from_expr(
                expr,
                seed=seed,
                category=category,
                current_mode=current_mode,
                angle_mode=angle_mode,
                depth=depth,
            )
            if task.expected.ok:
                return task
        raise RuntimeError(f"Could not generate valid calculator task for seed={seed} category={category}")

    def generate_task_for_index(
        self,
        index: int,
        current_mode: str = "basic",
        basic_only: bool = False,
        min_depth: int = 1,
        max_depth: int = 3,
        profile: str | None = None,
        category: str = "mixed",
    ) -> CalculatorTask:
        return self.generate_task(
            seed=index,
            category=category,
            current_mode=current_mode,
            basic_only=basic_only,
            min_depth=min_depth,
            max_depth=max_depth,
            profile=profile,
        )

    def _features(self, expression: str, plan: list[str]) -> list[str]:
        expr = normalize_expression(expression)
        features: set[str] = set()
        for op, name in [("+", "add"), ("-", "subtract"), ("*", "multiply"), ("/", "divide"), ("%", "percent"), ("^", "power")]:
            if op in expr:
                features.add(name)
        for func in FUNCTIONS:
            if f"{func}(" in expr:
                features.add(func)
        if any(f"{func}(" in expr for func in ["sin", "cos", "tan"]):
            features.add("trig")
        for const in CONSTANTS:
            if const in expr:
                features.add(const)
        if "." in expr:
            features.add("decimal")
        if "!" in expr:
            features.add("factorial")
        if "(" in expr:
            features.add("grouping")
        if "m" in plan:
            features.add("scientific_mode")
        return sorted(features) or ["integer"]
