"""
agents/oracle.py — Expression generator and key-sequence planner.

The Oracle is purely functional: no I/O, no state.
"""

import random


FUNCTIONS = ["sin", "cos", "tan", "log", "ln", "sqrt"]
OPERATIONS = ["+", "-", "*", "/"]

# Keys that icalc auto-appends '(' for
FUNCTION_KEYS = set(FUNCTIONS)

# Symbol display conversions applied by icalc
KEY_DISPLAY = {
    "pi": "π",
}


class Oracle:
    def __init__(self) -> None:
        self.functions  = FUNCTIONS
        self.operations = OPERATIONS

    # ------------------------------------------------------------------
    # Expression generation
    # ------------------------------------------------------------------

    def _number(self) -> str:
        return str(random.randint(1, 100))

    def _expression(self, depth: int = 2, scientific: bool = False) -> str:
        if depth == 0:
            return self._number()

        choices = ["number", "binary"]
        if scientific:
            choices.append("function")
        if random.random() < 0.2 and depth > 1:
            choices.append("parens")

        choice = random.choice(choices)

        if choice == "number":
            return self._number()
        elif choice == "binary":
            op = random.choice(self.operations)
            return f"{self._expression(depth-1, scientific)} {op} {self._expression(depth-1, scientific)}"
        elif choice == "function":
            func = random.choice(self.functions)
            return f"{func}({self._expression(depth-1, scientific)})"
        elif choice == "parens":
            return f"({self._expression(depth-1, scientific)})"

        return self._number()

    # ------------------------------------------------------------------
    # Key-sequence planning
    # ------------------------------------------------------------------

    def plan(self, expression: str, current_mode: str = "basic") -> list[str]:
        """Convert a math expression string to a list of keystroke actions."""
        steps: list[str] = []

        is_scientific = any(f in expression for f in self.functions)
        if is_scientific and current_mode != "scientific":
            steps.append("m")  # switch to scientific mode

        i = 0
        while i < len(expression):
            ch = expression[i]
            if ch == " ":
                i += 1
                continue
            # Greedy match on function names
            matched = None
            for func in self.functions:
                if expression.startswith(func, i):
                    matched = func
                    break
            if matched:
                steps.append(matched)
                i += len(matched)
                # icalc adds '(' automatically — skip it in the expression string
                if i < len(expression) and expression[i] == "(":
                    i += 1
            else:
                steps.append(ch)
                i += 1

        steps.append("Enter")
        return steps

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    def generate_task(
        self,
        current_mode: str = "basic",
        basic_only: bool = False,
        min_depth: int = 1,
        max_depth: int = 3,
    ) -> tuple[str, list[str]]:
        """Return (expression, keystroke_plan) for a random task."""
        scientific = False if basic_only else (random.random() < 0.3)
        depth      = random.randint(min_depth, max_depth)
        expr       = self._expression(depth, scientific)
        return expr, self.plan(expr, current_mode)

    def generate_task_for_index(
        self,
        index: int,
        current_mode: str = "basic",
        basic_only: bool = False,
        min_depth: int = 1,
        max_depth: int = 3,
    ) -> tuple[str, list[str]]:
        """Deterministic task generation (seeded by index)."""
        random.seed(index)
        scientific = False if basic_only else (index % 2 != 0)
        depth      = random.randint(min_depth, max_depth)
        expr       = self._expression(depth, scientific)
        return expr, self.plan(expr, current_mode)
