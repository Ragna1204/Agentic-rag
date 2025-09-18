"""
A sandboxed calculator tool.

This module provides a simple tool for performing mathematical calculations.
It uses `eval` in a restricted environment to safely evaluate mathematical
expressions.
"""
import math
from ..utils.logging import get_logger

logger = get_logger(__name__)

class Calculator:
    """
    A tool for evaluating mathematical expressions.
    """
    def __init__(self):
        """
        Initializes the Calculator with a safe execution environment.
        """
        # TODO: Enhance the safety of the sandbox. `eval` is risky.
        # Consider using a dedicated math parsing library like `numexpr` or `asteval`.
        self.safe_dict = {
            "acos": math.acos, "asin": math.asin, "atan": math.atan, "atan2": math.atan2,
            "ceil": math.ceil, "cos": math.cos, "cosh": math.cosh, "degrees": math.degrees,
            "e": math.e, "exp": math.exp, "fabs": math.fabs, "floor": math.floor,
            "fmod": math.fmod, "frexp": math.frexp, "hypot": math.hypot, "ldexp": math.ldexp,
            "log": math.log, "log10": math.log10, "modf": math.modf, "pi": math.pi,
            "pow": pow, "radians": math.radians, "sin": math.sin, "sinh": math.sinh,
            "sqrt": math.sqrt, "tan": math.tan, "tanh": math.tanh,
            "__builtins__": None
        }
        logger.info("Calculator tool initialized.")

    def execute(self, expression: str) -> str:
        """
        Safely evaluates a mathematical expression.

        Args:
            expression (str): The mathematical expression to compute (e.g., "2 * (3 + 4)").

        Returns:
            str: The result of the calculation as a string, or an error message.
        """
        logger.debug(f"Executing calculator with expression: '{expression}'")
        try:
            # Using eval is dangerous, but we restrict the environment with safe_dict.
            result = eval(expression, {"__builtins__": {}}, self.safe_dict)
            logger.info(f"Calculator result: {expression} = {result}")
            return str(result)
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            return f"Error: Could not evaluate the expression. Invalid syntax or operation."

# Example usage:
# if __name__ == '__main__':
#     calc = Calculator()
#     print(calc.execute("2 + 3 * sqrt(16)"))
#     print(calc.execute("pow(2, 10)"))
#     print(calc.execute("import os")) # This should fail
