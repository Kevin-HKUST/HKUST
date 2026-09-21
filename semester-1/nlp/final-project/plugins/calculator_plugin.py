from plugins.base_plugin import BasePlugin
import logging

logger = logging.getLogger(__name__)

class CalculatorPlugin(BasePlugin):
    def __init__(self):
        super().__init__("calculator_plugin")

    def execute(self, query: str, context=None):
        try:
            # 简单计算器：支持加减乘除
            expr = query.replace("等于", "").replace("是多少", "")
            expr = expr.replace("?", "").strip()

            # 只保留数字和运算符
            safe_expr = "".join(ch for ch in expr if ch in "0123456789+-*/(). ")

            result = eval(safe_expr)

            return self.format_output(
                data={"expression": safe_expr, "result": result},
                confidence=0.95,
                metadata={"success": True}
            )

        except Exception as e:
            logger.error(f"Calculator error: {e}")
            return self.format_output(
                data=f"[Calculator Error] {str(e)}",
                confidence=0,
                metadata={"success": False}
            )