from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BasePlugin(ABC):
    """插件基类"""

    def __init__(self, name: str = "base_plugin"):
        self.name = name
        self.version = "1.0.0"
        self.capabilities = []
        self.required_params = []

    # ======================================================
    # ⭐ 新增：所有插件统一 run()（保持原行为 + 失败时返回统一结构）
    # ======================================================
    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        workflow_engine 的统一入口。
        默认行为：
        - 正常：返回 execute() 的结果
        - 异常：通过 _handle_error 返回标准失败结构
        """
        try:
            result = self.execute(query, context)

            # ⭐ 保护：如果插件返回 None，则自动视为失败
            if result is None:
                return self.failure("Plugin returned None")

            # ⭐ 保护：如果 result 不是 dict，强制包装
            if not isinstance(result, dict):
                return self.failure("Plugin returned non-dict result")

            # ⭐ 插件正常返回
            return result

        except Exception as e:
            return self.failure(f"Exception: {str(e)}")

    # ======================================================
    # 原有执行逻辑（保持不动）
    # ======================================================
    @abstractmethod
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行插件主要逻辑"""
        pass

    # ======================================================
    # ⭐ 新增：统一的成功/失败结构（保证 workflow 可识别）
    # ======================================================
    def success(self, data: Any, confidence: float = 1.0,
                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """插件成功返回"""
        if metadata is None:
            metadata = {}
        metadata["success"] = True
        return self.format_output(data, confidence, metadata)

    def failure(self, msg: str = "unknown error") -> Dict[str, Any]:
        """插件失败返回（workflow_engine 用于 fallback 判断）"""
        return self.format_output(
            data={"error": msg},
            confidence=0.0,
            metadata={"success": False}
        )

    # ======================================================
    # 原有工具函数（保持不动）
    # ======================================================
    def validate_input(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """验证输入参数"""
        validation_result = {
            "valid": True,
            "missing_params": [],
            "errors": []
        }

        if params:
            for param in self.required_params:
                if param not in params or params[param] is None:
                    validation_result["valid"] = False
                    validation_result["missing_params"].append(param)

        if not query or not query.strip():
            validation_result["valid"] = False
            validation_result["errors"].append("Query cannot be empty")

        return validation_result

    def format_output(self, data: Any, confidence: float = 1.0,
                      metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """格式化输出"""
        if metadata is None:
            metadata = {}
        metadata["source"] = "plugin"

        return {
            "plugin": self.name,
            "content": data,
            "confidence": max(0.0, min(1.0, confidence)),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }

    # 信息类（保持不动）
    def get_capabilities(self) -> List[str]:
        return self.capabilities

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "required_params": self.required_params
        }

    # ======================================================
    # 原有错误处理保持一致，但底层使用 failure() 保证结构稳定
    # ======================================================
    def _handle_error(self, error_msg: str, error_type: str = "execution_error") -> Dict[str, Any]:
        logger.error(f"Plugin {self.name} error: {error_msg}")
        return self.failure(error_msg)