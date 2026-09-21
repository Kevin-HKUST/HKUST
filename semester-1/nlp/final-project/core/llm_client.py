import requests
import json
from typing import Dict, Any
from config import Config
import logging

logger = logging.getLogger(__name__)

class LLMClient:
    """HKGAI LLM 客户端（增强版，兼容多种输出格式 + 强健错误处理）"""

    def __init__(self):
        self.api_key = Config.HKGAI_API_KEY
        self.base_url = Config.HKGAI_BASE_URL
        self.model_id = Config.HKGAI_MODEL_ID
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self,
             system_prompt: str,
             user_prompt: str,
             max_tokens: int = 1500,
             temperature: float = 0.2) -> Dict[str, Any]:

        endpoint = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            logger.info(f"[HKGAI] Sending request... (system={system_prompt[:40]}...)")

            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=(30, 45)
            )
            response.raise_for_status()

            data = response.json()
            logger.info("[HKGAI] Raw response received")

            # -----------------------------
            # ⭐ 兼容两种可能的返回格式
            # -----------------------------
            content = ""

            try:
                # Chat schema
                content = (
                    data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                )
            except Exception:
                content = ""

            # text 架构 fallback
            if not content:
                content = (
                    data.get("choices", [{}])[0]
                        .get("text", "")
                )

            # 再次清理
            if content:
                content = content.strip()

            # -----------------------------
            # 返回空内容时，保留原始数据用于 debug
            # -----------------------------
            if not content:
                return {
                    "content": "",
                    "warning": "Empty response from HKGAI model",
                    "raw": data
                }

            return {
                "content": content,
                "raw": data
            }

        except requests.exceptions.HTTPError as e:
            logger.error(f"[HKGAI] HTTP error: {e}")
            return {
                "content": "抱歉，模型接口暂时不可用。",
                "error": str(e)
            }

        except Exception as e:
            logger.error(f"[HKGAI] Request failed: {e}")
            return {
                "content": "抱歉，AI 服务出现错误。",
                "error": str(e)
            }


# 全局实例
llm_client = LLMClient()