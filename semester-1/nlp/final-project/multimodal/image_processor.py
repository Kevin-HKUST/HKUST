import base64
import requests
from typing import Dict, Any


class ImageProcessor:
    """
    使用 Cloud Vision API（旧版）进行图像分析：
    - OCR 文本提取
    - 标签识别
    - 物体定位
    """

    def __init__(self):
        try:
            from config import Config
            self.api_key = Config.GOOGLE_VISION_KEY
        except Exception:
            raise RuntimeError("无法加载 GOOGLE_VISION_KEY，请检查 config.py")

        if not self.api_key or not self.api_key.startswith("AIza"):
            raise ValueError("GOOGLE_VISION_KEY 无效，请检查 API Key 是否正确")

        # Cloud Vision v1 endpoint
        self.endpoint = "https://vision.googleapis.com/v1/images:annotate"

    # ============================================================
    # 主接口：提取图片信息
    # ============================================================
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """
        向 Cloud Vision 发送请求，返回：
        {
            "text": "...",
            "labels": [...],
            "objects": [...],
        }
        """
        try:
            encoded_image = self._encode_image(image_path)
            payload = self._build_request(encoded_image)
            response = self._call_vision_api(payload)
            parsed = self._parse_vision_response(response)
            return parsed

        except Exception as e:
            return {"error": str(e)}

    # ============================================================
    # 把本地图片编码成 base64
    # ============================================================
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # ============================================================
    # 构造 Cloud Vision API 请求
    # ============================================================
    def _build_request(self, base64_img: str) -> Dict[str, Any]:
        return {
            "requests": [
                {
                    "image": {"content": base64_img},
                    "features": [
                        {"type": "TEXT_DETECTION"},
                        {"type": "LABEL_DETECTION"},
                        {"type": "OBJECT_LOCALIZATION"},
                    ],
                }
            ]
        }

    # ============================================================
    # 调用 Cloud Vision API
    # ============================================================
    def _call_vision_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        params = {"key": self.api_key}
        res = requests.post(self.endpoint, json=payload, params=params, timeout=10)
        res.raise_for_status()
        return res.json()

    # ============================================================
    # 解析 Cloud Vision 返回信息
    # ============================================================
    def _parse_vision_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        resp = data["responses"][0]

        # OCR
        text = ""
        if "fullTextAnnotation" in resp:
            text = resp["fullTextAnnotation"].get("text", "")

        # Labels
        labels = []
        if "labelAnnotations" in resp:
            labels = [x["description"] for x in resp["labelAnnotations"][:8]]

        # Object detection
        objects = []
        if "localizedObjectAnnotations" in resp:
            for obj in resp["localizedObjectAnnotations"]:
                objects.append(
                    {
                        "name": obj["name"],
                        "score": obj.get("score", 0),
                    }
                )

        return {
            "text": text,
            "labels": labels,
            "objects": objects,
        }