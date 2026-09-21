import os
from typing import Dict, Any

class Config:
    """项目配置类"""

    # API配置（密钥从环境变量读取，勿写死在代码里）
    HKGAI_API_KEY = os.getenv("HKGAI_API_KEY", "")
    HKGAI_BASE_URL = os.getenv("HKGAI_BASE_URL", "https://oneapi.hkgai.net/v1")
    HKGAI_MODEL_ID = os.getenv("HKGAI_MODEL_ID", "HKGAI-V1")

    # 搜索引擎API
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

    # 领域插件API密钥
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_KEY", "")
    # 真实 API 开关（优化：启用真实调用，提升准确度）
    USE_REAL_API = True  # True: 真实调用；False: mock 演示

    # 向量数据库配置
    VECTOR_DB_PATH = "data/vector_db"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # 缓存配置
    USE_REDIS = False
    CACHE_TTL = 3600

    # 性能配置
    MAX_WORKFLOW_STEPS = 10
    TIMEOUT = 30

    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/project_ise.log"

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """获取LLM配置"""
        return {
            "api_key": cls.HKGAI_API_KEY,
            "base_url": cls.HKGAI_BASE_URL,
            "model": cls.HKGAI_MODEL_ID,
            "temperature": 0.1,
            "max_tokens": 1000
        }

# 创建必要的目录
os.makedirs("data/vector_db", exist_ok=True)
os.makedirs("data/knowledge_base", exist_ok=True)
os.makedirs("data/test_questions", exist_ok=True)
os.makedirs("data/multimodal_files", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 禁用 SSL 验证（全局生效，解决证书报错）
import ssl
ssl._create_default_https_context = ssl._create_unverified_context