import os
from typing import Any, Dict

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is a convenience, not a requirement
    def load_dotenv(*_args, **_kwargs):
        return False

# Load a local .env if present. Exporting the variables directly works too.
# See .env.example for the accepted names.
load_dotenv()


class Config:
    """Runtime configuration. Secrets come from the environment, never from source."""

    # --- LLM ---
    HKGAI_API_KEY = os.getenv("HKGAI_API_KEY", "")
    HKGAI_BASE_URL = os.getenv("HKGAI_BASE_URL", "https://oneapi.hkgai.net/v1")
    HKGAI_MODEL_ID = os.getenv("HKGAI_MODEL_ID", "HKGAI-V1")

    # --- Plugin API keys. A blank value disables the plugin that needs it. ---
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_KEY", "")
    BAIDU_MAP_API_KEY = os.getenv("BAIDU_MAP_API_KEY", "")

    # --- Retrieval ---
    VECTOR_DB_PATH = "data/vector_db"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # --- Caching ---
    USE_REDIS = False
    CACHE_TTL = 3600

    # --- Limits ---
    MAX_WORKFLOW_STEPS = 10
    TIMEOUT = 30

    # --- Logging ---
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/project_ise.log"

    # Directories the runtime writes into.
    RUNTIME_DIRS = (
        "data/vector_db",
        "data/knowledge_base",
        "data/test_questions",
        "data/multimodal_files",
        "logs",
    )

    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        return {
            "api_key": cls.HKGAI_API_KEY,
            "base_url": cls.HKGAI_BASE_URL,
            "model": cls.HKGAI_MODEL_ID,
            "temperature": 0.1,
            "max_tokens": 1000,
        }

    @classmethod
    def has_llm_credentials(cls) -> bool:
        return bool(cls.HKGAI_API_KEY)


def ensure_runtime_dirs() -> None:
    """Create the directories the runtime writes into.

    Call this explicitly from an entry point. Importing a module must not
    create directories as a side effect.
    """
    for path in Config.RUNTIME_DIRS:
        os.makedirs(path, exist_ok=True)
