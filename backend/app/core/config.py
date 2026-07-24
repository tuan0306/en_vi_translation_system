import os
import yaml
from dotenv import load_dotenv

# Tự động tải các biến từ file .env
load_dotenv()

def load_yaml_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

config = load_yaml_config()

class Settings:
    HF_REPO_ID: str = config.get("hf_repo_id", "nddttt/en-vi-translation-model")
    HF_TOKEN: str = os.getenv("HF_TOKEN", None)
    
    MODEL_DIR: str = os.getenv("MODEL_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"))
    
    MAX_LENGTH: int = int(config.get("max_length", 128))
    BEAM_WIDTH: int = int(config.get("beam_width", 4))
    LEN_PENALTY_ALPHA: float = float(config.get("len_penalty_alpha", 0.6))

settings = Settings()
