import os
from dotenv import load_dotenv

# Tự động tải các biến từ file .env
load_dotenv()

class Settings:
    HF_REPO_ID: str = os.getenv("HF_REPO_ID")
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    
    MODEL_DIR: str = os.getenv("MODEL_DIR", os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"))
    
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "64"))
    BEAM_WIDTH: int = int(os.getenv("BEAM_WIDTH", "4"))
    LEN_PENALTY_ALPHA: float = float(os.getenv("LEN_PENALTY_ALPHA", "0.6"))

settings = Settings()
