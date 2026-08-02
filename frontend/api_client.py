from requests import request
import os
import yaml
import requests

def load_yaml_config():
    try:
        with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

config = load_yaml_config()
BACKEND_API_URL = os.getenv("BACKEND_API_URL", config.get("backend_api_url", "http://127.0.0.1:8000"))

class TranslationAPIClient:
    def __init__(self):
        self.base_url = BACKEND_API_URL.rstrip("/")
    
    def check_health(self):
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                try:
                    return response.json()
                except Exception:
                    return {"status": "unhealthy", "model_loaded": False}
            return {"status": "unhealthy", "model_loaded": False}
        except requests.exceptions.RequestException:
            return {"status": "offline", "model_loaded": False}

    def translate(self, text: str):
        url = f"{self.base_url}/api/v2/translate"
        payload = {"text": text}
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                try:
                    data = response.json()
                    return {
                        "success": True,
                        "translated_text": data.get("translated_text", ""),
                        "latency_seconds": data.get("latency_seconds", 0.0),
                        "cached": data.get("cached", False),
                        "error": None
                    }
                except Exception:
                    return {
                        "success": False,
                        "error": "Phản hồi kết quả dịch từ máy chủ không đúng định dạng JSON"
                    }
            elif response.status_code == 503:
                return {
                    "success": False,
                    "error": "Mô hình dịch đang được khởi động. Vui lòng thử lại sau"
                }
            else:
                try:
                    error_msg = response.json().get("detail", f"Lỗi máy chủ (HTTP {response.status_code})")
                except Exception:
                    error_msg = f"Máy chủ phản hồi lỗi (HTTP {response.status_code})"
                return {
                    "success": False,
                    "error": error_msg
                }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Yêu cầu dịch bị quá thời gian chờ. Vui lòng thử lại câu ngắn hơn"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Không thể kết nối tới máy chủ API"
            }
        except Exception as e:
            return {
                "success": False, 
                "error": f"Lỗi không xác định: {str(e)}"
            }

api_client = TranslationAPIClient()