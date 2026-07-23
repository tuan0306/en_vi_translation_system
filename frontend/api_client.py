from requests import request
import os
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

class TranslationAPIClient:
    def __init__(self):
        self.base_url = BACKEND_API_URL.rstrip("/")
    
    def check_health(self):
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"status": "unhealthy", "model_loaded": False}
        except requests.exceptions.RequestException:
            return {"status": "offline", "model_loaded": False}

    def translate(self, text: str):
        url = f"{self.base_url}/api/v2/translate"
        payload = {"text": text}
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "translated_text": data.get("translated_text", ""),
                    "latency_seconds": data.get("latency_seconds", 0.0),
                    "cached": data.get("cached", False),
                    "error": None
                }
            elif response.status_code == 503:
                return {
                    "success": False,
                    "error": "Mô hình dịch đang được khởi động. Vui lòng thử lại sau"
                }
            else:
                error_msg = response.json().get("detail", "Lỗi hệ thống Backend")
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
                "error": "Không thể kết nối tới máy chủ"
            }
        except Exception as e:
            return {
                "success": False, 
                "error": f"Lỗi không xác định: {str(e)}"
            }

api_client = TranslationAPIClient()