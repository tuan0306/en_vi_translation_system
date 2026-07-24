import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "frontend"))

from api_client import api_client,BACKEND_API_URL

def test_api_client_config():
    assert BACKEND_API_URL is not None, "Lỗi: Không tìm thấy BACKEND_API_URL"
    assert api_client.base_url==BACKEND_API_URL.rstrip("/"), "Lỗi: api_client không nhận đúng URL"
    assert isinstance(api_client.base_url,str),  "Lỗi: URL phải là kiểu chuỗi văn bản"
    print("Frontend Config Check Passed!")
