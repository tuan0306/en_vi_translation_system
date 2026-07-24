import sys
import os
from fastapi.testclient import TestClient  

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.main import app

client=TestClient(app)

def test_health_check():
    response=client.get("/health")
    assert response.status_code==200,  "Lỗi: API Health Check không trả về 200"

    data=response.json()
    assert "status" in data, "Lỗi: Không tìm thấy trường 'status' trong phản hồi"

    print("Backend Health Check Passed")