from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TranslationRequest(BaseModel):
    text: str= Field(...,min_length=1,max_length=300,
    description="Đoạn văn bản Tiếng Anh cần dịch sang Tiếng Việt.",
    examples=["Hello, welcome to my machine translation"])

class TranslationRespone(BaseModel):
    source_text: str=Field(
        ...,
        description="Văn bản gốc Tiếng Anh nhận được từ request",
    )

    translated_text: str=Field(
        ...,
        description="Văn bản Tiếng Việt sau khi được mô hình dịch máy dịch thành công"
    )

    latency_seconds: float=Field(
        ...,
        description="Thời gian xử lý dịch thuật (tính bằng giây)"
    )

    cached: bool=Field(
        default=False,
        description="Trạng thái phản hồi có được lấy từ Cache hay phải chạy Inference từ mô hình"
    )