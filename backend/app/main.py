import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.schemas.translation import TranslationRequest, TranslationRespone
from app.services.translator import translator_service
from cachetools import TTLCache

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Đang tải mô hình Dịch Máy và Tokenizers vào RAM")
    try:
        translator_service.load_model()
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {str(e)}")
    yield
    print("Đang tắt dịch vụ API và giải phóng tài nguyên")

app=FastAPI(
    title="Hệ thống Dịch máy Anh - Việt",
    description="API phục vụ mô hình Transformer dịch máy Anh-Việt sử dụng ONNX Runtime, Beam Search",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache với chiến lược LRU (tối đa 2.000 mục, hết hạn sau 24h) chống tràn RAM
translation_cache = TTLCache(maxsize=2000, ttl=86400)

@app.get("/health",status_code=status.HTTP_200_OK,tags=["System"])
async def health_check():
    model_loaded=translator_service.is_loaded()
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": "CPU (ONNX Runtime)"
    }

@app.post(
    "/api/v2/translate",
    response_model=TranslationRespone,
    status_code=status.HTTP_200_OK,
    tags=["Translation"])
async def translate_text(payload: TranslationRequest):
    if not translator_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình dịch máy hiện tại chưa được tải thành công"
        )

    start_time=time.time()
    source_text=payload.text.strip()

    cache_key=source_text
    if cache_key in translation_cache:
        latency=time.time()-start_time
        return TranslationRespone(
            source_text=source_text,
            translated_text=translation_cache[cache_key],
            latency_seconds=round(latency,4),
            cached=True
        )
    
    try:
        translated_text=translator_service.translate(source_text)
        translation_cache[cache_key]=translated_text
        latency=time.time()-start_time
        return TranslationRespone(
            source_text=source_text,
            translated_text=translated_text,
            latency_seconds=round(latency,4),
            cached=False
        ) 
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Đã xảy ra lỗi trong quá trình dịch: {str(e)}"
        )