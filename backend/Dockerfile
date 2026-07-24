FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /code

# Cài đặt các gói hệ thống cần thiết cho SentencePiece & ONNX
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Thiết lập User không phải root (UID 1000) theo bảo mật của Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Sao chép mã nguồn Backend dưới quyền sở hữu của user 1000
COPY --chown=user . $HOME/app

# Tạo sẵn thư mục chứa mô hình để sẵn sàng tải model về
RUN mkdir -p $HOME/app/models

EXPOSE 7860

# Khởi chạy Uvicorn lắng nghe trên cổng 7860
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","7860"]