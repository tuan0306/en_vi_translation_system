FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /code

COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Thiết lập User không phải root (UID 1000) theo chuẩn bảo mật của Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH 

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Thiết lập cấu hình Streamlit chạy Headless trên cổng 7860
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 7860

CMD ["streamlit","run","app.py"]