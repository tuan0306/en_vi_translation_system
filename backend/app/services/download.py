import os
from huggingface_hub import hf_hub_download
from app.core.config import settings

def download_model_files():
    os.makedirs(settings.MODEL_DIR,exist_ok=True)

    files_to_download={
        "model":"transformer_en_vi.onnx",
        "sp_en":"spm_en.model",
        "sp_vi":"spm_vi.model"
    }

    paths={}

    for key,filename in files_to_download.items():
        local_path=os.path.join(settings.MODEL_DIR,filename)

        if not os.path.exists(local_path):
            print(f"Đang tải {filename} từ Hugging Face Hub ({settings.HF_REPO_ID})...")
            downloaded_path=hf_hub_download(
                repo_id=settings.HF_REPO_ID,
                filename=filename,
                token=settings.HF_TOKEN,
                local_dir=settings.MODEL_DIR,
                local_dir_use_symlinks=False
            )
            print(f"Đã tải thành công {filename} về {downloaded_path}")
        else:
            print(f"File {filename} đã tồn tại cục bộ tại: {local_path}")

        paths[key]=local_path

    return paths