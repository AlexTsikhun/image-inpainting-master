import os
import gdown
from config import config

files = {
    "places2.ckpt-6666.data-00001-of-00002": config.FILE_ID_DATA,
    "places2.ckpt-6666.index": config.FILE_ID_INDEX,
}

model_dir = "inpaint/models"
os.makedirs(model_dir, exist_ok=True)

def download_file(file_name, file_id):
    output_path = os.path.join(model_dir, file_name)
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Завантаження {file_name}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"File {file_name} already exists.")

def download_all_weights():
    for name, fid in files.items():
        download_file(name, fid)
