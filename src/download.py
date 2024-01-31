import os

import requests
import gzip
import shutil
from tqdm import tqdm

SERVER_ADDRESS = "https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/"


def download_gz_file(local_folder: str, target_file: str) -> None:
    target_file_gz = target_file + ".gz"
    local_path_gz = local_folder + "/" + target_file_gz
    # download file
    download_file(local_folder, target_file_gz)
    # unpack archive
    print(f"Unpacking {local_path_gz}...")
    with gzip.open(local_path_gz, 'rb') as f_in:
        with open(local_path_gz[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    # remove archive
    os.remove(local_path_gz)


def download_file(local_folder: str, target_file: str) -> None:
    os.makedirs(local_folder, exist_ok=True)
    local_path = local_folder + "/" + target_file
    # Download archive file from Google Cloud Storage
    url = SERVER_ADDRESS + target_file
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            progress_bar.update(len(chunk))
            f.write(chunk)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
