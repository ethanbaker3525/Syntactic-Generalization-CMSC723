"""
use Tinker API to download model weights from their storage
"""

import argparse
from pathlib import Path
import tinker
import urllib.request
import os, tarfile


def download(model_path):
  if model_path is None:
    raise ValueError("Please provide model path to download weights from Tinker.")

  sc = tinker.ServiceClient()
  rc = sc.create_rest_client()
  future = rc.get_checkpoint_archive_url_from_tinker_path(model_path)
  checkpoint_archive_url_response = future.result()

  
  base_dir = Path(__file__).resolve().parent
  dest = base_dir / "archive.tar"

  # download weights archive file from Tinker (.tar)
  urllib.request.urlretrieve(checkpoint_archive_url_response.url, dest.as_posix())

  # extract .tar
  src = "train_tinker/archive.tar"
  dst = "train_tinker/adapters/my_adapter"
  os.makedirs(dst, exist_ok=True)
  with tarfile.open(src, "r:*") as tar:   
      tar.extractall(path=dst, filter="data")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clefting Filler Gap Training')
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    download(model_path=args.model_path)