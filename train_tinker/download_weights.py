from pathlib import Path
import tinker
import urllib.request
import os, tarfile


weights_path = "tinker://2eb94980-5c6d-43b2-b434-a4142668f25e/sampler_weights/epoch_1"

sc = tinker.ServiceClient()
rc = sc.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path(weights_path)
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