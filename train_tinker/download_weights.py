from pathlib import Path
import tinker
import urllib.request
import os, tarfile


weights_path = "tinker://9fdbefe2-dd83-5ec5-89dc-ee26b28b560d:train:0/sampler_weights/model_1_1_3B_epoch9"

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