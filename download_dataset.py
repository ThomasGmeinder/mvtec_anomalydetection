import urllib.request
import os
import sys

if os.path.isdir('carpet'):
    print("Dataset carpet is already downloaded, exiting...")
    sys.exit()


dataset_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz"
tar_file = "carpet.tar.xz"

print(f"Downloading Dataset from {dataset_url}")
urllib.request.urlretrieve(dataset_url,
                           tar_file)

import tarfile

print(f"Extracting {tar_file}")
with tarfile.open(tar_file) as f:
    f.extractall('.', numeric_owner=True)