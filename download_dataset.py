import urllib.request
urllib.request.urlretrieve("https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
                           "carpet.tar.xz")

import tarfile

with tarfile.open('carpet.tar.xz') as f:
    f.extractall('.')