"""Download CIFAR-10 from the Hugging Face mirror and save uint8 tensors to data/cifar10_{train,test}.pt.

(Much faster than torchvision's download from the Toronto server.) Label order is the standard
torchvision one: airplane, automobile, bird, cat(3), deer, dog(5), frog, horse, ship, truck.
"""
import io
import os
import urllib.request

import numpy as np
import pandas as pd
import torch
from PIL import Image

URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    for split in ["train", "test"]:
        pq = f"data/cifar10_{split}.parquet"
        urllib.request.urlretrieve(URL.format(split), pq)
        d = pd.read_parquet(pq)
        x = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in d["img"]])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(d["label"].values)
        torch.save({"x": x, "y": y}, f"data/cifar10_{split}.pt")
        os.remove(pq)
        print(split, tuple(x.shape), torch.bincount(y).tolist())
