from torch.utils.data import Dataset
from glob import glob
import os
import re
import numpy as np


class VoiceDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.id_to_speakers = dict(enumerate(map(int, os.listdir(path))))
        self.speakers_to_id = dict(
            map(lambda x: (int(x[1]), x[0]), enumerate(os.listdir(path)))
        )

        self.files = glob(path + "/**/**.npy")

        self.regexp = re.compile(r"Speaker_(\d+)_\d+_\d+.npy")

        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        speaker_id = int(self.regexp.findall(file_path)[0])
        file = np.load(file_path)
        if self.transforms is not None:
            file = self.transforms(file)
        return file, self.speakers_to_id[speaker_id]


if __name__ == "__main__":
    ds = VoiceDataset("../res/preprocessed_data/train")
    for d, l in ds:
        print(d, l)
