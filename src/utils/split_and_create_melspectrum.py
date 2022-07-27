import argparse
import os

import pandas as pd
import numpy as np

from tqdm import tqdm
import librosa
import librosa.display

parser = argparse.ArgumentParser(description="Voice Cutter")
parser.add_argument(
    "--metadata",
    type=str,
    help="Specify a metadata(csv)",
    default="res/speaker-recognition/metadata.csv",
)
parser.add_argument(
    "--target_dir",
    type=str,
    help="Specify a target directory",
    default="res/preprocessed_data/",
)
parser.add_argument(
    "--prefix",
    type=str,
    default="./",
    help="Specify a prefix to adjust difference between the path\
            on metadata file and running environment",
)
args = parser.parse_args()

metadata = pd.read_csv(args.metadata)
metadata_test = metadata[metadata.audio_file_id == 10]
metadata_valid = metadata[metadata.audio_file_id == 11]
metadata = metadata[metadata.audio_file_id < 10]

# Output metadata
data_size = metadata.shape[0]
print("Data Size: ", data_size)

unique_speakers = metadata.speaker_id.unique()
print("Speaker IDs: ", unique_speakers)


# ========== Utils ==============
# load a wave data
def load_wave_data(file_name):
    x, fs = librosa.load(file_path, sr=44100)
    return x, fs


# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) ** 2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp


# ===============================


# Create Data folders
min_value = 2641760
folders = ["train", "test", "validation"]
for folder in folders:
    dir_name = os.path.join(args.target_dir, folder)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for speaker in unique_speakers:
        speaker_dir_name = os.path.join(dir_name, str(speaker))
        if not os.path.exists(speaker_dir_name):
            os.mkdir(speaker_dir_name)

    if folder == "train":
        data = metadata
    elif folder == "test":
        data = metadata_test
    elif folder == "validation":
        data = metadata_valid

    for idx, row in tqdm(data.iterrows()):
        file_path = os.path.join(args.prefix, row["path"])
        speaker_id = row["speaker_id"]
        speaker_dir_name = os.path.join(dir_name, str(speaker_id))

        filename_with_extention = row["filename"]
        filename = row["filename"][:-4]

        try:
            wave_file, fs = load_wave_data(file_path)
        except Exception as e:
            print(e)
            print(filename)
        else:
            wave_file = wave_file[:min_value]
            files = np.array_split(wave_file, 10)
            for idx, file in enumerate(files):
                melsp = calculate_melsp(file)
                np.save(
                    os.path.join(speaker_dir_name, filename + "_" + str(idx)),
                    melsp,
                )
