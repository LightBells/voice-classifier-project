import pandas as pd
import numpy as np
from glob import glob
import argparse

parser = argparse.ArgumentParser(description="Metadata generator")
parser.add_argument("--source_dir", type = str, help="Specify a source directory", default="res/speaker-recognition/")
parser.add_argument("--target_dir", type = str, help="Specify a target directory", default="res/speaker-recognition/")
args = parser.parse_args()

paths = glob(args.source_dir+"/**/**.wav", recursive=True)
paths = np.array(paths)
df = pd.DataFrame(paths.T, columns=["path"])
df.loc[:, "filename"] = df.path.apply(lambda x: x.split("/")[-1])
df.loc[:, "speaker_id"] = df.filename.apply(lambda x: int(x.split(".")[0].split("_")[1]))
df.loc[:, "audio_file_id"] = df.filename.apply(lambda x: int(x.split(".")[0].split("_")[2]))

df.to_csv(args.target_dir+"/metadata.csv")
