import os
from glob import glob
import re

import argparse

parser = argparse.ArgumentParser(description="Directory and filename renamer")
parser.add_argument("target_dir", type = str, help="Specify a target directory")
args = parser.parse_args()

# define Utils
def confirmer() -> str:
    confirm = input("Confirm(Y/n)?: ") 

    while confirm.lower() not in ["", "y", "n"]:
        confirm = input("Confirm(Y/n)?: ")

    if confirm == "":
        confirm = 'y'
    confirm = confirm.lower()

    return confirm


# Change directory name
print("# Directory Renamer")

dirs = glob(os.path.join(args.target_dir, "50_speakers_audio_data/Speaker*"))
subject_dirs = list(filter(lambda x: re.search("Speaker\d{4}", x) is not None, dirs))

replace_dirname = lambda x: re.sub("Speaker(\d{4})", "Speaker_\\1", x)
mapping = list(map(lambda x: (x, replace_dirname(x)), subject_dirs))

print("The directory names are replaced like following way.")
for b, a in mapping[:min(len(mapping), 4)]:
    print(b, "->", a)

confirm = confirmer()

if confirm == "y":
    for b, a in mapping:
        os.rename(b, a)
    print("Renamed")
else:
    exit()


# Change name of each file
print("# File renamer")

files = glob(os.path.join(args.target_dir, "50_speakers_audio_data/**/**.wav"), recursive=True)
subject_files = list(filter(lambda x: re.search("Speaker\d+_\d+.wav", x), files))

replace_filename = lambda x: re.sub("Speaker(\d+_\d+).wav", "Speaker_\\1.wav", x)
mapping = list(map(lambda x: (x, replace_filename(x)), subject_files))

print("The file names are replaced like following way.")
for b, a in mapping[:min(len(mapping), 4)]:
    print(b, "->", a)

confirm = confirmer()

if confirm == "y":
    for b, a in mapping:
        os.rename(b, a)
    print("Renamed")
else:
    exit()
