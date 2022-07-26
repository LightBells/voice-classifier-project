#Voice Classification Project

## Members
- m5261126 Ryo Kokubun #
- m5261158 Yumihosuke Sato #
- m5261170 Hikaru Takahashi #
(# equal contribution)

## Proposals
Proposals are shown [here](./docs/proposal.md)


## Usage

### Prepare Environment
1. Install [Poetry](https://python-poetry.org/docs/).
1. Install dependency packages via poetry by executing `poetry install` command.
1. Enter poetry shell by executing `poetry shell` command.

### Prepare Dataset
1. Download voice dataset from [Here](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset).
1. Extract the files into `res` folder. (e.g. `unzip -d speaker-recognition speaker-recognition-audio-dataset.zip`)
1. Execute re-format command on Project Root. The re-format command should be run in poetry shell.   
  The command is `python src/utils/converter.py ./res/{extracted_dir_name}` (e.g.`python src/utils/converter.py ./res/speaker-recognition/`).
1. Execute metadata creatation command on Project Root in poetry shell. The command is `python src/utils/metadata_generator.py`
1. Create the folder to store preprocessed data.  
  `mkdir ./res/preprocessed_data`
1. Execute file check and preparation command. The command is `python src/utils/split_and_create_melspectrum.py`. 
1. If there are broken files, please remove those file and preprocessed data folder,then go back to step 4.
1. After all, you obtain clean data.

### Training
1. See `./src/config.py`. The must of configurations are written on the file and can be modified.
1. Move into `src` folder and run `python train.py` command. Then, the training will begin.  
  The progress of training are shown in terminal via standard outputs, and we also provide a tensorboard support.   
  If you want to know training progress in GUI, run the command `tensorboard --logdir runs` on `src` directory.  

### Inference and Evaluation
TBC
