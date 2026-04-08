
# Data-Efficient Down Syndrome Network (DE-DSNet) v1.00

## Associated Publications
If you use this work or any part of this repository in your research, please cite the following paper:
https://github.com/cwwang1979/Down-Syndrome-Net
- (Under submission) Wang et al. (2025) Deep learning for early Down syndrome screening on first-trimester ultrasound images
## Setup

#### Requirements
- Ubuntu 18.04
- GPU Memory => 16 GB
- GPU driver version >= 530.30.02
- GPU CUDA >= 12.1
- Python (3.8.20), opencv-python (4.11.0.86), PyTorch (2.4.1), torchvision (0.19.1).

#### Download
Data is download from the [zip](https://drive.google.com/drive/folders/1cOJHQR1HMLbqTkAoDAGilKgpWxn6wqf4) file. Please use the password on the associated paper to decompress the file.

Code, execution file, configuration file, and models are download from the [zip](https://drive.google.com/drive/folders/1amVTucHGRIKUqrMT0nC4Cq_kJ2q8Y3kA) file. Please use the password on the associated paper to decompress the file.

## Steps
### 1.Installation

Please refer to the following instructions.
```
# create and activate the conda environment
conda create -n DE_DSNet python=3.8 -y
conda activate DE_DSNet

# install related package
pip install -r requirements.txt
```
### 2. Run the augmentation script 

This project provides a parameterized augmentation script that can be executed from the terminal.

```
python propose_aug.py \
--input_dir ../Down_Syndrome_Data/Data \
--output_dir ../Down_Syndrome_Data/Data \
--aug_times 65 \
--apply_n_ops 2 \
--suffix D 

```

| Argument                                      | Description                                                        |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `--input_dir `                             | Path to the input image folder.                     |
| `--output_dir `                     | Path to the output folder for augmented images. |
| `--aug_times ` | Number of augmented images generated for each original image.             |
| `--apply_n_ops `                                | Number of random augmentation operations applied to each generated image.                 |
| `--suffix `                | Only images whose filename ends with this suffix before the file extension will be processed.                        |

### 3. Creates 5 folds cv dataset
This script creates classification dataset folders from a CSV split definition.  

#### Input requirements
The CSV file should contain at least the following columns:

- `file`: image filename
- `pos/neg`: class label (`1` for positive, `0` for negative)
- `f1`, `f2`, `f3`, `f4`, `f5`: split assignment for each fold (`train` or `test`)

#### Dataset structure

For each fold, the output directory will be organized as:

```bash
out_root/
├── folder1/
│   ├── train/
│   │   ├── pos/
│   │   └── neg/
│   └── test/
│       ├── pos/
│       └── neg/
├── folder2/
├── folder3/
├── folder4/
└── folder5/
```
#### Run 5 folds cv dataset script
```


```
#### Run single folds dataset script
```
python create_5cv_dataset.py \
  --csv_path ../Down_Syndrome_Data/list/split_temporal_with_Aug.csv \
  --main_folder ../Down_Syndrome_Data/Data \
  --out_root cls_dataset/split_temporal_with_Aug \
  --fold 1 \
  --prefer_symlink \
  --allow_hardlink \
  --save_miss_report \
  --make_inference_set
```
| Argument | Description |
| --- | --- |
| `--csv_path` | Path to the CSV file. |
| `--main_folder` | Folder containing the source images. |
| `--fold` | Fold index to run, such as `1`, `2`, `3`, `4`, or `5`. |
| `--run_all_folds` | Run all folds from `1` to `5`. |
| `--out_root` | Root output directory. |
| `--prefer_symlink` | Prefer symbolic links when creating dataset files. |
| `--allow_hardlink` | Allow hard links if symbolic links fail. |
| `--save_miss_report` | Save a CSV report for missing or invalid entries. |
| `--make_inference_set` | make infernce set. |

### 4. Training
The training script supports configurable parameters from the terminal.  
It can be used to train:

- a single fold
- all 5 folds sequentially
#### File Structure

Before training, the dataset directory should be organized as follows:

```bash
cls_dataset/
└── 5fold_CV_2020sample_with_Aug/
    ├── folder1/
    │   ├── train/
    │   │   ├── pos/
    │   │   └── neg/
    │   └── test/
    │       ├── pos/
    │       └── neg/
    ├── folder2/
    ├── folder3/
    ├── folder4/
    └── folder5/
```

#### Train 5 folds cv dataset 
```
python train.py \
  --model DE_DS_net.pt \
  --data 5fold_CV_2020sample_with_Aug \
  --epochs 500 \
  --imgsz 1024 \
  --batch 8 \
  --run_all_folds
```
#### Train a single fold
```
python train_cls.py \
  --model DE_DS_net.pt \
  --data split_temporal_with_Aug \
  --fold 1 \
  --epochs 500 \
  --imgsz 1024 \
  --batch 8
```
#### Output Directory
Training results will be saved under:

-./cls_trained_model/{data}/folder{fold}/{mode}/weights/best.pt

If all 5 folds are trained, the output structure will look like:

```
cls_trained_model/
└── 5fold_CV_2020sample_with_Aug/
    ├── folder1/
    │   └── DE_DS_net/weights/best.pt
    ├── folder2/
    │   └── DE_DS_net/
    ├── folder3/
    │   └── DE_DS_net/
    ├── folder4/
    │   └── DE_DS_net/
    └── folder5/
        └── DE_DS_net/
```

| Argument | Description |
| --- | --- |
| `--model` | Path or name of the model file. |
| `--data` | Dataset name under `./cls_dataset/`. |
| `--epochs` | Number of training epochs. |
| `--imgsz` | Input image size. |
| `--batch` | Batch size. |
| `--name` | Experiment name used for saving results. |
| `--fold` | Fold index to train, such as `1`, `2`, `3`, `4`, or `5`. |
| `--run_all_folds` | Train all folds from `1` to `5`. |

### 5. Inference 

To generate the prediction outcome of the DE_DSNet model in Down syndrome dataset, 

#### Inference 5 folds cv dataset 
```
python inference.py \
  --modelname DE_DS_net \
  --data_name 5fold_CV_2020sample_with_Aug \
  --imgsz 1024 \
  --run_all_folds
```
Or use a trained model directory:
```
python inference.py \
  --modelname best_model \
  --data_name 5fold_CV_2020sample_with_Aug \
  --imgsz 1024 \
  --run_all_folds
```
#### Inference a single fold
```
python inference.py \
  --model DE_DS_net \
  --data split_temporal_with_Aug \
  --fold 1 \
  --imgsz 1024 \
```


| Argument | Description |
| --- | --- |
| `--modelname` | Model experiment folder name. |
| `--data_name` | Dataset or experiment group name used in model and output paths. |
| `--fold` | Fold index to run, such as `1`, `2`, `3`, `4`, or `5`. |
| `--run_all_folds` | Run all folds from `1` to `5`. |
| `--imgsz` | Inference image size. |


#### Output strusture
Prediction results will be saved under:
```
./DE_DSNet_predictions/{data_name}/folder{fold}/{modelname}
```

```
DE_DSNet_predictions/
└── {data_name}/
    ├── folder1/
    │   └── {modelname}
    │       └── labels
    │           └── output.txt
    ├── folder2/
    │   └── {modelname}/
    ├── folder3/
    │   └── {modelname}/
    ├── folder4/
    │   └── {modelname}/
    └── folder5/
        └── {modelname}/

```
Each output .txt file corresponds to one input image and lists the classes in descending order of predicted probability (highest first):
```
<probability> <class_of_highest_probability>
<probability> <class_of_second_highest_probability>

```
For this study:

pos = Down syndrome fetus

neg = Normal fetus

## License
This Python source code is released under a creative commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology
