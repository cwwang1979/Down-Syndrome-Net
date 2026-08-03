
# Data-Efficient Trisomy 21 Network (DE-T21Net) v1.00

## Associated Publications
If you use this work or any part of this repository in your research, please cite the following paper:
https://github.com/cwwang1979/Data-Efficient-Trisomy-21-Network
## Setup

#### Requirements
- Ubuntu 18.04
- GPU Memory => 16 GB
- GPU driver version >= 530.30.02
- GPU CUDA >= 12.1
- Python (3.8.20), opencv-python (4.11.0.86), PyTorch (2.4.1), torchvision (0.19.1).

#### Download
The trisomy 21 dataset that supports the findings of this study has been made publicly accessible on [zip](https://drive.google.com/drive/folders/1cOJHQR1HMLbqTkAoDAGilKgpWxn6wqf4). Please use the password on the associated paper to decompress the file.

The proposed DL model was deployed using the Pytorch framework in Python and the program code has been made publicly accessible on [zip](https://drive.google.com/drive/folders/1hD_05tudxcx7TBhWxnPPVxbX1CNlt3hP?usp=sharing). Please use the password on the associated paper to decompress the file.

## Steps
Three experiments were performed in this project:

- **A complete dataset with 2020 samples for 5-fold cross-validation**  
  This experiment trains the model on the full 2020-sample dataset using 5-fold cross-validation.

- **A reduced dataset with 1320 (65%) samples for 5-fold cross-validation to test Data Efficiency**  
  This experiment trains the model on the reduced 1320-sample dataset using 5-fold cross-validation.

- **Temporal-split dataset**  
  This experiment trains the model on the temporal-split dataset to evaluate performance under a chronological data split.
  
### 1.Installation

Please refer to the following instructions.
```
# create and activate the conda environment
conda create -n DE_DSNet python=3.8 -y
conda activate DE_DSNet

# install related package
pip install -r requirements.txt
```
### 2. Run the Proposed Augmentation script 

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

### 3. Creates dataset
This script creates classification dataset folders from a CSV split definition.  

#### Input requirements
The CSV file should contain at least the following columns:

- `file`: image filename
- `pos/neg`: class label (`1` for positive, `0` for negative)
- `f1`, `f2`, `f3`, `f4`, `f5`: split assignment for each fold (`train` or `test`)


#### Create 2020-sample dataset with 5-fold cross-validation
```
python create_5cv_dataset.py \
  --csv_path list/5fold_CV_2020sample_with_Aug.csv \
  --main_folder ../Down_Syndrome_Data/Data \
  --out_root cls_dataset/5fold_CV_2020sample_with_Aug \
  --run_all_folds \
  --prefer_symlink \
  --allow_hardlink \
  --save_miss_report \
  --make_inference_set
```

#### Create 1320-sample dataset with 5-fold cross-validation
```
python create_5cv_dataset.py \
  --csv_path list/5fold_CV_1320sample_with_Aug.csv \
  --main_folder ../Down_Syndrome_Data/Data \
  --out_root cls_dataset/5fold_CV_1320sample_with_Aug \
  --run_all_folds \
  --prefer_symlink \
  --allow_hardlink \
  --save_miss_report \
  --make_inference_set
```
#### Create temporal-split dataset
```
python create_5cv_dataset.py \
  --csv_path list/split_temporal_with_Aug.csv \
  --main_folder ../Down_Syndrome_Data/Data \
  --out_root cls_dataset/split_temporal_with_Aug \
  --fold 1 \
  --prefer_symlink \
  --allow_hardlink \
  --save_miss_report \
  --make_inference_set
```
#### Dataset structure

For each fold, the output directory will be organized as:

```
{out_root}/
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
 
#### File Structure

Before training, the dataset directory should be organized as follows:

```bash
cls_dataset/
└── {out_root}/
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

#### Train on the 2020-sample dataset with 5-fold cross-validation
```
python train.py \
  --model DE_DS_net.pt \
  --data 5fold_CV_2020sample_with_Aug \
  --epochs 500 \
  --imgsz 1024 \
  --batch 8 \
  --run_all_folds
```
#### Train on the 1320-sample dataset with 5-fold cross-validation
```
python train.py \
  --model DE_DS_net.pt \
  --data 5fold_CV_1320sample_with_Aug \
  --epochs 500 \
  --imgsz 1024 \
  --batch 8 \
  --run_all_folds
```

#### Train on the temporal-split dataset
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
└── {data}/
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

To generate the prediction outcome of the DE_DSNet model in trisomy 21 dataset, 

#### Inference on the 2020-sample dataset with 5-fold cross-validation
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
#### Inference on the 1320-sample dataset with 5-fold cross-validation
```
python inference.py \
  --modelname DE_DS_net \
  --data_name 5fold_CV_1320sample_with_Aug \
  --imgsz 1024 \
  --run_all_folds
```

#### Inference on the temporal-split dataset
```
python inference.py \
  --modelname DE_DS_net \
  --data_name split_temporal_with_Aug \
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

pos = Fetus with trisomy 21 

neg = Normal fetus

## 6. DE-T21Net-Lite Inference for Fetal Ultrasound Images and Videos

The DE-T21Net-Lite inference pipeline described in this section is separate from the standard PyTorch inference pipeline described in Section 5.

DE-T21Net-Lite uses a hardware-optimized TensorRT inference engine to improve inference efficiency on the target GPU platform. The Lite inference script supports image-level classification and frame-by-frame classification of prerecorded videos. It generates class probabilities, TP/TN/FP/FN results, inference-time measurements, annotated outputs, and CSV summaries.

The two inference pipelines use different scripts, model formats, software environments, and output formats:

| Pipeline | Script | Model format | Environment |
|---|---|---|---|
| Standard dataset inference | `inference.py` | PyTorch `.pt` | Training and standard inference environment |
| DE-T21Net-Lite inference | `DE_T21Net_Lite_inference.py` | TensorRT `.engine` | Separate Lite inference environment |

---

### 6.1 Download and Package Structure

The DE-T21Net-Lite inference package is distributed separately from the training and standard PyTorch inference code.

Recommended archive filename:

```text
DE-T21Net_v1.00_Lite_Inference_for_Fetal_Ultrasound_Images_and_Videos.zip
```

<!-- Replace LITE_INFERENCE_PACKAGE_DOWNLOAD_LINK with the actual download URL. -->

The DE-T21Net-Lite inference package is available on [zip](https://drive.google.com/file/d/10btthL5D3CQaf5xVs80bpu1dIe3Jgmkv/view?usp=drive_link). Please use the password provided in the associated paper to decompress the file.

After decompression, the package should be organized as follows:

```text
DE-T21Net_v1.00_Lite_Inference_for_Fetal_Ultrasound_Images_and_Videos/
├── DE_T21Net_Lite_inference.py
├── requirements.txt
├── models/
│   └── DE_T21Net_Lite_fp16.engine
├── input/
│   ├── pos/
│   └── neg/
└── output/
```

The released package includes:

- the DE-T21Net-Lite inference script;
- the trained TensorRT `.engine` model;
- a separate `requirements.txt` file for the Lite inference environment; and
- fetal ultrasound images for image inference.

The original video dataset is not included in the package. Video inference can still be performed using user-provided videos that follow the required directory structure.

Eight six-second video inference demonstrations are provided in Section 6.6.

---

### 6.2 DE-T21Net-Lite Environment Setup

The DE-T21Net-Lite inference pipeline requires a separate environment from the training and standard PyTorch inference pipeline described in the previous sections.

The tested Lite inference environment includes:

| Package | Version |
|---|---:|
| Python | 3.11.15 |
| CUDA Toolkit | 13.0.3 |
| TensorRT | 11.1.0.106 |
| PyTorch | 2.13.0+cu130 |
| torchvision | 0.28.0+cu130 |
| Ultralytics | 8.4.104 |
| OpenCV | 4.13.0 |
| NumPy | 1.26.4 |
| pandas | 3.0.3 |

The extracted Lite inference package contains its own:

```text
requirements.txt
```

This file is different from the `requirements.txt` used in Section 1.

The main project environment in Section 1 is installed using:

```bash
pip install -r requirements.txt
```

In contrast, the `requirements.txt` included in the DE-T21Net-Lite inference package is a Conda package specification file. It must be used with `conda create --file` rather than `pip install -r`.

From the extracted Lite inference package directory, create and activate the environment as follows:

```bash
# create the separate DE-T21Net-Lite inference environment
conda create --name DE_T21Net_Lite_Inference \
  --file requirements.txt

# activate the environment
conda activate DE_T21Net_Lite_Inference
```

Do not install the Lite inference package requirements using:

```bash
pip install -r requirements.txt
```

The environment requirements in the main Setup section apply to model training and standard PyTorch inference. They do not apply to the DE-T21Net-Lite inference pipeline described in this section.

---

### 6.3 Fixed Inference Settings

The following settings are fixed in the DE-T21Net-Lite inference source code to ensure consistent evaluation:

| Setting | Value |
|---|---:|
| Input image size | 1024 × 1024 |
| POS confidence threshold | 0.5 |
| Video POS ratio threshold | 0.5 |
| Rotate phase duration | 3.0 seconds |
| Terminal print interval | 10 frames |

For image-level and frame-level classification:

```text
POS confidence >= 0.5 -> POS
POS confidence < 0.5  -> NEG
```

For video-level classification:

```text
positive_frames / total_frames >= 0.5 -> POS
positive_frames / total_frames < 0.5  -> NEG
```

The input image size, classification thresholds, rotate phase duration, and terminal print interval are fixed in the source code and cannot be modified through command-line arguments.

The rotate phase duration is used to identify the transformation phase in the six-second augmented demonstration videos. The first 3.0 seconds correspond to the rotation phase. This setting is used for phase display and does not change the frame-level or video-level classification thresholds.

The class labels are defined as follows:

```text
pos -> fetus with trisomy 21
neg -> healthy fetus
```

---

### 6.4 Image Inference

The DE-T21Net-Lite script can perform classification on fetal ultrasound images placed under the `input` directory.

#### Input Directory Structure

For image inference, the `input` directory must contain `pos` and `neg` subdirectories:

```text
input/
├── pos/
│   ├── positive_image_1.png
│   └── positive_image_2.png
└── neg/
    ├── negative_image_1.png
    └── negative_image_2.png
```

The parent directory determines the ground-truth label:

```text
input/pos/... -> GT = pos
input/neg/... -> GT = neg
```

#### Run Image Inference

Run the following command from the extracted DE-T21Net-Lite package directory:

```bash
python DE_T21Net_Lite_inference.py \
  --model-path models/DE_T21Net_Lite_fp16.engine \
  --input-dir input \
  --output-dir output
```

To select a specific GPU and display the OpenCV inference window, use:

```bash
python DE_T21Net_Lite_inference.py \
  --model-path models/DE_T21Net_Lite_fp16.engine \
  --input-dir input \
  --output-dir output \
  --device 0 \
  --show-window
```

| Argument | Description |
|---|---|
| `--model-path` | Path to the DE-T21Net-Lite TensorRT `.engine` model. |
| `--input-dir` | Root input directory containing the `pos` and `neg` subdirectories. |
| `--output-dir` | Root output directory for annotated images and `image_summary.csv`. |
| `--device` | GPU device index used for inference, such as `0`. |
| `--show-window` | Display the OpenCV inference window while processing the input files. |

Users with a different directory structure should replace the example paths with their local paths:

```bash
python DE_T21Net_Lite_inference.py \
  --model-path /path/to/DE_T21Net_Lite_fp16.engine \
  --input-dir /path/to/input \
  --output-dir /path/to/output
```

#### Image Output Structure

```text
output/
├── image_summary.csv
└── images/
    ├── pos/
    └── neg/
```

The `pos` and `neg` output subdirectories contain annotated image results corresponding to the input ground-truth class directories:

```text
input/pos/... -> output/images/pos/...
input/neg/... -> output/images/neg/...
```

Each annotated image displays:

```text
Ground-truth label
Predicted image label
TP/TN/FP/FN result
POS class probability
NEG class probability
Inference time
```

The displayed confidence values are the model probabilities for the two classes:

```text
POS conf = model probability for the POS class
NEG conf = model probability for the NEG class
```

In `image_summary.csv`, `Probability` represents the POS-class probability:

```text
Image Probability = POS conf
```

The final image prediction is determined from the POS-class probability:

```text
POS conf >= 0.5 -> POS
POS conf < 0.5  -> NEG
```

Example:

```text
GT: NEG
Image Pred: NEG
Result: TN
POS conf: 0.0000
NEG conf: 1.0000
```

---

### 6.5 Video Inference

The original video dataset is not distributed in the DE-T21Net-Lite inference package. However, the script supports frame-by-frame inference on prerecorded videos supplied by the user.

#### Input Directory Structure

For video inference, the `input` directory must contain `pos` and `neg` subdirectories:

```text
input/
├── pos/
│   ├── positive_video_1.mp4
│   └── positive_video_2.mp4
└── neg/
    ├── negative_video_1.mp4
    └── negative_video_2.mp4
```

The parent directory determines the ground-truth label:

```text
input/pos/... -> GT = pos
input/neg/... -> GT = neg
```

#### Run Video Inference

Run the following command from the extracted DE-T21Net-Lite package directory:

```bash
python DE_T21Net_Lite_inference.py \
  --model-path models/DE_T21Net_Lite_fp16.engine \
  --input-dir input \
  --output-dir output
```

To select a specific GPU and display the OpenCV inference window, use:

```bash
python DE_T21Net_Lite_inference.py \
  --model-path models/DE_T21Net_Lite_fp16.engine \
  --input-dir input \
  --output-dir output \
  --device 0 \
  --show-window
```

| Argument | Description |
|---|---|
| `--model-path` | Path to the DE-T21Net-Lite TensorRT `.engine` model. |
| `--input-dir` | Root input directory containing prerecorded videos under the `pos` and `neg` subdirectories. |
| `--output-dir` | Root output directory for annotated videos and `video_summary.csv`. |
| `--device` | GPU device index used for inference, such as `0`. |
| `--show-window` | Display the OpenCV frame-by-frame inference window during video processing. |

Users with a different directory structure should replace the example paths with their local paths:

```bash
python DE_T21Net_Lite_inference.py \
  --model-path /path/to/DE_T21Net_Lite_fp16.engine \
  --input-dir /path/to/input \
  --output-dir /path/to/output
```

#### Video Output Structure

```text
output/
├── video_summary.csv
├── pos/
└── neg/
```

The `pos` and `neg` output subdirectories contain annotated video results corresponding to the input ground-truth class directories.

Each video is processed frame by frame. Every frame receives:

```text
POS conf
NEG conf
Frame-level prediction
TP/TN/FP/FN result
Inference time
```

For videos, `Probability` is calculated as:

```text
Video Probability = positive_frames / total_frames
```

This video-level `Probability` is the proportion of frames classified as POS. It is not the average POS confidence across all frames.

The final video-level prediction is determined using the fixed video POS ratio threshold:

```text
Video Probability >= 0.5 -> POS
Video Probability < 0.5  -> NEG
```

Example:

```text
positive_frames = 120
total_frames = 180

Video Probability = 120 / 180
                  = 0.6667

Video prediction = POS
```

---

### 6.6 Video Inference Demonstrations

The following six-second videos demonstrate frame-by-frame DE-T21Net-Lite inference on augmented fetal ultrasound videos.

These links provide inference-result demonstrations only. The original video dataset is not distributed in the DE-T21Net-Lite inference package.

#### Trisomy 21

1. [Trisomy 21 inference demonstration 1](https://youtube.com/shorts/JF52DIa5GIc)
2. [Trisomy 21 inference demonstration 2](https://youtube.com/shorts/tiqn9pOM-JM)
3. [Trisomy 21 inference demonstration 3](https://youtube.com/shorts/VMcSjfZ5Oao)
4. [Trisomy 21 inference demonstration 4](https://youtube.com/shorts/Lh6ruJX24wY)

#### Healthy Fetus

1. [Healthy fetus inference demonstration 1](https://youtube.com/shorts/ej8i_97H4Kw)
2. [Healthy fetus inference demonstration 2](https://youtube.com/shorts/eNmrUT0pbv8)
3. [Healthy fetus inference demonstration 3](https://youtube.com/shorts/sOaof6E9pwE)
4. [Healthy fetus inference demonstration 4](https://youtube.com/shorts/vcc03yzgzK8)

---

### 6.7 Reproducibility

To reproduce the DE-T21Net-Lite inference results as closely as possible, use the same:

- TensorRT engine file;
- input images or videos;
- fixed inference settings;
- Python and package versions;
- CUDA version;
- TensorRT version; and
- compatible NVIDIA GPU architecture.

When the same TensorRT engine, input data, inference settings, software environment, and compatible hardware are used, the predicted classes and TP/TN/FP/FN results are expected to remain consistent.

Small numerical differences in `POS conf` and `NEG conf` may still occur because of floating-point computation, TensorRT optimization, GPU architecture, and software-version differences. A small confidence difference usually does not change the final classification unless the POS-class probability is close to the fixed threshold of `0.5`.

Runtime measurements are hardware- and system-dependent. The following values may vary between runs:

```text
Inference time
Processing FPS
Elapsed time
GPU utilization
GPU memory usage
```

Possible causes include GPU load, CPU load, system temperature, storage performance, video-decoding overhead, and background processes.

Therefore, runtime measurements should be evaluated under a controlled and clearly documented environment.

---

### 6.8 DE-T21Net-Lite Engine Compatibility

The DE-T21Net-Lite model is provided as a hardware-optimized TensorRT `.engine` file. TensorRT engine files are platform-dependent binary files, and compatibility may depend on:

- NVIDIA GPU architecture;
- CUDA version;
- TensorRT version;
- NVIDIA driver version; and
- TensorRT build configuration.

The released DE-T21Net-Lite inference package provides the TensorRT `.engine` model but does not include the original PyTorch or ONNX model used to build the engine.

If the provided engine is compatible with the target environment, it can be loaded directly:

```bash
python DE_T21Net_Lite_inference.py \
  --model-path models/DE_T21Net_Lite_fp16.engine \
  --input-dir input \
  --output-dir output
```

If the engine cannot be loaded because of a GPU, CUDA, TensorRT, or driver incompatibility, a new TensorRT engine must be exported from the original PyTorch or ONNX model in the target environment.

Because the original PyTorch and ONNX model files are not included in the DE-T21Net-Lite inference package, an incompatible engine cannot be rebuilt from the released package alone.



## License
This Python source code is released under a creative commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology
