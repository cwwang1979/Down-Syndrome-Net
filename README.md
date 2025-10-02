
# Down-Syndrome-Net

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
Data, execution file, configuration file, and models are download from the [zip](https://drive.google.com) file.

## Steps
#### 1.Installation

Please refer to the following instructions.
```
# create and activate the conda environment
conda create -n DSNet python=3.8 -y
conda activate DSNet

# install related package
pip install ultralytics
```

#### 2. Inference 

To generate the prediction outcome of the DSNet model in partial Down syndrome dataset, 

```
python inference.py --stage predict --model DSNet1.pt --source "../Down_Syndrome_dataset/inference_dataset" --imgsz 1024 --save_txt=True --project "./inference_result" --name "DSNet1_predictions"
```

| Argument                                      | Description                                                        |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `--stage `                             | Indicates that the model is in inference mode.                     |
| `--model `                     | Path to the proposed model DSNet1.pt                            |
| `--source ` | Directory containing the test images             |
| `--imgsz `                                | Input feature space of size 1024×1024.                 |
| `--save_txt`                             | Saves predictions (classes, probabilities) in `.txt` format.      |
| `--project `                | Base directory where results will be saved.                        |
| `--name `                         | Subdirectory name under the project folder for this run's results. |




After inference, the output directory structure will be:

```
./DSNet1_predictions
└── DSNet1
    └── labels
        ├── test_image1.txt
        ├── test_image1.txt
        ├── ⋮
        └── test_imagen.txt

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

