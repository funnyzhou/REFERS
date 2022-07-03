# Code for REFERS
This repository provides the official implementation of our paper "[Generalized radiograph representation learning via cross-supervision between images and free-text radiology reports](https://www.nature.com/articles/s42256-021-00425-9.pdf)", which was published at Nature Machine Intelligence, 2022.

**Based on cross-supervised learning (text as supervision), REFERS generates pre-trained models which produce state-of-the-art results on different radiograph benchmarks!**

This project is divided into two parts: pre-training and fine-tuning. 

## Updates
7.3.2022 Upload missing files: train/val_img_captions_patients.txt ([Google Drive](https://drive.google.com/drive/folders/1tsSbvrDVJeDNBRV0QE0VCcVwM8lh8UBk?usp=sharing))

---
## Installation
#### 1. System requirements
This software was originally designed and run on a system running Ubuntu 18.01, with Python 3.6, PyTorch 1.7, and CUDA 10.1. For a full list of software packages and version numbers, see the Conda environment file `environment.yml`. 

This software leverages graphical processing units (GPUs) to accelerate neural network training and evaluation; systems lacking a suitable GPU will likely take an extremely long time to train or evaluate models. The software was tested with the NVIDIA RTX 2080 TI GPU, though we anticipate that other GPUs will also work, provided that the unit offers sufficient memory. 

#### 2. Installation guide
We recommend installation of the required packages using the Conda package manager, available through the Anaconda Python distribution. Anaconda is available free of charge for non-commercial use through [Anaconda Inc](https://www.anaconda.com/products/individual). After installing Anaconda and cloning this repository, use the `conda` command to install necessary packages:
    `conda env create -f environment.yml` 

#### 3. Description of folders
- **Pre-train**
This folder contains the implementation of REFERS.
- **Fine-tune**
The fine-tuning codes are stored in this folder. The pretrained model named *caption_100_bestauc_checkpoint.bin* is also provided for your reference.
- **DataProcessed**
This folder contains the data pre-processing codes used in the pre-training and fine-tuning stages. We also include lists of training (100%), validation and testing on all datasets.
- **Data**
The downloaded data should be put in this directory.

---

## Pre-training
We use MIMIC-CXR-JPG for pre-training. You can acquire more information about this dataset at [Johnson et al. MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

#### 1. Setting up datasets
While we provide code to load data for training a deep-learning model, you will first need to download images from the above repositories. We provide a data TXT file, the content inside is a collection of the path name and label of the picture, which will facilitate the reading of the data.

Regarding the downloaded data, I will not introduce too much here, you can go to the corresponding website to view it. Please organize MIMIC  dataset in **DataProcessed** as follows:

```python
./DataProcessed/
    MIMIC/
		train_label.npy
		test_label.npy
		train.txt
		test.txt
		mimic_resize_image.ipynb
```

After the data are downloaded, we preprocess the data according to our needs to obtain the required image paths and label files (train_label.npy, test_label.npy, train_data.txt, test_data.txt). In order to facilitate the reading of the data, we first preprocess the image using offline scaling and save it locally. You can refer to *mimic_resize_image.ipynb* for more details.

#### 2. Conduct pre-training using REFERS

This is the command for using 1 GPU for pre-training. It takes about 2 days on one NVIDIA 2080Ti.

```
python3 pretrain_refers.py --num-gpus-per-machine 1 --cpu-workers 32  --serialization-dir "../data/"  --dist-url "tcp://127.0.0.1:12343"  --checkpoint-every 100
```

---

## Fine-tuning
#### 1. Datasets
Datasets can be downloaded via following links:

**Dataset I**
[COVID-19 Image Data Collection]( https://github.com/ieee8023/covid-chestxray-dataset)

**Dataset II**
[Shenzhen Tuberculosis](https://www.kaggle.com/raddar/tuberculosis-chest-xrays-shenzhen)

**Dataset III**
[NIH ChestX-ray](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)

**Dataset IV**
[VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)

#### 2. Setting up datasets
Similar to pre-training, you need to first download images from above repositories before running our codes.

Notes: 
a) Perhaps you need to modify the data paths by yourself.
b) The provided training lists are for fine-tuning using 100% labeled data. For fine-tuning with 1% or 10% labeled data, you just to randomly selected 1% or 10% rows (i.e., data) from the provided training lists. We strongly recommend to repeat each experiment for 3 times.

The 4 datasets for fine-tuning in **DataProcessed** have been organized as follows:

    ./DataProcessed/
        COVID-19_Image_Data_Collection/
    		train.txt
    		val.txt
    		test.txt
        Shenzhen_Tuberculosis/
    		train.txt
    		val.txt
    		test.txt
        NIH_ChestX-ray/
    		train.txt
    		val.txt
    		test.txt	
        VinBigData_Chest_X-ray/
    		train.txt
    		val.txt
    		test.txt
    		data_processed.ipynb

#### 3. Fine-tune models
##### Data preparation

- COVID-19 Image Data Collection
Put the downloaded data in `./data/COVID-19_Image Data/.` You will need to put all images in `./data/COVID-19_Image Data/images` and use the provided list files in `./DataProcessed/COVID-19_Image_Data_Collection/` to read the data.
- Shenzhen Tuberculosis
Put the downloaded data in `./data/Shenzhen_Tuberculosis/.` You will need to put all images in `./data/Shenzhen_Tuberculosis/images` and use the list files in `./DataProcessed/Shenzhen_Tuberculosis/` for training, validation and testing splits.
-  NIH ChestX-ray
Put the downloaded data in `./data/NIH_ChestX-ray` .  You will need to extract all zip files and put all images in `./data/NIH_ChestX-ray/images` and use the list files in `./DataProcessed/NIH_ChestX-ray/` to read the data.
- VinBigData Chest X-ray
Please put the downloaded data in `./data/VinBigData_Chest_X-ray/.` You will need to extract all zip files and divide all images into two directories `./data/VinBigData_Chest_X-ray/train` and `./data/VinBigData_Chest_X-ray/test`. Considering the original X-ray is quite large, to speed up the training efficiency, we performed an offline scaling process on the picture. You can get the scaled picture by running the data_processed.ipynb code, and then modify the corresponding data paths and use the provided list files in `./DataProcessed/VinBigData_Chest_X -ray/` to read the data.

##### Training models

Firstly, you should download the pre-trained model via [this link](https://drive.google.com/file/d/1JvtcYOdLWC9BTzCnCJqNzCP7R6CmUK1N/view?usp=sharing) and put the model in `Fine-tune/checkpoint/`.

- COVID-19 Image Data Collection
```
python3 -m torch.distributed.launch --nproc_per_node=2  --master_addr 127.0.0.2 --master_port 29504  train.py --name caption_100 --stage train --model_type ViT-B_16 --num_classes 1 --pretrained_dir "../checkpoint/refers_checkpoint.pth" --output_dir "./output/" --data_volume '100' --num_steps 500  --eval_batch_size 512 --img_size 224 --learning_rate 3e-2 --warmup_steps 50 --fp16 --fp16_opt_level O2 --train_batch_size 128
```

- Shenzhen Tuberculosis
```
python3 -m torch.distributed.launch --nproc_per_node=2  --master_addr 127.0.0.2 --master_port 29504  train.py --name caption_100 --stage train --model_type ViT-B_16 --num_classes 1 --pretrained_dir "../checkpoint/refers_checkpoint.pth" --output_dir "./output/" --data_volume '100' --num_steps 100  --eval_batch_size 512 --img_size 224 --learning_rate 3e-2 --warmup_steps 5 --fp16 --fp16_opt_level O2 --train_batch_size 128
```

-  NIH ChestX-ray
```
python3 -m torch.distributed.launch --nproc_per_node=2  --master_addr 127.0.0.2 --master_port 29509  train.py --name caption_100 --stage train --model_type ViT-B_16 --num_classes 14 --pretrained_dir "../checkpoint/refers_checkpoint.pth" --output_dir "./output/" --data_volume '100' --num_steps 60000  --eval_batch_size 512 --img_size 224 --learning_rate 1e-2 --warmup_steps 1000 --fp16 --fp16_opt_level O2 --train_batch_size 128
```

- VinBigData Chest X-ray
```
python3 -m torch.distributed.launch --nproc_per_node=2  --master_addr 127.0.0.2 --master_port 29504  train.py  --name caption_100 --stage train --model_type ViT-B_16 --num_classes 14 --pretrained_dir "../checkpoint/refers_checkpoint.pth" --output_dir "./output/" --data_volume '100' --num_steps 100  --eval_batch_size 512 --img_size 224 --learning_rate 3e-2 --warmup_steps 10 --fp16 --fp16_opt_level O2 --train_batch_size 128
```

#### 3. Evaluation to reproduce the results

Once you have finished the fine-tuning process, you can choose to run `python test.py`in corresponding folders to generate the evaluation results (i.e., AUC) on all 4 datasets.

Results should be saved to `Results_of_fine-tuning`, where we provided our prediction files (for once run) to reproduce simililar ROC curves compared to those provided in the supplementary information. The folders have been organized as follows:

    ./Results_of_fine-tuning/
        COVID-19_Image_Data_Collection/
    		covid_others/
    		bacterial_viral/
        Shenzhen_Tuberculosis/
    		tuberculosis/
        NIH_ChestX-ray/
    		data1/
    		data10/
         	data100/
        VinBigData_Chest_X-ray/
    		data1/
    		data10/
         	data100/

In each sub-folder, we provide the prediction files of 6 approaches (i.e., 5 baselines + ours). Each prediction is organised as `True label (1/0) + probability (0-1)`. We recommend you to use the officially released code of all baselines to generate pre-trained models, on top of which you can conduct fine-tuning on 4 downstream datasets using provided fine-tuning scripts (just remember to modify the codes for loading models). 