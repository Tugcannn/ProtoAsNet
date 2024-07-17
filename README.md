# ProtoASNet

> **ProtoASNet: Dynamic Prototypes for Inherently Interpretable and Uncertainty-Aware Aortic Stenosis Classification in Echocardiography**              
> Hooman Vaseli*, Ang Nan Gu*, S. Neda Ahmadi Amiri*, Michael Y. Tsang*, Andrea Fung, Nima Kondori, Armin Saadat, Purang Abolmaesumi, Teresa S. M. Tsang </br>
> (*Equal Contribution) </br> 
> **Published in MICCAI 2023** </br> 
> [Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_36) </br> 
> [arXiv Link](https://arxiv.org/abs/2307.14433) 

--------------------------------------------------------------------------------------------------------
## Contents
- [Introduction](#Introduction)
- [Environment Setup](#Environment-Setup)
- [Train and Test](#Train-and-Test)
- [Description of Files and Folders](#Description-of-Files-and-Folders)
- [Citation](#Citation)


## Introduction 

ProtoASNet: Dynamic Prototypes for Inherently Interpretable and Uncertainty-Aware Aortic Stenosis Classification in Echocardiography for medmnist and ultrasound dataset.


--------------------------------------------------------------------------------------------------------
## Environment Setup

1. Clone the repo

```bash
git clone https://github.com/Tugcannn/ProtoAsNet.git
cd ProtoASNet
```
2. Place your data in the `data` folder. Download Ultrasound dataset and put inside 'data' folder https://drive.google.com/drive/my-drive ,

3. Also, we already trained bloodmnist model for 3-5-10 prototypes. Also you can find in driver and put inside 'logs/Image_ProtoAsNet'

4. Python library dependencies can be installed using:

```bash
pip install --upgrade pip
pip install torch torchvision  
pip install pandas wandb tqdm seaborn torch-summary opencv-python jupyter jupyterlab imageio array2gif moviepy scikit-image scikit-learn torchmetrics termplotlib
pip install -e .
# sanity check 
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

4. 

--------------------------------------------------------------------------------------------------------
## Train and Test

To train the model `cd` to the project folder, then use the command `python main.py` with the arguments described below:

- Before training you need to check

- "src/configs/Ours_ProtoASNet_Image.yml"`: yaml file containing hyper-parameters for model, experiment, loss objectives.

- You can change model architecture, epoch, num_classes and prototype_shape.

- "src/data/as_dataloader" : Implement class labels and dataset (data = MedMnist or data = Ult())

- After control these files run "main.py". You can follow backlogs

- "logs/<path-to-save>"` the folder to save all the trained model checkpoints, evaluations, and visualization of learned prototypes, you can try last model which is best validation accuracy.

**Dataset Checking and Evaluation**

*medmnist*

- "src/data/medMnistx.py" python file dataloading for medmnist dataset, you can try different medmnist datasets, batch sizes, normalizations and checking datasets.

- "src/data/evalx.py" evaluating the model on the test dataset, Computes and displays various evaluation metrics, including confusion matrix, classification report, Matthews correlation coefficient (MCC), ROC curves, and AUC scores.

- "src/data/explain_images.py" visualizing the prototypes and their influences on the model's predictions and saving the visualization results.

- "src/data/explain_images_without_names.py" visualizing the prototypes and their influences on the model's predictions and saving the visualization results without real names just (category 0,1,2..).

**Carefull**
- Need to write correct model_path, num_class and prototype shape in "evalx.py" and "explain_images.py"
- Use for ex: test_run_01 and epoch_30.pth 

*ultrasound*

- "src/data/ultrasoundload.py" python file dataloading for ultrasound dataset

- "src/data/ultraeval.py" evaluating the model on the ultrasound test dataset, Computes and displays various evaluation metrics, including confusion matrix, classification report, Matthews correlation coefficient (MCC), ROC curves, and AUC scores.

### outputs 

the important content saved in save_dir folder are:

- `model_best.pth`: checkpoint of the last model which is best model based on a metric of ACC


--------------------------------------------------------------------------------------------------------
## Description of files and folders

### logs
Once you run the system, it will contain the saved models, logs, and evaluation results (visualization of explanations, etc)

### pretrained_models
When training is done for the first time, pretrained backbone models are saved here.

### src
- `agents/`: folder containing agent classes for each of the architectures. contains the main framework for the training process
- `configs/`: folder containing the yaml files containing hyper-parameters for model, experiment, loss objectives, dataset, and augmentations.
- `data/`: folder for dataset, dataloader classes, evaluations and explanations
- `loss/`: folder for loss functions
- `models/`: folders for model architectures
- `utils/`: folder for some utility scripts and local explanation 


--------------------------------------------------------------------------------------------------------

## Citation
If you find this work useful in your research, please cite:
```
@InProceedings{10.1007/978-3-031-43987-2_36,
author="Vaseli, Hooman and Gu, Ang Nan and Ahmadi Amiri, S. Neda and Tsang, Michael Y. and Fung, Andrea and Kondori, Nima and Saadat, Armin and Abolmaesumi, Purang and Tsang, Teresa S. M.",
editor="Greenspan, Hayit and Madabhushi, Anant and Mousavi, Parvin and Salcudean, Septimiu
and Duncan, James and Syeda-Mahmood, Tanveer and Taylor, Russell",
title="ProtoASNet: Dynamic Prototypes for Inherently Interpretable and Uncertainty-Aware Aortic Stenosis Classification in Echocardiography",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="368--378",
isbn="978-3-031-43987-2"
}
```
