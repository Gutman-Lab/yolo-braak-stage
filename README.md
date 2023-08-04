# YOLO Braak Stage

This script accompanies the journal article (currently in process of submission) titled: **Toward a Generalizable Machine Learning Workflow for Disease Staging with Focus on Neurofibrillary Tangles**

[Journal](Link coming soon...)

Codebase accompanying the project for detection NFTs in WSIs and prediction of Braak stages in neuropathology cases.

This project is in the process of being published. When the publication is available it will found in this URL:

This project made use of various hundreds of WSIs, hosted in an instance of the Digital Slide Archive (DSA). We did not make these WSIs readily available but made the annotated images, used to train and evaluate machine learning (ML) models, availble for download.

We provide the original scripts and Jupyter notebooks used in the project to generate all the results and data.
   
## Data availability

We provide various data for use in recreating the results / analysis and for others to train their own models. All data is hosted as zip files for download from a Google drive [here](https://drive.google.com/drive/folders/16LUMrIMdp4LlvWQk5Dp3eVQHWY472jN5?usp=sharing).
The data descriptions is as follows:

**ROIs.zip (30.36 GB).** All the ROI with the best labels created during model-assisted-labeling. Includes the 28 ROIs from testing. A csv file is included with metadata for each ROI. Labels are in text file format where the format follows the YOLO format of "label x-center y-center box-width box-height" normalized the the image width and height. This can be used to train new models and with our tutorial notebook / scripts that will be made available soon.

**models.zip (13.61 GB).** All the models trained.
* expert and novice named models are models trained on human labeled data from a single individual
* models ihe model-assisted-labeling directory were all trained using the model assisted labeling (MAL) workflow on the large dataset
* models with the name #-models-consensus-n# were models trained on the large dataset with labels created from consensus of various model predictions

**annotations.zip (10.78 GB)**. Small images centered on each NFT annotation provided by annotators.

**rois.zip (22.58 GB).** Large ROIs (regions of interest) that were annotated by human annotators.
* The ROI labels are saved in the format: label, x1, y1, x2, y2, with the coordinates relative to the ROI (e.g. int format)

**datasets.zip (98.51 GB).** Various datasets created for training in this project.
* annotator-datasets: contains the tile images for specific annotator models
* test-datasets: contains both the ROIs and tiles for the test dataset
* model-assisted-labeling: contains all the ROIs used in model-assisted-labeling and various predictions for the ROIs.

**wsi-inference.zip (288.5 MB).** WSI inference files, including low resolution binary masks of WSIs and the features extracted from NFT detection.

## Docker environment
This project is meant to be run in our Docker image - this will allow proper recreation of the results and the code to run without errors. The main use of this is not only that the Python libraries are compatible but also the filepaths to the many files needed in this project match.

Docker image is: jvizcar/braak-study:latest

This image should be run while mounting three directories:
* A data directory, if you download the zip files and extract them, they should be placed in this directory.
* A code directory, which mounts this repository.
* A yolo directory, clone [our fork](https://github.com/jvizcar/nft-detection-yolov5) of Ultralytic's YOLOv5 repo.

## ClearML
The training of models in this repository **requires** the use of a [ClearML](clear.ml) account and credentials. After creating and logging into your ClearML account, go to settings -> workspace -> and create new credentials. Copy the text shown to a save location.

Once in the Docker terminal, initiate ClearML using ```$ clearml-init``` and paste the credentials you copied.

## Project scripts
The sctipts are numbered based on which part of the project they belong to, we include this for inspection but will not be runnable because of the need to access to the WSIs, which must be hosted in the DSA to work. 

Script descriptions:
1. Download annotated ROIs and images centered on NFT annotations. QC notebook double checks the bounding boxes and allows modification when needed.
2. Creates the ML datasets to train models specific to each annotator - trains these models.
3. Creates the MAL dataset - but does not run MAL. These scripts label the ROIs using the consensus approach, varying the number of models that must agree to give a label.
4. MAL workflow.
5. Adding additional datasets to train with MAL labeled data. Also includes the WSI inference workflow.

Results notebooks generates the results shown in the paper.

## Tutorials
Coming soon, these will work with the ROI.zip file. This zip file includes the set of ROI images and best labels. It also includes boundary files which are needed because the ROIs are rotated and the images are the smallest bounding box around the annotated ROI. Tutorials coming include:
* Tiling ROI: the approach we use to split the ROI into smaller images, which saves each tile image with its appropriate label.
* Create dataset: create the appropriate text and yaml files needed to train / evaluate YOLO models
* Train: how to train models
* Evaluate: how to run validation on a dataset
* ROI inference: how to create ROI inference from tile predictions
* WSI inference: how to inference on a WSI
