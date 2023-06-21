# yolo-braak-stage
Workflow for training YOLO models for detection of NFTs to use imaging features in Braak stage predictions.

This repository is used in an upcoming publication, the link to this publication will be added here when available.

The repository contains several sections:
* Collection of annotations from HistomicsUI / Digital Slide Archive instance
* An analysis of Braak stage inter-rater agreement and NFT inter-annotator agreement
* Creation of datasets and training of YOLOv5 object detection models for specific annotators
* Training models on larger datasets labeled by model prediction - using a consensus approach to intelligently combine predictions from models trained on data labeled by different annotators
  * In short - this computationally mimics a labeling workflow of multiple annotators sitting together to annotate the same images
* A model-assisted-labeling workflow to improve labels from the previous step
* Inference workflow of our models to predict NFTs on images of any size - including very large WSIs
* Feature extraction from predictions on WSIs - creates a feature vector for each case.
* Training of random forest models on the case feature vector to predict Braak stage

## Docker environment
This project is meant to be run in our Docker image - this will allow proper recreation of the results and the code to run without errors. The main use of this is not only that the Python libraries are compatible but also the filepaths to the many files needed in this project match.

Docker image is: jvizcar/braak-study:latest

This image should be run while mounting three directories:
* A data directory where all your output files will be saved - with the exception of some csv files
* A code directory, which mounts this repo
* A yolo directory, where our fork of Ultralytics' YOLOv5 repo exists
  * This is an older version of the current version found in Ultralytics GitHub, with slight modifications added to work in our project
* For the latter part of this project, we need access to the WSI files - and will not run unless you have those locally downloaded which requries hundreds of gigabytes of local storage. To do this, please contact us at jvizcar@emory.edu

