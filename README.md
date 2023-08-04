# YOLO Braak Stage

Codebase accompanying the project for detection NFTs in WSIs and prediction of Braak stages in neuropathology cases.

This project is in the process of being published. When the publication is available it will found in this URL:

This project made use of various hundreds of WSIs, hosted in an instance of the Digital Slide Archive (DSA). We did not make these WSIs readily available but made the annotated images, used to train and evaluate machine learning (ML) models, availble for download.

We provide the original scripts and Jupyter notebooks used in the project to generate all the results and data.

## Data availability

We provide various data for use in recreating the results / analysis and for others to train their own models. All data can be accessed using the following [link](https://drive.google.com/drive/folders/16LUMrIMdp4LlvWQk5Dp3eVQHWY472jN5?usp=sharing). In all there is close to 200GB of data but not all is needed to download, and we break the data into different zip files for download. Only download what you need. 

For obtaining all the region of interests (ROIs) images with best labels you should simply download the "ROIs.zip" file and extract. 

Additional zip files contain data created throughout the project and is only needed for inspection of results.:
* models.zip: contains all the models trained
   - expert and novice named models are models trained on human labeled data from a single individual
   - models ihe model-assisted-labeling directory were all trained using the model assisted labeling (MAL) workflow on the large dataset
   - models with the name #-models-consensus-n# were models trained on the large dataset with labels created from consensus of various model predictions
* annotations.zip: contains small images centered on each NFT annotation provided by annotators
* rois.zip: contains large ROIs (regions of interest) that were annotated by human annotators
   - The ROI labels are saved in the format: label, x1, y1, x2, y2, with the coordinates relative to the ROI (e.g. int format)
* datasets.zip: Contains the various datasets created for training in this project.
   - annotator-datasets: contains the tile images for specific annotator models
   - test-datasets: contains both the ROIs and tiles for the test dataset
   - model-assisted-labeling: contains all the ROIs used in model-assisted-labeling and various predictions for the ROIs.
* results.zip: Contains the results created for the paper.
* wsi-inference.zip: Contains the WSI inference files, including low resolution binary masks of WSIs and the features extracted from NFT detection.

## Docker environment
This project is meant to be run in our Docker image - this will allow proper recreation of the results and the code to run without errors. The main use of this is not only that the Python libraries are compatible but also the filepaths to the many files needed in this project match.

Docker image is: jvizcar/braak-study:latest

This image should be run while mounting three directories:
* A data directory, if you download the files they should be placed in this directory.
* A code directory, which mounts this repository.
* A yolo directory, clone [our fork](https://github.com/jvizcar/nft-detection-yolov5) of Ultralytic's YOLOv5 repo.

## ClearML
The training of models in this repository **requires** the use of a [ClearML](clear.ml) account and credentials. After creating and logging into your ClearML account, go to settings -> workspace -> and create new credentials. Copy the text shown to a save location.

Once in the Docker terminal, initiate ClearML using ```$ clearml-init``` and paste the credentials you copied.

# Project scripts
The sctipts are numbered based on which part of the project they belong to, we include this for inspection but will not be runnable because of the need to access to the WSIs, which must be hosted in the DSA to work. 

Script descriptions:
1. Download annotated ROIs and images centered on NFT annotations. QC notebook double checks the bounding boxes and allows modification when needed.
2. Creates the ML datasets to train models specific to each annotator - trains these models.
3. Creates the MAL dataset - but does not run MAL. These scripts label the ROIs using the consensus approach, varying the number of models that must agree to give a label.
4. MAL workflow.
5. Adding additional datasets to train with MAL labeled data. Also includes the WSI inference workflow.

Results notebooks generates the results shown in the paper.


