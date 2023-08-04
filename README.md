# YOLO Braak Stage
[Manuscript is currently in submission] [[Data](https://drive.google.com/drive/folders/16LUMrIMdp4LlvWQk5Dp3eVQHWY472jN5?usp=sharing)]

Codebase accompanying the project for detection NFTs in WSIs and prediction of Braak stages in neuropathology cases.

We provide the original scripts and Jupyter notebooks used in the project to generate all the results and data.

## Data availability
This project utilized hundreds of WSIs, hosted in an instance of the Digital Slide Archive (DSA). We did not make these WSIs readily available but made the annotated images, used to train and evaluate machine learning (ML) models, availble for download.

We provide the final data directories used during this project as well as a simplified zip file ("ROIS.zip") containing all the ROIs with out best set of labels, created from model-assisted-labeling workflow. While some of the scripts may run, any of the scripts that require integration with the DSA will not. We do not recommend running these scripts but instead taking a look a the tutorial notebooks.

**ROIs.zip** contains the all the ROIs, including our test dataset, with the best set of labels. 

Additional zip files contain data created throughout the project and is only needed for inspection:
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
A Docker image is provided for running our scripts and tutorial Jupyter notebooks (see README.md in tutorial folder).

Docker image is: jvizcar/braak-study:latest

This image contains this repo and our [fork](https://github.com/jvizcar/nft-detection-yolov5) of Ultralytics YOLOv5 [repo](https://github.com/ultralytics/yolov5).

When running the Docker image follow this command and make sure to mount your data directory (you should put extracted data zip files in this location).

```
$ docker run -it --rm --ipc=host --net=host --gpus all -v "your data directory":/workspace/data jvizcar/braak-study:latest
```

\*Note: make the data directory open to read & write access, do this before mounting. One way to do this in Linux terminal is shown below (example taken from [here](https://stackoverflow.com/questions/1580596/how-do-i-make-git-ignore-file-mode-chmod-changes))
```
cd "data directory"
find . -type d -exec chmod a+rwx {} \; # Make folders traversable and read/write
find . -type f -exec chmod a+rw {} \;  # Make files read/write 
```

# Tutorials
We provide Jupyter notebook tutorials for tiling ROI(s), created dataset required files for training YOLOv5 models, training models, and inferencing on ROIs and WSIs. See README.md in the tutorial folder.

# Project scripts
The sctipts are numbered based on which part of the project they belong to, we include this for inspection but will not be runnable because of the need to access to the WSIs, which must be hosted in the DSA to work. 

Script descriptions:
1. Download annotated ROIs and images centered on NFT annotations. QC notebook double checks the bounding boxes and allows modification when needed.
2. Creates the ML datasets to train models specific to each annotator - trains these models.
3. Creates the MAL dataset - but does not run MAL. These scripts label the ROIs using the consensus approach, varying the number of models that must agree to give a label.
4. MAL workflow.
5. Adding additional datasets to train with MAL labeled data. Also includes the WSI inference workflow.

Results notebooks generates the results shown in the paper.


