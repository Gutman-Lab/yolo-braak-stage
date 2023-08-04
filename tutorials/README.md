# Tutorials
## Setup
These tutorials should be run using the Docker image "[jvizcar/braak-study:latest](https://hub.docker.com/repository/docker/jvizcar/braak-study/general)". 

```
$ docker run -it --rm --ipc=host --net=host --gpus all -v "your data directory":/workspace/data jvizcar/braak-study:latest
```

Download the "ROIs.zip" file from [here](https://drive.google.com/drive/folders/16LUMrIMdp4LlvWQk5Dp3eVQHWY472jN5?usp=sharing). Extract this file to your data directory*.

\*Note: make the data directory open to read & write access, do this before mounting. One way to do this in Linux terminal is shown below (example taken from [here](https://stackoverflow.com/questions/1580596/how-do-i-make-git-ignore-file-mode-chmod-changes))
```
cd "data directory"
find . -type d -exec chmod a+rwx {} \; # Make folders traversable and read/write
find . -type f -exec chmod a+rw {} \;  # Make files read/write 
```

Note: the Docker image contains a copy of this repository in /workspace/code (when in the container). If you plan to modify the contents of this repository and wish to keep your changes, please mount this repo when starting the Docker container: ```-v "path to repo":/workspace/code```. If doing so, also make sure to make this repo open to read & write access as shown above.

## Tutorials

**Tutorial 1: tile ROI**

**Tutorial 2: create text and yaml dataset files for YOLO training / evaluation**

**Tutorial 3: Train a model**

**Tutorial 4: Inference**

## Coming Soon
1. Tutorials in Python script format with CLIs
2. Approach to run this outside of Docker image, needs testing for dependencies