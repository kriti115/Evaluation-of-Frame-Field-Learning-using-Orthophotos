# Evaluation-of-Frame-Field-Learning-using-Orthophotos
This repository attempts to produce the results  

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
DATASET
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. INRIA Aerial Dataset can be downloaded from the link below:


2. Large Scale Real World Dataset can be downloaded from the link below:
https://tubcloud.tu-berlin.de/s/M6PobTMpaX6q7Ap

Train on raw images from scratch:
The raw images are cropped into 725 x 725 patches and stored in a folder called processed, which is used for training. This is the course to take in case you want to train using new dataset.

Processed Folder:
The processed images have been provided which can be directly used for training. This is the course to take if we want to train on the existing dataset, either the INRIA or the large scale real world dataset.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
CONFIGURATIONS
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
There is one main configuration file for each of the datasets, namely:
1.
2.

The parameters can be changed accordingly depending on the experiment one wants to perform. The other config files are all connected to the above two main files.
1. 
2.
3.
4.
5.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
TRAINING
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
python main.py --config configs/<name_of_config> 


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
PRE-TRAINED MODEL
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. Upload the zipped folder onto jupyter notebook.
2. Unzip it using the unzip.py script.
3. Rename the file separating the name and date stamp with a '|' like so: 


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
INFERENCE
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
The inference can be run on any image using the pre-trained models provided above or using a new run.

python main.py --run_name <name_of_run> --in_filepath <path_to_image>

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
METRICS
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
The IoU and tangent angle metrics can be evaluated for each image as below:

python 


![image](https://user-images.githubusercontent.com/60517504/183388572-f455dc82-647d-475f-aa8e-8eb0aed09db1.png)

