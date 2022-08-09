# Evaluation-of-Frame-Field-Learning-using-Orthophotos
This repository attempts to produce the results

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
ENVIRONMENT SETUP
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
environment.yml 
conda activate

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
PULL
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
The following submodules must be requested to be pulled from Github and be saved in respected folders as below:
1. lydorn_utils: https://github.com/Lydorn/lydorn_utils/tree/884684a8a95bf7e279f69c908dd9e4eca524d574
2. pytorch_lydorn: https://github.com/Lydorn/pytorch_lydorn/tree/69d6b0a3ace94cdb3204708b00862b30391f727c

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
1. inria_dataset_polygonized.unet_resnet101_pretrained
2. private_dataset_polygonized.unet_resnet101_pretrained

The parameters can be changed accordingly depending on the experiment one wants to perform. The other config files are all connected to the above two main files. The following parameters can be changed in order to perform the experiments explained in the paper.
1. 
2.
3.
4.
5.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
TRAINING
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
python main.py --config configs/<name_of_config> --gpus 1
python main.py --config private_dataset_polygonized.unet_resnet101_pretrained --gpus 1

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
PRE-TRAINED MODEL
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
The zip folder of the pre-trained models can be downloaded from here: 

1. Upload the zipped folder onto jupyter notebook.
2. Unzip it using the following:

  # Fixes the zip file in case it is corrupted
  !zip -FF inria_dataset_polygonized_unet_resnet101_pretrained_2022_05_10_10_05_30.zip -O private_dataset_polygonized_unet_resnet101_pretrained_2022_05_10_10_05_30.fixed.zip 

  # Unzips the file and saves it in the same location
  !unzip ~/Polygonization-by-Frame-Field-Learning/frame_field_learning/runs/private_dataset_polygonized_unet_resnet101_pretrained_2022_05_10_10_05_30.fixed.zip

3. Rename the file separating the name and datetime stamp with a '|' like so: 
  private_dataset_polygonized.unet_resnet101_pretrained | 2022_05_10_10_05_30
  
 This can be used as run_name during inference without the datetime stamp.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
INFERENCE
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
The inference can be run on any image using the pre-trained models provided above or using a new run.

python main.py --in_filepath <path_to_image> --run_name <name_of_run>

python main.py --in_filepath /home/jovyan/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/test/images/bad_bodenteich3.tif --run_name private_dataset_polygonized_unet_resnet101_pretrained

Saves the predicted shapefiles in the same folder.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
METRICS
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
The IoU and tangent angle metrics can be evaluated for each image as below using the image and ground truth shapefile:

1. change directory to scripts: cd scripts
2. python eval_shapefiles.py --im_filepath <path_to_image> --gt_filepath <path_to_gt_shapefile> --pred_filepath <path_to_predicted_shapefiles>
  python eval_shapefiles.py --im_filepath ~/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/test/images/bad_bodenteich3.tif --gt_filepath ~/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/test/shp/bad_bodenteich3.shp --pred_filepath ~/Polygonization-by-Frame-Field-Learning/data/PrivateDataset/raw/test/images/poly_shapefile.simple.tol_1/bad_bodenteich3.shp
3. Run check.py to get the average values of IoU and tangent angle for each image.




![image](https://user-images.githubusercontent.com/60517504/183388572-f455dc82-647d-475f-aa8e-8eb0aed09db1.png)

