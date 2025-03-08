# main.py
# written by Ian Robertson (ARL at UH)

'''
This program implements an already trained combined CNN (YOLO) and random forest model, labeling wave clips as plunging or spilling.
To train the YOLO and Random Forest models, use the function set in \Train_YOLO_RF_2.
To test the YOLO - Random Forest model on pre-labeled clips, use the function set in \Test_YOLO_RF_2.

Steps:
    0. Check GPU configuration.
    1. Extract video clips using CNN.
    2. Apply the Random Forest model to labeled sequences and output a list with file name and breaker type.
    3. Obtain statistics on breaker types by file name (test case number)


********************************   INSTRUCTIONS   **************************************************
Note: this script was developed using a conda virtual environment (in the 'env' folder), operated using Microsoft Visual Studio Code. For best results, use the same configuration.
Use the main 'env' virtual environment (in the outermost directory) to run commands on this script (unless otherwise noted)

 - This script is meant to be a guide to the functions and libraries needed to build and implement machine learning training. 
 - Programs can be called from this main.py file or called directly from the command line. Be sure to change default inputs (i.e., path names) appropriately in each script if calling direct scripts. 
 - This script can be run straight through for one video input. Multiple video inputs require manual input of video file name.
 - Change directory (cd) and conda activate comments are noted with **


*********************************************************************************************************************************
'''
#import packages
import torch
import sys
from functools import reduce
from os.path import join
import os
import numpy as np

#import custom functions (use "python test_imports.py" from command line if this doesn't work)
from extract_wave_clips_ML import extract_waves_func
from copy_files import copy_text_files_func
from copy_files import copy_files_func
from combine_sequences import combine_sequences_func
from random_forest import random_forest_func
from analyze_breaking import analyze_breaking_func
from analyze_breaking import analyze_breaking_func_single

# define inputs
#home_folder = r".\\Classify_YOLO_RF_2" #project folder
base_folder = os.path.abspath(os.path.join(os.getcwd(), "..")) #one level up
train_home_folder = base_folder + r"\\Train_YOLO_RF_2" #project folder for training
test_home_folder = base_folder + r"\\Test_YOLO_RF_2"
#model_folder = train_home_folder + '\\runs\\detect\\train3\\weights\\best.pt' #choose YOLO model to use, use model in train folder if you have trained your own model
model_folder = r'.\\models\\YOLO_train3_best.pt' #choose YOLO model to use, use model in current folder if you have not trained your own model
raw_videos_folder = base_folder + r'\\raw_videos' #for videos on external drive (saves space on local drive)
#random_forest_model_folder = train_home_folder + '\\data\\random_forest_0.joblib' #choose which random forest model to use (0, 1, 2, 3, 4, 5), use file in train folder if you have trained your own model
random_forest_model_folder = r'.\\models\\random_forest_2.joblib'

# don't change unless altering path structure
clips_folder = r'.\\videos\\'  # if doing step 1, use this path
test_folder = r'.\\videos_test\\' # if doing step 1, use this path
#clips_folder = test_home_folder + '\\videos\\' # if step 1 already completed in testing, use test path to get videos
#test_folder = test_home_folder + '\\videos_test\\' # if step 1 already completed in testing, use test path to get videos 
results_folder = r'.\\results_by_condition\\'

####################################################################################################

# make sure python interpreter is set to the 'env' virtual environment in the project folder (main environment for this project)

## STEP 0: Check GPU configuration

# check if GPU is working
if torch.cuda.is_available():
    print("GPU is working")
else:
    sys.exit("GPU could not be initialized")

## STEP 1: Extract video clips using CNN

#choose file
file = '002H30T33irr_camF_3'

#create directories
data_folder = results_folder + file
if not os.path.exists(data_folder): 
    os.mkdir(data_folder)
labels_sequence_folder = data_folder + '\\wave_sequences'
if not os.path.exists(labels_sequence_folder): 
    os.mkdir(labels_sequence_folder)

#use a trained YOLO CNN to label features and split up videos into wave clips
output_type = 'clips'
extract_waves_func(output_type, raw_videos_folder, clips_folder, test_folder, model_folder, file)

# copy wave sequences to the labels_sequence_folder (or drag manually)
file_path = clips_folder + file +'_processed'# copy specific files
#file_path = clips_folder  #copy all files
copy_text_files_func(file_path, labels_sequence_folder) #can also be done manually

## STEP 2: apply the Random Forest model to labeled sequences

# combine sequences into single txt file (all in the same file since class is unknown)
combine_sequences_func(labels_sequence_folder, data_folder, 'general') #combined file output in labels_sequence_folder as 'general.txt'

#use a random forest model to classify videos based on the corresponding feature occurence (results in results.txt file)
random_forest_func(data_folder, labels_sequence_folder, random_forest_model_folder, data_folder) #output results in data_folder

## STEP 3: plot and save results
test_case = '002H30T33irr' #wave conditions you would like to analyze
#analyze_breaking_func(results_folder, test_case) #for multiple test cases
analyze_breaking_func_single(results_folder, test_case) #for one test case

#plots and CSV files are output in results_folder, labeled by test_case 