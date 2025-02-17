# main.py
# written by Ian Robertson (ARL at UH)

'''
This program tests an already trained combined CNN (YOLO) and random forest model that classifies 
videos of breaking waves as plunging or spilling.
To train the YOLO and Random Forest models, use the function set in a different project folder (Train_YOLO_RF_2)

Steps:
    0. Check GPU configuration.
    1. Extract video clips using CNN.
    2. Label videos and separate sequence files into appropriate folders (plunging, spilling, or unclear).
    3. Apply the Random Forest model to labeled sequences.
    4. Develop statistics on the correctness of each decision.


********************************   INSTRUCTIONS   **************************************************
Note: this script was developed using a conda environment (referred to as 'yoloenv2'), operated using Microsoft Visual Studio Code. For best results, use the same configuration.
Use the yoloenv2 virtual environment to run commands on this script (unless otherwise noted)

 - This script is meant to be a guide to the functions and libraries needed to build and implement machine learning training. 
 - Programs can be called from this main.py file or called directly from the command line. Be sure to change default inputs (i.e., path names) appropriately in each script if calling direct scripts. 
 - DO NOT run this script straight through. Many sections require you to repeat function calls manually or change virtual environments so running this file all the way through will result in errors.
 - Change directory (cd) and conda activate comments are noted with **


*********************************************************************************************************************************
'''
#import packages
import torch
import sys

#import custom functions
from extract_wave_clips_ML import extract_waves_func
from copy_files import copy_files_func
from combine_sequences import combine_sequences_func
from random_forest import random_forest_func


# define inputs
# note: only home_folder and raw_videos_folder needs to be adjusted
home_folder = r".\\YOLO_RF\\Test_YOLO_RF_2" #project folder
train_home_folder = r".\\YOLO_RF\\Train_YOLO_RF_2" #project folder for training
raw_videos_folder = r'.\\YOLO_RF\\raw_videos' #edit to path of raw_videos
model_folder = train_home_folder + '\\runs\\detect\\train3\\weights\\best.pt'  #update with path of appropriate trained model
random_forest_model_folder = train_home_folder + '\\data\\random_forest_0.joblib' #choose which random forest model to use (0, 1, 2, 3, 4, 5)
#raw_videos_folder = home_folder + '\\raw_videos'  #for videos in project folder

# don't change unless altering path structure
clips_folder = home_folder + '\\videos\\'
test_folder = home_folder + '\\videos_test\\'
labels_sequence_folder = home_folder + '\\labeled_wave_sequences'
labeling_folder = home_folder + "\\clip_labeler\\clips\\"
data_folder = home_folder + '\\data'

####################################################################################################
## STEP 0: Check GPU configuration

# make sure python interpreter is set to the yoloenv2 virtual environment (main environment for this project)

# check if GPU is working
if torch.cuda.is_available():
    print("GPU is working")
else:
    sys.exit("GPU could not be initialized")

## STEP 1: Extract video clips using CNN
#use a trained YOLO CNN to label features and split up videos into wave clips
output_type = 'clips'
file = '000_002H030T33irr_camF_1'
extract_waves_func(output_type, raw_videos_folder, clips_folder, test_folder, model_folder, file)

## STEP 2: label videos and separate sequence files into appropriate folders
# copy files to clip_labeler folder for labeling
copy_files_func(clips_folder, labeling_folder)

# change directory to clip_labeler folder and change python interpreter to .\env virtual environment
# ** cd clip_labeler
# ** conda activate .\env

#run the following lines or change input in .../clip_labeler/clip_labeler.py and run directly
from clip_labeler import clip_labeler_func
home_folder = r"C:\\Users\\Public\\Documents\\Yolo_v8\\Train_YOLO_RF_2"
save_path = home_folder + '\\labeled_wave_sequences'
#change file name and rerun clip labeler function to label all clips (or run loop to label all at once)
file = '004_004H030T33irr_camF_3_processed'
path = home_folder + "\\clip_labeler\\clips\\"

#label one video at a time
todo_path = path + file #path where input videos are stored
clip_labeler_func(todo_path, home_folder, save_path)
#file = file_list[1]

#uncomment to label all videos at once
#file_list = [f.path for f in os.scandir(clips_folder) if f.is_dir()]
#for file in file_list: 
    #todo_path = path + file
    #clip_labeler_func(todo_path, home_folder, save_path)

## return to STEP 1: if more videos need to be labeled

## STEP 3: apply the Random Forest model to labeled sequences
# ** cd to the home_folder
# ** conda activate yoloenv2

# combine sequences into single txt files according to class type
combine_sequences_func(labels_sequence_folder, data_folder, 'plunging') #combined file output in labels_sequence_folder as 'plunging.txt'
combine_sequences_func(labels_sequence_folder, data_folder, 'spilling') #combined file output in labels_sequence_folder as 'spilling.txt'

#use a random forest model to classify videos based on the corresponding feature occurence
random_forest_func(data_folder, labels_sequence_folder, random_forest_model_folder, data_folder) #output results in data_folder
