# main.py
# written by Ian Robertson

'''
This program provides the commands to build and train a combined CNN (YOLO) and random forest model that classifies 
videos of breaking waves as plunging or spilling.

Steps:
    0. Check GPU configuration.
    1. Extract desired frames for feature labeling using a computer vision tool.
    2. Label 5 wave features in images using the labelImg tool.
    3. Train the YOLO model.
    4. Extract video clips using CNN.
    5. Label videos and separate sequence files into appropriate folders (plunging, spilling, or unclear).
    6. Apply the Random Forest model to labeled sequences.


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
from sklearn.model_selection import train_test_split

#import custom functions
from extract_wave_frames import extract_wave_frames_func
from build_training_set import build_training_set_func
from yolo_train import yolo_func
from extract_wave_clips_ML import extract_waves_func
from copy_files import copy_files_func
from combine_sequences import combine_sequences_func
from random_forest import random_forest_func


# define inputs
# note: only home_folder and raw_videos_folder needs to be adjusted
home_folder = r".\\YOLO_RF\\Train_YOLO_RF_2" #project folder
#raw_videos_folder = r'E:\\2023-02\\ML videos\\camF\\training' #for videos on external drive (saves space on local drive)
raw_videos_folder = '.\\YOLO_RF\\raw_videos'  #for videos in project folder

# don't change unless altering path structure
images_folder = home_folder + '\\cnn_frames\\'
clips_folder = home_folder + '\\videos\\'
test_folder = home_folder + '\\videos_test\\'
model_folder = home_folder + '\\runs\\detect\\train3\\weights\\best.pt'  #pick folder of desired model after training the YOLO model
labels_sequence_folder = home_folder + '\\labeled_wave_sequences'
train_path = home_folder + '\\train'
val_path = home_folder + '\\val'
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

#Note: STEP 1 and STEP 2 can be run in tandem in different terminals to speed up processing

## STEP 1: Extract desired frames for feature labeling using a computer vision tool.
output_type = 'frames'
#repeat the following 2 lines (changing vid_file each time) until you have extracted frames from all desired videos
vid_file = '000_002H030T33irr_camF_1' #choose raw video file you want to extract frames from
extract_wave_frames_func(output_type, raw_videos_folder, images_folder, vid_file) 
# or run extract_wave_frames.py directly from terminal, changing the 'file' variable each time in that script.
# this allows you to run multiple cmd prompts at the same time to speed up processing 
# copy and paste this into terminal: python extract_wave_frames.py

## STEP 2: Label 5 wave features in images using the labelImg tool.
## label the dataset using labelImg program
# ** change directory: cd labelImg-master
# if still in conda environment, deactivate conda environment: conda deactivate
# run labelImg: python labelImg.py
# use the labelImg GUI to label images (automatically saves labels to text file)
    #a. choose open_dir on the top right
    #b. navigate to ./cnn_frames folder, then pick a folder with images 
    #c. label a good variety and roughly even amount of features in each category (prebreaking, curling, splashing, whitewash, crumbling) 
    #d. repeat a-c until you have labeled enough images across a variety of videos (at least 1,000 total images if possible)
    
## STEP 3: Train the YOLO model
# ** cd to home_folder
# ** conda activate yoloenv2

#split data set into train and val and move files to train and val folders
val_percent = 0.25 #percentage (between 0.0 and 1.00) of data to put in the validation set
build_training_set_func(images_folder, train_path, val_path, val_percent)

#run YOLO training
#make sure config.yaml file has the correct path (should be the same as home_folder) and that class names are appropriate. Make sure train and val folders are correct and empty. 
yolo_func(home_folder)

#optional: use the trained CNN to label more images
#model_folder = home_folder + '\\runs\\detect\\train3\\weights\\best.pt' #make sure to update model folder each time if repeating step 3
#from videocnn_test import videocnn_func
#manual_test_folder = home_folder + '\\manual_test_videos\\'
#file = '004_008H040T36reg_camF_1 - Trim' #input raw video file
#videocnn(manual_test_folder, model_folder, file)
#repeat step 3, manually labeling and/or using the ML model to label until model performs well (unsure of accuracy requirement for this stage)

## STEP 4: Extract video clips using CNN
#after model is trained, use the best model to label features and split up videos into wave clips
output_type = 'clips'
file = '000_002H030T33irr_camF_1'
#file = '000_010H030T33irr_HT_camL_1 - Trim'
extract_waves_func(output_type, raw_videos_folder, clips_folder, test_folder, model_folder, file)


## STEP 5: label videos and separate sequence files into appropriate folders
# copy files to clip_labeler folder for labeling
copy_files_func(clips_folder, labeling_folder)

# change directory to clip_labeler folder and change python interpreter to .\env virtual environment
# ** cd clip_lableler
# ** conda activate .\env

import os, ntpath
from clip_labeler import clip_labeler_func
home_folder = r".\\YOLO_RF\\Train_YOLO_RF_2"
save_path = home_folder + '\\labeled_wave_sequences'
#change file name and rerun clip labeler function to label all clips (or run loop to label all at once)
file = '004_004H030T33irr_camF_3_processed'
path = home_folder + "\\clip_labeler\\clips\\"
todo_path = path + file #path where input videos are stored
clip_labeler_func(todo_path, home_folder, file, save_path)
#file = file_list[1]
#uncomment to label all videos at once
#file_list = [f.path for f in os.scandir(clips_folder) if f.is_dir()]
#for file in file_list: 
    #file = '000_002H030T33irr_camF_1_processed'
    #clip_labeler_func(file, labels_sequence_folder)
## return to STEP 4: if more videos need to be labeled

## STEP 6: apply the Random Forest model to labeled sequences
# ** cd to the home_folder
# ** conda activate yoloenv2

# combine sequences into single txt files according to class type
combine_sequences_func(labels_sequence_folder, data_folder, 'plunging') #combined file output in labels_sequence_folder as 'plunging.txt'
combine_sequences_func(labels_sequence_folder, data_folder, 'spilling') #combined file output in labels_sequence_folder as 'spilling.txt'

#use a random forest model to classify videos based on the corresponding feature occurence
random_forest_func(data_folder, data_folder, test_size=0.2, n_estimators=100, random_state=5) #output results in data_folder

#to test the final combined CNN-Random Forest model on new video data, use the modified function set in a different project folder (Test_YOLO_RF_2)