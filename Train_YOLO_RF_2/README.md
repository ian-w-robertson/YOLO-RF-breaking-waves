# YOLO-RF Model for Breaking Wave Classification

## Overview

This set of programs is designed to train a machine learning algorithm that classifies videos of breaking waves as plunging or spilling. The program was designed to analyze videos shot in a wave flume from head-on overhead angles, but with proper modification could be applied to field settings. The scripts were developed in a conda virtual environment and implemented in Microsoft Visual Studio Code, using a GPU to speed up processing. For best results, a similar setup is recommended.

This folder has a `main.py` file with detailed instructions on how to execute programs. This file is meant to be run one line at a time, as many lines need to be edited and repeated and results need to be analyzed before proceeding to the next step. Do not attempt to run these files all the way through. Instead, run 1 line at a time from the `main.py` file or edit inputs directly in the appropriate function and run or call the function directly (calling the function directly makes it easier to run multiple instances in different terminal windows). If running directly from the function file, make sure all the variables at the top of the function are correct and match what is in `main.py` if you have changed them there.


## Training (Train_YOLO_RF_2)

The first step in developing a machine learning model is to train your model. This consists of choosing a model, preprocessing data, labeling data, and feeding data into the model for training. In this case, we implement the [Ultralytics YOLO v8](https://docs.ultralytics.com/quickstart/) Convolutional Neural Network (CNN) object detection model to recognize wave features in single video frames, and an [sklearn](https://scikit-learn.org/stable/install.html) Random Forest model to understand the statistical occurence of these features over the course of the video. The goal of the training is to output accurate model files for each model that can be applied to previously unseen videos and classify breaking waves as 'plunging' or 'spilling.' To develop accurate models, we must thoughtfully label data by hand and feed this information to the model so it can learn how to label data autonomously. The following steps outline the process for training both models.

### Step 1: Extract desired frames for feature labeling using a computer vision tool

**Purpose**: In order to train the YOLO model, we need to carefully pick images that provide a representative sample of wave breaking. To speed up this process, we use a computer vision tool that detects white pixels as a proxy for wave breaking and outputs images before and during breaking that can be labeled. In this step it is important to output frames that are representative of the data that you want the ML model to perform well on. Choosing a variety of data is imperative for improving model applicability across the entire dataset.

**Instructions**: 

1. Either in the `main.py` file or the `extract_wave_clips.py` function itself, change the **file** variable to the raw video which you wish to extract frames from (make sure the path to this file is also correct and save paths are correct). 
2. Run the `extract_wave_frames_func.` An image will pop up and you will have the option to crop or to proceed with the current cropping. To use the current cropping, press `y` and then the return key. To crop, press `c` and then the return key. Follow the instructions to adjust the number of pixels cropped from the right side and the top of the video. Do the same process for adusting the region of interest (ROI). Choose coordinate (guess and check) that will outline a section of the image free from glare interference. When satisfied, press `y` and the program will start detecting features extracting wave frames.
3. Repeat this process, changing the **file** name until you have output enough frames to train the CNN. You can return to this step to label more videos after and/or begin labeling (next step) while processing more images. To speed up processing, extract waves from multiple videos at a time by calling `extract_wave_clips.py` in multiple terminal windows (highly recommended if sufficient computing power is available).

**Outputs**:

Cropped output images (of breaking wave frames) will be placed in the **save_path**. 

### Step 2: Label 5 wave features in images using the labelImg tool

**Purpose**: Our approach picks five wave features that give useful information about the breaking of the wave:

* prebreaking
* curling
* splashing
* whitewash
* crumbling

Hand-labeling these 5 "objects" in a variety of images will result in a training set for the CNN.

**Instructions**: We use the `labelImg.py` tool to draw boxes around features and label them accordingly, then save the information in a `.txt` file. To label images with this tool, you must be inside the appropriate folder. 

1. To complete this step, open a new terminal window. Change directory to the **labelImg-master** folder (`cd labelImg-master`). Then run `python labelImg.py`. A GUI will pop up on the screen. 
2. Choose the 'Open dir' icon (second one down on the left toolbar). 
3. Then, navigate to the directory where extracted photos are stored from the previous step (images_folder). Choose a folder with images in it.
4. Make sure YOLO is selected as the save format on the left toolbar (8th icon down)
5. Label features by choosing 'Create RectBox' icon (9th icon down), drawing a box around each feature, choosing the appropriate label in the pop-up box, then pressing 'Save' (7th icon down on the left toolbar).
6. Repeat step 5 until you have labeled enough images from a given video. 
7. Repeat steps 2-6 until you have labeled a roughly equal amount of images for each label across a variety of wave conditions. 

**Outputs**:

Label classes and coordinates will be saved as individual .txt files in the same directory as the image (with the same name as the image).

### Step 3: Train the YOLO model

**Purpose**: Now that we have waves labeled, we will use these labels to train a YOLO CNN object detection model to detect these features automatically. First, we will split the data into training and validation sets, and then we will feed those into the pretrained base YOLO model.

**Instructions**:

#### Split data into train and val sets:
1. Change directory to home_folder (or parent folder where env is located): `cd..`
2. Activate virtual environment: `conda activate .\env`
3. Choose 'val_percent' to be the amount of data you want to be moved to be used for your validation set. For example, if 'val_percent = 0.2" then the training set will be 80% of the full dataset and the testing set will be 20% of the full dataset.
4. Run `build_training_set_func(images_folder, train_path, val_path, val_percent)` or run directly from `build_training_set.py` (see overview).

Note - this process will split up images by wave event and not by image. Therefore, the train/test split might not be precisely what is specified in the 'val_percent' variable. Splitting up images this way avoids the potential issue of very similar issues from the same wave ending up in the train and test set which could lead to overfitting of the model.

#### Train the YOLO model:
1. First, make sure the **config.yaml** file has the correct path (should be the same as home_folder) and that class names are appropriate. Make sure train and val folders are correct and empty. 
2. Run `yolo_func(home_folder)` from main or `yolo_train.py` directly. The number of epochs is set to 300 but can be changed directly in the `yolo_train.py`function.
3. This will take a while, especially if running on a CPU. Take note of the save location when the program completes. Results should be saved in the home_folder under runs\detect\. This will include the `best.pt` model file which is the best performing CNN model, as well as training metrics and plots.
4. Optional: Use `videocnn_test.py` to check results and/or to label more photos using the CNN to include in retraining (note - this may cause overfitting). Follow the instructions in `main.py` to run this program, choosing a video to test and looking at the output. 
    * The test video is in manual_test_folder\[file]_processed and can be used to visually verify the performance of the CNN. 
    * Images and .txt files are output in manual_test_folder\[file]_cnn_frames. These can be moved to the cnn_frames folder and used in retraining of the CNN, if desired. 
5. Use the output metrics and manual testing, stop training when satisfied with the performance of the CNN. The best model is saved in the `best.pt` in the latest training folder (e.g. runs\detect\train4 if trained 4 times).

**Outputs**:
The best and last model are saved in runs/detect/train/weights as best.pt and last.pt, respectively. The YOLO model also outputs a lot of training and validation metrics in the form of plots and CSV files. See the [Ultralytics website](https://docs.ultralytics.com/guides/yolo-performance-metrics/) for more information on the metrics and how to interpret them. Click [here](https://docs.ultralytics.com/modes/) for more information on adjusting the model parameters and outputs.

### Step 4: Extract video clips using CNN.
**Purpose**: Now that we have a trained CNN model, we will generate sequences of wave features that will be used to train a random forest model. During this step we will utilize the ability of the CNN to detect distinct objects in a single frame. The program identifies features in each frame, keeping track of frame number and position of each feature. Using the time and location of each feature, the program calculates whether features in new frames belong to existing frames or are part of new wave events. It then stores the feature, the wave it corresponds to, and the frame it occured in. After extracting these data from each frame, the program can generate a video clip of each wave and a sequence of numbers (.txt file) that correspond to the features recognized by the CNN on each wave (the sequence only includes labels for that particular wave and not other waves that may be present in the same frame). These videos and sequences are then labeled and used for training the random forest model.

**Instructions**:
1. Make sure to update the 'model_folder' variable at the top of the `main.py` script to point to the best model that was trained. 
2. Choose a video file to label and set 'file' accordingly.
3. Run `extract_waves_func(output_type, raw_videos_folder, clips_folder, test_folder, model_folder, file)` from `main.py` or run `extract_wave_clips_ML.py` directly, making sure inputs are correct.
4. An image will pop up and you will have the option to crop or to proceed with the current cropping. To use the current cropping, press `y` and then the return key. To crop, press `c` and then the return key. Follow the instructions to adjust the number of pixels cropped from the right side and the top of the video. When satisfied, press `y` and the program will start detecting features.
5. To label multiple videos simultaneously, use multiple terminals. Open `extract_wave_clips_ML.py`, change the 'file' to the appropriate video, open a new cmd terminal in VScode, activate the conda environment, and run `python extract_wave_clips_ML.py` in the conda terminal. Repeat this to process multiple videos at the same time (parallel processing).

**Outputs**:

in the clips_folder:
* a cropped square video (same length as original)
* video clips (.mp4) for each unique wave event
* corresponding .txt file of all the labels associated with each video clip (one per clip)

in the test_folder:
* a cropped square video with feature labels drawn on the video (same length as orginal)
* frames (.jpgs) with features labeled on them
* .txt files associated with each labeled frame that includes label info (label, center x, center y, box width, box height, confidence, wave number, frame number)

Note: materials in the test_folder are for testing the CNN. This can be commented to save space on the hard drive if you have high confidence in CNN performance.

### Step 5: Label videos and separate sequence files into appropriate folders (plunging, spilling, or unclear)

**Purpose**: This step is where we manually label videos as 'plunging' or 'spilling' so we can train the Random Forest model. It is very important in this step to establish clear metrics for distinguishing between the two waves and to place waves that are dificult to distinguish in the 'unclear' category sos as not to 'confuse' the model. Refer to the `labeling_guidance.txt` file for more clear instructions.

**Instructions**
1. First, run `copy_files_func(clips_folder, labeling_folder)` from `main.py` or `copy_files.py` directly from function, making sure input folder (clips_folder) and (output_folder) are correct. This copies the output clips to a new folder for labeling, storing the original files in case re-labeling is required later on.
2. Then, change directory to the **clip-labeler** folder (`cd clip-labeler`). Then activate the **env** virtual environment in this folder (`conda activate .\env`). 
3. Run all the lines in the python script to set variable names, add paths, etc. and choose a video file to label clips for first (file) or run directly in `clip_labeler.py` and change variables in the header. 
4. A GUI will pop up. Choose 'plunging' or 'spilling' if the wave obviously fits into one of these categories. Choose 'unclear' if it is hard to tell (note - wave clips can always be manually dragged to the appropriate folder later so it is best to place in unclear if at all unsure). If the clip is erroneous and does not have a wave, it can be deleted entirely by choosing the null button. Fore more info, see README in the clip-labeler folder. At then end, choose yes in the pop up box to apply decisions and move files to the appropriate folder.
5. Repeat steps 3 and 4, labeling all the clips in each video folder, or uncomment the appropriate lines in main.py to label all videos at once.
6. Go into the unclear folder and watch all videos using your preferred video viewer. Use the slider in the video viewer to pause, click through frames, etc. to get a better look at each wave. Manually move waves (.mp4 and .txt files) to plunging or spilling folders if they belong.

**Outputs**

This process first copies video clips and .txt files from clips_folder to labeling_folder (step 1). Then, it moves video_clips and .txt files to plunging, spilling, or unclear subfolders in the labeled_wave_sequences folder (steps 4-5).

### Step 6: Apply the Random Forest model to labeled sequences.

**Purpose**: Now that we have all the video clips (wave sequences) labeled (organized into folders), we can train the random forest model. We will use the .txt files that include a list of all the features detected in each wave. We will tally these up and calculate the percent occurence of each of these features, and store this information as a 5x1 array for each wave sequence. Finally, we will use these vectors to train the random forest model.

**Instructions**:

1. Change directory back to the home_folder (or parent folder) to activate the main conda environment (`cd..`). Activate conda virtual environment (`conda activate .\env`)
2. Run `combine_sequences_func` from `main.py` for plunging and spilling waves. This combines all wave sequences for each category into one .txt file. 
3. Then run `random_forest_func` from `main.py`, changing the inputs appropriately. The model can be run with different random_state integer values to give different randomized models. The n_estimators parameter is set to 100 (default) but can be adjusted. The test size of 20% (.2) is standard practice but can be adjusted.

**Outputs**:
The data_folder includes a few different ouputs. The random forest model is saved as random_forest_[random_state].joblib. The classification report contains information on the model accuracy. The confusion matrix contains similar inofrmation in a visual form. The feature importance file contains information on which features were most important to the model in making classification decisions.

- - -
© Ian Robertson
