# YOLO-RF Model for Breaking Wave Classification

## Overview

This set of programs is designed to train, test, and implement a machine learning algorithm that classifies videos of breaking waves as plunging or spilling. The program was designed to analyze videos shot in a wave flume from head-on overhead angles, but with proper modification could be applied to field settings. The scripts were developed in a conda virtual environment and implemented in Microsoft Visual Studio Code, using a GPU to speed up processing. For best results, a similar setup is recommended.

This project is organized into 3 project folders called **Train_YOLO_RF_2**, **Test_YOLO_RF_2**, and **Classify_YOLO_RF_2**. Each of these folders has a set of subfolders and scripts associated with them. Some programs have duplicate names between folders but are customized to run within each folder. Programs are all self-contained with options to pull data from other folders. Paths can be decided and customized at the top of each script. Each project folder has a `main.py` file with detailed instructions on how to execute programs. This file is meant to be run one line at a time, as many lines need to be edited and repeated and results need to be analyzed before proceeding to the next step. Do not attempt to run these files all the way through. Instead, run 1 line at a time from the `main.py` file or edit inputs directly in the appropriate function and run or call the function directly (calling the function directly makes it easier to run multiple instances in different terminal windows). If running directly from the function file, make sure all the variables at the top of the function are correct and match what is in `main.py` if you have changed them there.

Examples files for each of the files (including intermediary outputs) are included under the **examples** folder, with the same file organization as the 3 main project folders.

## Installation

### Basic Python and Conda Installation

This project uses python software and conda as a package manager. For best results, run in the Microsoft Visual Studio Code (VScode) code editor.

Install these programs by following the instructions in the following links:
* [python](https://www.python.org/)
* [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) (note - check the first 2 boxes on the installer to ensure conda is accessible from VS code)
* [Visual Studio Code](https://code.visualstudio.com/)


After installing these programs, you will need to create a virtual conda environment for this project. To setup the appropriate virtual environment on a new machine/account, you can either clone the original environment used for this project, create from an `environment.yaml` file, or create it again from scratch. All three options are outlined below

### Option 1: Clone the virtual environment from env folder (preferred if files are available)
The env virtual environment can be cloned with the following commands:
* `conda create -p .\env --clone myenv` (installs virtual environment in current project folder where 'myenv' is the existing environment)
* `conda create --name myclone --clone myenv` (creates a new virtual environment called 'myclone' in the conda directory where 'myenv' is the existing environment)

### Option 2: Create virtual environment from `environment.yml` file
A full list of package dependencies and versions are in the `environment.yml` file. 
The environment can be replicated from the `environment.yml` file with the following commands:
* `conda env create -p .\env -f environment.yml` (installs virtual environment in current project folder)
* `conda env create --name myenv -f environment.yml` (creates a new environment called 'myenv' in the conda directory)

Note: This method may cause package version issues due to the order programs are installed. 

### Option 3: Create a new virtual environment from scratch

Note: Some of the dependencies might need to be installed manually due to version compatability issues between programs.

* create a python 3.11.5 conda virtual environment
    * `conda env create -p .\env python=3.11.5` (installs virtual environment in current project folder)
    * `conda env create --name myenv python=3.11.5` (creates a new environment called 'myenv' in the conda directory)

For CPU setup:
* install pytorch with this command: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

For GPU setup with CUDA 12.1:
* install CUDA with this command: `conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit`
* install pytorch with this command: `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

* install ultralytics: `pip install ultralytics` 

* install Sklearn: `pip install -U scikit-learn`

    if that causes issues, create a separate virtual environment for sklearn. Note, you will have to deactivate other environments and activate the sklearn environment to use sklearn in this case.
    * `conda create -n sklearn-env -c conda-forge scikit-learn`
    * `conda activate sklearn-env`

if not already installed, install opencv: `pip3 install opencv-python`

* install labelImg: copy labelImg-master folder into **Train_YOLO_RF_2** and **Test_YOLO_RF_2** or follow instructions in the documentation below

### Additional Documentation:
* [Conda installation](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)
* [PyTorch installation](https://pytorch.org/get-started/locally/)
* [Ultralytics YOLO v8](https://docs.ultralytics.com/quickstart/)
* [Sklearn installation](https://scikit-learn.org/stable/install.html)
* [NVIDIA CUDA download](https://developer.nvidia.com/cuda-downloads)
* [labelImg documentation](https://pypi.org/project/labelImg/)
* [Open-cv documentation](https://opencv.org/get-started/)

### Computer Specs used in development and testing:
-   OS: Microsoft Windows 10 Enterprise 10.0.19045
-   Manufacturer: Gigabyte Technology Co., Ltd.
-   Model: TRX40 AORUS PRO WIFI
-   System type: x64-based PC
-   Processor: AMD Ryzen Threadripper 3960X 24-Core Processor 3.80 Ghz
-	Installed Memory: (RAM): 256 GB
-	System type: 64-bit Operating System, x64-based processor
-	Storage: 2TB SSD + 5TB external drive
-	GPU: 2x NVIDIA GeForce RTX 3090
-	NVIDIA CUDA and CUDNN toolkits installed

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

## Testing
The next step in developing a machine learning model is to test your model. While we have already verified the accuracy of both our models to a certain extent using the validation set in the YOLO model and the test set in the Random Forest model, our accuracy is somewhat biased because the data used is not completely unseen by both models. To get a better indicator of model performance, we test our model on new unseen data. This process is quite similar to the training step, except we don't need to train the YOLO CNN or the Random Forest model because now we will use the models generated during training. We still need to label save clips as plunging or spilling to assess model accuracy. Note - we are testing the accuracy of the entire combined CNN-random forest model (in the final output) so we will not be labeling and checking accuracy of CNN feature labeling in this step.

For this step, we will navigate to the **Test_YOLO_RF_2** project folder. Some of the programs here have the same name as **Train_YOLO_RF_2** but are slightly different. To ensure you are using the correct version, make sure you quit your **Train_YOLO_RF_2** terminal and change directory into the **Test_YOLO_RF_2** folder before running programs.

### Step 1: Extract video clips using CNN.

**Purpose**: Now that we have a YOLO CNN model from training above, we can start directly with this step. This step is identical to step 4 from Training. 

**Instructions**:
1. Activate the `env` virtual environment from the parent folder. Make sure to update the 'model_folder' variable at the top of the `main.py` script to point to the best model that was trained (in training folder). 
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

### Step 2: Label videos and separate sequence files into appropriate folders (plunging, spilling, or unclear)

**Purpose**: Now we manually label videos as 'plunging' or 'spilling' so we can test the accuracy of our combined CNN and Random Forest model. As a reminder, it is very important in this step to establish clear metrics for distinguishing between the two waves and to place waves that are dificult to distinguish in the 'unclear' category sos as not to 'confuse' the model. Refer to the `labeling_guidance.txt` file for more clear instructions. This step is identical to step 5 from Training.

**Instructions**
1. First, run `copy_files_func(clips_folder, labeling_folder)` from `main.py` or `copy_files.py` directly from function, making sure input folder (clips_folder) and (output_folder) are correct. This copies the output clips to a new folder for labeling, storing the original files in case re-labeling is required later on.
2. Then, change directory to the **clip-labeler** folder (`cd clip-labeler`). Then activate the **env** virtual environment in this folder (`conda activate .\env`). 
3. Run all the lines in the python script to set variable names, add paths, etc. and choose a video file to label clips for first (file) or run directly in `clip_labeler.py` and change variables in the header. 
4. A GUI will pop up. Choose 'plunging' or 'spilling' if the wave obviously fits into one of these categories. Choose 'unclear' if it is hard to tell (note - wave clips can always be manually dragged to the appropriate folder later so it is best to place in unclear if at all unsure). If the clip is erroneous and does not have a wave, it can be deleted entirely by choosing the null button. Fore more info, see README in the clip-labeler folder. At then end, choose yes in the pop up box to apply decisions and move files to the appropriate folder.
5. Repeat steps 3 and 4, labeling all the clips in each video folder, or uncomment the appropriate lines in main.py to label all videos at once.
6. Go into the unclear folder and watch all videos using your preferred video viewer. Use the slider in the video viewer to pause, click through frames, etc. to get a better look at each wave. Manually move waves (.mp4 and .txt files) to plunging or spilling folders if they belong.

**Outputs**:

This process first copies video clips and .txt files from clips_folder to labeling_folder (step 1). Then, it moves video_clips and .txt files to plunging, spilling, or unclear subfolders in the labeled_wave_sequences folder (steps 4-5).

### Step 3: Apply the Random Forest model to labeled sequences.

**Purpose**: Now that we have all the video clips (wave sequences) labeled (organized into folders), we can test the random forest model generated in Training. We will use the .txt files that include a list of all the features detected in each wave. We will tally these up and calculate the percent occurence of each of these features, and store this information as a 5x1 array for each wave sequence. Finally, we will feed these vectors to the trained model, get results, and compare to the manual labeling to assess the performance of the model.

**Instructions**:

1. Change directory back to the home_folder (or parent folder) to activate the main conda environment (`cd..`). Activate conda virtual environment (`conda activate .\env`)
2. Run `combine_sequences_func` from `main.py` for plunging and spilling waves. This combines all wave sequences for each category into one .txt file. 
3. Then run `random_forest_func` from `main.py`, changing the inputs appropriately. Choose whichever model you would like to use from the model files you generated in Training.

**Outputs**:
The data_folder includes the same ouputs as the random forest training, except for the model itself. The classification report contains information on the model accuracy. The confusion matrix contains similar inofrmation in a visual form. The feature importance file contains information on which features were most important to the model in making classification decisions. The spilling and plunging .npy files contain the label occurence percentages (5x1 array) for each wave sequence.

Note - if not satisfied with results, more training is required (return to Training steps). Try adding more labeled features to the CNN training or labeled video clips to the Random Forest training. Experiment with model parameters and training data size until satisfied with accuracy.

## Classifying
Once the model has been trained and tested and we have enough confidence to use the model for it's intended purpose (classifying breaking waves), we can implement it. This Classifying step implements a trained YOLO CNN model and Random Forest model in tandem to make a list of each video clip and it's classification. Then, if files are named according to wave condition, date, etc., breaker types can be compared according to file name.

For this step, we will navigate to the **Classify_YOLO_RF_2** project folder. Some of the programs here have the same name as **Train_YOLO_RF_2** and **Test_YOLO_RF_2** but are slightly different. To ensure you are using the correct version, make sure you quit your other terminals and change directory into the **Classify_YOLO_RF_2** folder before running programs.

### Steps:

### Step 1: Extract video clips using CNN.
**Purpose**: Now that we have a trusted YOLO CNN model from Training and Testing above, we can start directly with this step. This step is identical to step 4 from Training and step 1 from Testing. 

**Instructions**:
1. Activate the `env` virtual environment from the parent folder. Make sure to update the 'model_folder' variable at the top of the `main.py` script to point to the best model that was trained (in training folder). 
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

### Step 2: Apply the Random Forest model to labeled sequences.

**Purpose**: Now that we have video clips ready to be classified, we can apply the random forest model generated in Training. We will use the .txt files that include a list of all the features detected in each wave. We will tally these up and calculate the percent occurence of each of these features, and store this information as a 5x1 array for each wave sequence. Finally, we will feed these vectors to the trained model and get classifications for each wave.

**Instructions**:

1. Run `combine_sequences_func` from `main.py`. This combines all wave sequences for each category into one `general.txt` file. 
2. Then run `random_forest_func` from `main.py`, changing the inputs appropriately. Choose whichever model you would like to use from the model files you generated in Training.
3. To store results from all the different wave test conditions and set up data for Step 3, do the following:

    * rename each data folder based on the wave conditions, (e.g., **data_000_002H030T33irr**) 
    * move the wave_sequences folder into this new folder
    * if it doesn't already exist, create a new folder called **results_by_condition** and drag the renamed data folder into this new folder
    * make two new folders (**data** and **wave_sequences**) to replace the ones that were just moved
    * repeat for all wave conditions tested

**Outputs**:
The data_folder includes three outputs. The `general.txt` file contains all the wave feature sequences generated by the CNN. The `general_labels.npy` file contains the label occurence percentages (5x1 array) for each wave sequence. The `results.txt` file contains the classification results, providing the name of each video, a binary number classification (0 = plunging, 1 = spilling), and a text classification (plunging or spilling) for each wave sequence input.

### Step 3: Obtain statistics on breaker types by file name (wave conditions)

**Purpose**: Now that we have a classification for each breaking wave, we can compare breaking occurences based on the input conditions for each wave. In the lab (wave flume), this includes parameters like significant wave height, dominant wave period, and layout of breakwaters. Splitting up waves into categories based on their conditions, we can compare breaking statitics between wave/bathymetry conditions in a simple bar chart.

This program is set up to compare waves that had the same wave conditions (significant wave height and dominant period) but different breakwater layouts. It will require some editing to plot other types of comparisons.

**Instructions**:

1. Set test_case to be a string of characters present in the files that you want to compare (wave conditions).
2. Run `analyze_breaking_func` to save results.
3. Repeat steps 1 and 2, changing test_case each time to save results for all desired wave conditions.

**Outputs**:
The `analyze_breaking_func` outputs 4 files in the **results_by_condition** folder. The first is [test_case]_breaker_stats.csv, which contains the counts of plunging and spilling waves for each layout. The second is [test_case]_breaker_stats_plot.png, which contains a bar chart of these numbers. The third is [test_case]_breaker_stats__norm_plot.png, which contains a normalized version of the previous plot (percent occurence/total occurence). The final file is [test_case]_combined_results.csv which includes clip name, binary value, and label for each wave in the test_case conditions. This is essentially just a subset of `results.txt` from step 2.




- - -
© Ian Robertson
