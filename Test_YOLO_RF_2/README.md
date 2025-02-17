# YOLO-RF Model for Breaking Wave Classification

## Overview

This program is designed to test a YOLO-RF machine learning algorithm that classifies videos of breaking waves as plunging or spilling. The program was designed to analyze videos shot in a wave flume from head-on overhead angles, but with proper modification could be applied to field settings. The scripts were developed in a conda virtual environment and implemented in Microsoft Visual Studio Code, using a GPU to speed up processing. For best results, a similar setup is recommended.

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

- - -
© Ian Robertson
