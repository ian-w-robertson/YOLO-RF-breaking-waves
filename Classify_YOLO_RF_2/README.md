# YOLO-RF Model for Breaking Wave Classification (Classify)

## Overview

This program is designed to implement a machine learning algorithm that classifies videos of breaking waves as plunging or spilling. The program was designed to analyze videos shot in a wave flume from head-on overhead angles, but with proper modification could be applied to field settings. The scripts were developed in a conda virtual environment and implemented in Microsoft Visual Studio Code, using a GPU to speed up processing. For best results, a similar setup is recommended.

Programs are all self-contained with options to pull data from other folders. Paths can be decided and customized at the top of each script. Each project folder has a `main.py` file with detailed instructions on how to execute programs. This file is meant to be run one line at a time, as many lines need to be edited and repeated and results need to be analyzed before proceeding to the next step. Do not attempt to run these files all the way through. Instead, run 1 line at a time from the `main.py` file or edit inputs directly in the appropriate function and run or call the function directly (calling the function directly makes it easier to run multiple instances in different terminal windows). If running directly from the function file, make sure all the variables at the top of the function are correct and match what is in `main.py` if you have changed them there.

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
