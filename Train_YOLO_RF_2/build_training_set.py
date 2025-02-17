# python build_training_set.py
#written by Ian Robertson (ARL at UH)
#moves labeled files and associated images to the train folder for the YOLO CNN


import os
import shutil
from os.path import join
from sklearn.model_selection import train_test_split


home_folder = r".\\YOLO_RF\\Train_YOLO_RF_2" #project folder
file_path = home_folder + '\\cnn_frames\\' #path where extracted images are stored for labeling (output of extract_wave_frames.py)
train_path = home_folder + '\\train' #path to place images for CNN training
val_path = home_folder + '\\val' #path to place images for CNN validation
val_percent = 0.20 #percent of total dataset to set aside for validation (determined by wave event, not by image)

def build_training_set_func(file_path, train_path, val_path, val_percent):
    folder_list = [x[0] for x in os.walk(file_path)]
    folder_list = folder_list[1:]
    master_txt_list = []
    for folder in folder_list: 
        txt_paths = [join(folder, file) for file in os.listdir(folder) if file.endswith('.txt')] #make a list of text files
        txt_paths = txt_paths[:-1]  #remove the classes.txt file
        master_txt_list = master_txt_list + txt_paths

    ### Use this approach for randomization by photo (comment following 3 lines if randomizing by wave event)
    #master_img_list = [file[:-4] + '.jpg' for file in master_txt_list] #validation set
    #split the data into training and validation sets
    #img_train, img_val, txt_train, txt_val = train_test_split(master_img_list, master_txt_list, test_size=val_percent)
        
    ### Use this approach for randomization by unique wave event (comment until 'Resume...' comment if using above approach)
    
    unique_waves = list(set([os.path.basename(file)[:-8] for file in master_txt_list])) # make a list of unique waves
    waves_train, waves_val, _, _ = train_test_split(unique_waves, unique_waves, test_size=val_percent) # split waves into train and test sets
    
    #initialize empty train and val arrays
    txt_train = []
    txt_val = []

    # extract files corresponding to each wave from the master list and add to train and val lists
    for name in waves_train:
        wave_files = [file for file in master_txt_list if name in file]
        txt_train = txt_train + wave_files

    for name in waves_val:
        wave_files = [file for file in master_txt_list if name in file]
        txt_val = txt_val + wave_files

    #make a list of images based corresponding to the txt file names
    img_train = [file[:-4] + '.jpg' for file in txt_train] #training set
    img_val = [file[:-4] + '.jpg' for file in txt_val] #validation set


    ### Resume with both approaches here:

    #copy files to train and val folders
    for file in img_train:
        shutil.copy(file, train_path)
    for file in img_val:
        shutil.copy(file, val_path)
    for file in txt_train:
        shutil.copy(file, train_path)
    for file in txt_val:
        shutil.copy(file, val_path)


# run function
if __name__=='__main__':
    build_training_set_func(file_path, train_path, val_path, val_percent)