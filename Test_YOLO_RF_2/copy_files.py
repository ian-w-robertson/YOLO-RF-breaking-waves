# python copy_files.py
# written by Ian Robertson (ARL at UH)
# copies files from given subdirectories (in file_path) to a new folder (output_path)

#import packages
import os
import shutil

#inputs
home_folder = r".\\Test_YOLO_RF_2" #project folder
test_home_folder = r".\\Test_YOLO_RF_2" #test project folder
file_path = test_home_folder + '\\videos\\'
output_path = home_folder + "\\clip_labeler\\clips\\"

def copy_files_func(file_path, output_path):

    #get a list of all subfolders of the "file_path" directory
    folder_list = [x[0] for x in os.walk(file_path)] 
    folder_list = folder_list[1:]

    #copy each folder to the "output_path" location
    for folder in folder_list:

        #copy file to new location if the path doesn't already exist
        save_path = output_path + os.path.basename(folder)
        if not os.path.exists(save_path):   
            shutil.copytree(folder, save_path)

        #if path already exists, give user a warning (don't copy files)
        else:
            print("The following path already exists: " + save_path + ". Contents not copied.")

#run script     
if __name__=='__main__':
    copy_files_func(file_path, output_path)