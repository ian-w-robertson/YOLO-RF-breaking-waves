# python copy_files.py
# written by Ian Robertson (ARL at UH)
# copy_files_func copies files from given subdirectories (in file_path) to a new folder (output_path)
# copy_text_files_func copies text files from file_path to output_path

#import packages
import os
import shutil

#inputs
#home_folder = r".\\Classify_YOLO_RF_2" #project folder
base_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
#test_home_folder = base_folder + r".\\Test_YOLO_RF_2" #test project folder
#file_path = test_home_folder + '\\videos\\'
file_path = r'.\\videos\\'
output_path = r".\\clip_labeler\\clips\\"

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

def copy_text_files_func(file_path, output_path):

    file_list = []
    #get a list of all subfolders of the "file_path" directory
    for root, dirs, files in os.walk(file_path):
        print(files)
        for file in files:
            if file.endswith('.txt'):
                file_list.append(os.path.join(root, file)) # get directory list of txt files 



    #copy each folder to the "output_path" location
    for file in file_list:

        #copy file to new location if the path doesn't already exist
        save_path = os.path.join(output_path, os.path.basename(file))
        if not os.path.exists(save_path):   
            shutil.copyfile(file, save_path)

        #if path already exists, give user a warning (don't copy files)
        else:
            print("The following path already exists: " + save_path + ". Contents not copied.")

#run script     
if __name__=='__main__':
    copy_files_func(file_path, output_path)