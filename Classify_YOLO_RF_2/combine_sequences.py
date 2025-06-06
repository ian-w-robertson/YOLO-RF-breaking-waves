#python combine_sequences.py
# written by Ian Robertson (ARL at UH)

#combines individual sequences stored in separate .txt files into one master .txt file

#import functions
import os

#inputs
#home_folder = r".\\Classify_YOLO_RF_2" #project folder
labels_sequence_folder = r'.\\wave_sequences' #folder where individual sequences are stored
output_folder = r'.\\data' #folder to output master .txt file
class_name = 'general' 

def combine_sequences_func(labels_sequence_folder, output_folder, class_name):

    #get file names for all the sequences
    data_dir = labels_sequence_folder
    for root, dirs, files in os.walk(data_dir):        
        file_names = [file for file in files if file.endswith('.txt')]
        path_names = [os.path.join(root,file_name) for file_name in file_names]

    #prepare the output file
    output_path = os.path.join(output_folder, class_name + '.txt')
    if os.path.exists(output_path):
        os.remove(output_path)

    #open the output .txt file and write each individual sequence to that file
    with open(output_path, 'w') as outfile:
        for fname in path_names:
            with open(fname) as infile:
                first_line = infile.readline()
                outfile.write(first_line)
                outfile.write('\n')

# run script
if __name__=='__main__':
    combine_sequences_func(labels_sequence_folder, output_folder, class_name)