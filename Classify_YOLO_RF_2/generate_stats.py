# Calculates percent occurence of each label in a list of label sequences (.txt file)
# written by Ian Robertson

from os.path import join
import numpy as np

#home_folder = r".\\YOLO_RF\\Classify_YOLO_RF_2"
labels_sequence_folder = r'.\\data' # path where 'plunging.txt' and 'spilling.txt' are stored
class_name = 'plunging' # 'plunging' or 'spilling'

def generate_stats_func(labels_sequence_folder, class_name):
    sequences_list = join(labels_sequence_folder, class_name + '.txt') #list of sequences
    
    #import list into python
    with open(sequences_list) as f:
        sequences = [line.rstrip() for line in f] 

    #get counts and percents of specific frame labels for each wave event (numbers 1-5)
    class_count = []
    class_percent = []
    k = 0
    for sequence in sequences:
        total_label_count = 0
        label_count = [0]*5
        label_percent = [0]*5
        for i in range(0,5):
            count = sequence.count(str(i+1))
            label_count[i]=count
            total_label_count = total_label_count + count
        label_percent = [count/total_label_count for count in label_count]
        class_count.append(label_count)
        class_percent.append(label_percent)
        k+=1
        
    #save stats as a .npy file and return stats list as a function output
    np.save(labels_sequence_folder + '\\' + class_name + '_labels', class_percent)
    return class_percent

if __name__=='__main__':
    generate_stats_func(labels_sequence_folder, class_name)