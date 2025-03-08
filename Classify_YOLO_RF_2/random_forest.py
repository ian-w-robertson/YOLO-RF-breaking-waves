# This function takes a .txt file list of feature occurences across a set of wave clips and applies a trained random forest model to
# identify waves as plunging or spilling

#adapted from https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/
#edited by Ian Robertson

# import required libraries 
import pandas as pd 
import os
import joblib
import numpy as np
from generate_stats import generate_stats_func


# function inputs
#train_home_folder = r".\\Train_YOLO_RF_2"
model_folder = r'.\\models\\random_forest_0.joblib' #change to appropriate random forest model
data_folder = r'.\\data'
labels_sequence_folder = r'.\\wave_sequences' 
save_path = data_folder #where output data is saved

def random_forest_func(data_folder, labels_sequence_folder, model_folder, save_path):

    #generate feature label statistics based on feature label lists
    data = generate_stats_func(data_folder, 'general') #full list of feature stats
    #data = np.load('D:\\Yolo_v8\\Classify_YOLO_RF_2\\results_by_condition\\data_000_002H030T33irr\\general_labels.npy')
    #make a list of video clip names associated with each category
    video_names = [txt.split('.')[0] for txt in os.listdir(labels_sequence_folder) if txt.endswith('.txt')] #get names of videos

    #load the random forest model
    clf = joblib.load(model_folder)
    class_names = ['plunging', 'spilling']

    # perform predictions on the test dataset 
    y_pred = clf.predict(data) 
    pred_labels = [class_names[int(i)] for i in y_pred]
    for i in range(0, len(data)):
        if data[i][0] == 9999:
            y_pred[i] = 9999
            pred_labels[i]='null'

    #make a data frame of results and labels
    data_frame = {'name': video_names,
            'value': y_pred, 'label': pred_labels}
    df = pd.DataFrame(data_frame)
    #print(df)

    #save results to text file    
    results_file = save_path + '\\'+ 'results.txt'
    if os.path.exists(results_file):
        os.remove(results_file)
    with open(results_file, 'a') as f:
        results = df.to_string()
        f.write(results)
    
if __name__=='__main__':
    random_forest_func(data_folder, labels_sequence_folder, model_folder, save_path)


