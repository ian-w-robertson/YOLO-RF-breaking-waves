# This function takes a .txt file list of feature occurences in plunging and spilling wave videos and 
# outputs a random forest model for classifying waves

#adapted from https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/
#edited by Ian Robertson

# import required libraries 
# import Scikit-learn library and datasets package  
import pandas as pd 
from sklearn import metrics 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from generate_stats import generate_stats_func
import os


# function inputs
home_folder = r".\\YOLO_RF\\Test_YOLO_RF_2"
train_home_folder = r".\\YOLO_RF\\Train_YOLO_RF_2"
model_folder = train_home_folder + '\\data\\random_forest_0.joblib'
data_folder = home_folder + '\\data'
labels_sequence_folder = home_folder + '\\labeled_wave_sequences'  
save_path = data_folder #where output data is saved

def random_forest_func(data_folder, labels_sequence_folder, model_folder, save_path):

    #generate feature label statistics based on feature label lists
    plunging_percent_frames = generate_stats_func(data_folder, 'plunging') #plunging
    spilling_percent_frames = generate_stats_func(data_folder, 'spilling') #spilling

    #create label arrays then combine datasets
    plunging_class = np.zeros(len(plunging_percent_frames))
    spilling_class = np.ones(len(spilling_percent_frames))
    data = np.concatenate((plunging_percent_frames, spilling_percent_frames), axis = 0)
    labels = np.concatenate((plunging_class, spilling_class))

    #make a list of video clip names associated with each category
    plunging_names = [txt.split('.')[0] for txt in os.listdir(labels_sequence_folder + '\\plunging') if txt.endswith('.txt')]
    spilling_names = [txt.split('.')[0] for txt in os.listdir(labels_sequence_folder + '\\spilling') if txt.endswith('.txt')]
    video_names = plunging_names + spilling_names

    #load the random forest model
    clf = joblib.load(model_folder)
    class_names = ['plunging', 'spilling']

    # Specify feature names
    feature_names = ['prebreaking', 'curling', 'splashing', 'whitewash', 'crumbling']

    # perform predictions on the test dataset 
    y_pred = clf.predict(data) 
    pred_labels = [class_names[int(i)] for i in y_pred]
    for i in range(0, len(data)):
        if data[i][0] == 9999:
            y_pred[i] = 9999
            pred_labels[i]='null'

    # see values of incorrect predictions
    #for i in range (0, len(y_pred)):
        #if y_pred[i]!=labels[i]:
            #print(video_names[i], y_pred[i], labels[i])

    if labels is None:

        data_frame = {'name': video_names,
                'value': y_pred, 'label': pred_labels}
        df = pd.DataFrame(data_frame)
        print(df)
        
        results_file = save_path + '\\'+ 'results.txt'
        if os.path.exists(results_file):
            os.remove(results_file)
        with open(results_file, 'a') as f:
            results = df.to_string()
            f.write(results)

    else:
        
        # metrics are used to find accuracy or error 
        matrix = metrics.confusion_matrix(labels, y_pred)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        # Build the plot
        plt.figure(figsize=(16,12))
        sns.set(font_scale=1.4)
        sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                    cmap=plt.cm.Greens, linewidths=0.2)
        # Add labels to the plot
        tick_marks = np.arange(len(class_names))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks, class_names, rotation=25)
        plt.yticks(tick_marks2, class_names, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix for Random Forest Model')
        #plt.show()
        # using metrics module for accuracy calculation 
        print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(labels, y_pred)) 
        print(metrics.classification_report(labels, y_pred))

        # save confusion matrix
        plt.savefig(save_path+'\\confusion_matrix.png')

        # using the feature importance variable 
        feature_imp = pd.Series(clf.feature_importances_, index = feature_names).sort_values(ascending = False) 
        #print(feature_imp)

        results_file = save_path + '\\'+ 'feature_importance.txt'
        if os.path.exists(results_file):
            os.remove(results_file)
        with open(results_file, 'a') as f:
            feature_imp_string = feature_imp.to_string()
            f.write(feature_imp_string)

        results_file = save_path + '\\'+ 'classification_report.txt'
        if os.path.exists(results_file):
            os.remove(results_file)
        with open(results_file, 'a') as f:
            f.write("ACCURACY OF THE MODEL: " + str(metrics.accuracy_score(labels, y_pred))+'\n')
            f.write(metrics.classification_report(labels, y_pred))
     
if __name__=='__main__':
    random_forest_func(data_folder, labels_sequence_folder, model_folder, save_path)


