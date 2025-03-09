# This function takes a .txt file list of feature occurences in plunging and spilling wave videos and 
# outputs a random forest model for classifying waves
# python random_forest.py

#adapted from https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/
#edited by Ian Robertson

# import required libraries 
# import Scikit-learn library and datasets package 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
import pandas as pd 
from sklearn import metrics 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from generate_stats import generate_stats_func
import os

# function inputs
#home_folder = r".\\Train_YOLO_RF_2"
data_folder = r'.\\data' 
save_path = data_folder #where output data is saved
test_size=0.2 #decimal percentage of data to reserve for testing (validation)
n_estimators=100
random_state = 4

def random_forest_func(data_folder, save_path, test_size, n_estimators, random_state):
    # Load the wave datasets
    feature_names = ['prebreaking', 'curling', 'splashing', 'whitewash', 'crumbling'] #make sure this order matches the order on the config.yaml file
    plunging_labels = generate_stats_func(data_folder, 'plunging') #generates feature label statistics based on feature label lists
    spilling_labels = generate_stats_func(data_folder, 'spilling') #generates feature label statistics based on feature label lists
    #plunging_labels = np.load(r".\\plunging_labels.npy")
    #spilling_labels = np.load(r".\\spilling_labels.npy")
    plunging_class = np.zeros(len(plunging_labels))
    spilling_class = np.ones(len(spilling_labels))


    # divide the datasets into two parts, i.e., train datasets and test datasets 
    X = np.concatenate((plunging_labels, spilling_labels), axis = 0)
    y = np.concatenate((plunging_class, spilling_class))

    # Split arrays or matrices into random train and test subsets 
    # i.e. 70 % training dataset and 30 % test datasets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify=y, random_state=2) #random state can be adjust or eliminated if desired

    # create a RF classifier
    clf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state) 

    # other options to experiment with  
    #clf = RandomForestRegressor(n_estimators = 100, random_state=random_state) 
    #clf = HistGradientBoostingClassifier(max_iter = 100, random_state=random_state) 
    #clf = HistGradientBoostingRegressor(max_iter = 100, random_state=random_state) 
    #clf = GradientBoostingClassifier(n_estimators = 100, random_state=random_state) 
    #clf = GradientBoostingRegressor(n_estimators = 100, random_state=random_state) 

    # Train the model on the training dataset 
    # fit function is used to train the model using the training sets as parameters 
    clf.fit(X_train, y_train) 

    # perform predictions on the test dataset 
    y_pred = clf.predict(X_test) 

    # see values of incorrect predictions (for Regressors only)
    #for i in range (0, len(y_pred)):
        #if abs(y_pred[i]-y_test[i])>0.5:
            #print(i, y_pred[i],y_test[i])

    # metrics are used to find accuracy or error 
    matrix = metrics.confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    # Build the plot
    plt.figure(figsize=(16,12))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2)
    # Add labels to the plot
    class_names = ['Plunging', 'Spilling']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    #plt.show()

    # using metrics module for accuracy calculation 
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) 
    print(metrics.classification_report(y_test, y_pred))
    # predict which type of wave it is. 
    # print(clf.predict([[.5, .2, .02, .01, .07, .2]]))

    # obtain the feature importance variable (shows which labels were most important in the classification decision)
    feature_imp = pd.Series(clf.feature_importances_, index = feature_names).sort_values(ascending = False) 
    print(feature_imp)

    ## save data
    # save confusion matrix
    confusion_file = save_path+'\\confusion_matrix_' + str(random_state) + '.png'
    plt.savefig(confusion_file)

    # save feature importance data
    feature_imp_file = save_path + '\\feature_importance_' + str(random_state) + '.txt'
    if os.path.exists(feature_imp_file):
        os.remove(feature_imp_file)
    with open(feature_imp_file, 'a') as f:
        feature_imp_string = feature_imp.to_string()
        f.write(feature_imp_string)

    # save classification report
    class_report_file = save_path + '\\'+ 'classification_report_' + str(random_state) + '.txt'
    if os.path.exists(class_report_file):
        os.remove(class_report_file)
    with open(class_report_file, 'a') as f:
        f.write("ACCURACY OF THE MODEL: " + str(metrics.accuracy_score(y_test, y_pred))+'\n')
        f.write(metrics.classification_report(y_test, y_pred))
    
    # save random forest model
    joblib.dump(clf, save_path + '//random_forest_' + str(random_state) + '.joblib')
     


if __name__=='__main__':
    random_forest_func(data_folder, save_path, test_size, n_estimators, random_state)

