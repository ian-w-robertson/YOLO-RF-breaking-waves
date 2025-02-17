# This folder contains a variety of input and output files for the random forest model. 
- plunging.txt  and spilling.txt are numeric feature lists for all plunging and spilling waves in the training set respectively
- plunging_labels.npy and spilling_labels.npy are the normalized vectors of features for all plunging and spilling waves in the dataset
- random_forest_[#].joblib is the random forest model file, labeled according to the number (#)
- classification_report_[#], confusion_matrix_[#], and feature_importance_[#] are the performance reports associated with the # random forest model
