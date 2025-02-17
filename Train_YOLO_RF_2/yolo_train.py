#train the dataset
from ultralytics import YOLO

home_folder = r".\\Train_YOLO_RF_2" #project folder

def yolo_func(home_folder):
    # Load a Model (choose n, s, m, l, or x)
    model = YOLO("yolov8x.yaml") #largest YOLO model
    #model = YOLO("yolov8n.yaml") #smallest YOLO model (quickest)
    project = home_folder + '\\runs\\detect'

    #use model (check config.yaml file to make sure paths are set correctly)
    results = model.train(data=home_folder + "\\config.yaml", epochs=300, save=True, project = project, verbose=True) #adjust epochs to desired value
    results = model.val(project = project)  # evaluate model performance on the validation set

#run function
if __name__=='__main__':
    yolo_func(home_folder)
