'''
#Written by Ian Robertson (ARL at UH), adapted from Alejandro Alvaro (FAU)
#python videocnn_test.py

This program 
   1. Crops input videos to a square resolution.
   2. Uses a trained CNN model to identify wave features.
   3. Outputs a single video of labeled wave features.

Input: 
    - a .mp4 video file containing wave breaking (in the raw_videos_folder)
    - an appropriately trained YOLO CNN model (in the model_folder)

Output:
    in the test_folder:
    - a cropped square video with feature labels drawn on the video (same length as orginal)
    - frames (.jpgs) with features labeled on them
    - .txt files associated with each labeled frame that includes label info (coordinates, confidence, etc.)

Note: the first output (video) is for evaluating the CNN performance. This can be commented to save space on the hard drive if you have high confidence in CNN performance.
'''

from ultralytics import YOLO
import cv2
import os
from os.path import join
import operator
import ntpath
from pathlib import Path

#import glob, ntpath

#define output type. options are 'clips' or 'frames'. 'clips' exports video clips of individual wave events. 'frames' exports folders with all the frames from a single wave event
file = '000_002H030T33irr_camF_2_wave193' #input raw video file
home_folder = r".\\YOLO_RF\\Train_YOLO_RF_2" # project folder
test_folder = home_folder + '\\manual_test_videos\\'
model_folder = home_folder + '\\runs\\detect\\train3\\weights\\best.pt' #.pt YOLO CNN file


def videocnn(test_folder, model_folder, file):
    #user inputs
    #Video path and name
    path = test_folder
    save_path = path
    ext = '.mp4'
    start_time = 0 #how many seconds into the video to start (accounts for camera movement at the beginning of the video)

    #choose how many pixels to crop from each side of the image (initial guess)
    #keep these settings for 1024x1024 video
    crop_resolution = 1024
    crop_right = 150
    crop_left = 1920-crop_resolution-crop_right
    crop_bottom = 0
    crop_top = 1080-crop_resolution-crop_bottom
    crop_right = 0
    crop_left = 0
    crop_bottom = 0
    crop_top = 0

    ##############################################################################################################################

    #Set up save path and video capture
    save_name = file + '_processed' + ext

    label_folder_out = test_folder + ntpath.basename(file) + '_cnn_label' + '\\'
    if not os.path.exists(label_folder_out):
        os.makedirs(label_folder_out)

    video = join(path, file + ext)
    main_capture = cv2.VideoCapture(video)
    fps = main_capture.get(cv2.CAP_PROP_FPS)
    main_capture.set(cv2.CAP_PROP_POS_FRAMES, start_time*fps) #set to fps*10 to start 10 seconds into video

    #Get dimensions to utilize if needed.
    width  = main_capture.get(3)  # float `width`
    height = main_capture.get(4)  # float `height`

    # define resolution and crop frame
    RES = int(width-(crop_left+crop_right)), int(height - (crop_top+crop_bottom)) #output video resolution (width, height)
    #RES = int(width), int(height) #output video resolution (width, height)
    ret, frame = main_capture.read()
    frame = frame[crop_top:int(height)-crop_bottom,crop_left:int(width)-crop_right]

    i=0
    proceed = 'n'

    # view first frame of video and make crop adjustments
    while proceed != 'y':
        #redefine video
        main_capture = cv2.VideoCapture(video)
        fps = main_capture.get(cv2.CAP_PROP_FPS)
        main_capture.set(cv2.CAP_PROP_POS_FRAMES, fps*start_time)
        width  = main_capture.get(3)  # float `width`
        height = main_capture.get(4)  # float `height`

        # manually change cropping
        if i >= 1:
            if proceed == 'c':
            #choose how many pixels to crop from each side of the image
                #crop_left = int(input("Insert number of pixels to crop from left of frame. Previous value was " + str(crop_left) + ": "))
                crop_right = int(input("Insert number of pixels to crop from right of frame. Previous value was " + str(crop_right) + ": "))
                crop_bottom = int(input("Insert number of pixels to crop from bottom of frame. Previous value was " + str(crop_bottom) + ": "))
                #crop_top = int(input("Insert number of pixels to crop from top of frame. Previous value was " + str(crop_top) + ": "))
                crop_left = 1920-crop_resolution-crop_right
                crop_top = 1080-crop_resolution-crop_bottom

            RES = int(width-(crop_left+crop_right)), int(height - (crop_top+crop_bottom)) #for 1080p resolution
            ret, frame = main_capture.read()
            frame = frame[crop_top:int(height)-crop_bottom,crop_left:int(width)-crop_right]

        # show resulting image
        cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('image', frame)
        cv2.resizeWindow('image',int(width/2),int(height/2)) #may need to be edited depending on size of computer screen

        if cv2.waitKey(1) == 27:
                main_capture.release()
                cv2.destroyAllWindows()

        # go back and change crop/ROI settings or continue to making video
        proceed = input ('Press c to crop image or press y to continue making video with current settings: ')
        i+=1



    fps = main_capture.get(cv2.CAP_PROP_FPS)


    #Resolution is needed to set up the recording object to save the edited video. Add desired path for video to be saved
    recorder_test = cv2.VideoWriter(join(save_path,save_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, RES) #save to folder for CNN verification/testing

    # Import model
    model = YOLO(model_folder)  # load a custom model
    threshold = 0.8 #choose threshold (only features above threshold will be output) 


    #initialize counters and arrays
    count = -1 #initialize frame count 
    idnum = 0 #initialize wave id tracker
    wave_count = 0 # initialize wave count tracker
    count_list = [] #array to keep track of frames with features  
    idnum_list = [] #array to keep track of unique wave number
    label_list = [] #array to keep track of corresponding feature label
    center_y_list = [] # array to store center y coordinate (for distace calculations)
    prev_frame_data = [] #holds previous frame data
    cur_frame_data_archive = [] # holds data from all frames

    #read in first frame
    ret, frame = main_capture.read()

    # main loop
    while ret:
    #while count < 400: # (for testing)
        #crop frame to desired size to reduce computation cost and reduce possible inaccuracies
        frame = frame[crop_top:int(height)-crop_bottom,crop_left:int(width)-crop_right]
        H, W, _ = frame.shape
        #Increase Frame Count
        count +=1
        #print(count)

        #Create Arrays to hold important Data
        cur_frame_data = []
        results = model(frame)[0]
        score_pass = 0

        # get bounding boxes for waves and sort by descending y coordinate (waves near the bottom of frame listed first)
        raw_results = results.boxes.data.tolist()
        sorted_results = sorted(raw_results, key=operator.itemgetter(1), reverse=True) #sort by y coordinate
        #print(results)

        #decide whether frame should be saved (if the frame contains a feature above confidence threshold)
        for result in sorted_results:
            x1, y1, x2, y2, score, class_id = result
            
            if score > threshold:
                score_pass = 1

        
        if score_pass == 1:    #this means there are features present in the frame
            image_name = Path(file).stem+'_frame'+str(count).zfill(6)
            cv2.imwrite(label_folder_out + image_name + '.jpg', frame)
            for result in sorted_results: # for each feature
                x1, y1, x2, y2, score, class_id = result
                
                if score > threshold: #if the feature confidence value exceeds the threshold
                    # get normalized coordinates
                    center_x = ((x1+x2)/2)/W
                    center_y = ((y1+y2)/2)/H
                    box_width = (x2-x1)/W
                    box_height = (y2-y1)/H
                    # if there are duplicate labels for the same wave, only use one
                    diff = [abs(cur_frame_data[k][2]-center_y) for k in range(0,len(cur_frame_data))]
                    
                    # compare to labels from previous frames
                    if prev_frame_data and not any(dist<0.05 for dist in diff): 
                        #consider frames within 1 second of current frame            
                        last_counts_ind = [i for i, v in enumerate(count_list) if v >=(count - fps*1.0)] 
                        if len(last_counts_ind) >=1:
                            last_y_vals = center_y_list[min(last_counts_ind):] #get y_values of previous 1 second of frames
                            last_idnums = idnum_list[min(last_counts_ind):] #get corresponding wave id numbers
                            prev_wave_y_diff = [center_y-last_y_vals[k] for k in range(0,len(last_y_vals))] #calculate differences
                            prev_wave_y_diff_pos = [i for i in prev_wave_y_diff if i>=-0.02] #get counts of waves that are close enough (and behind current wave)
                            # if there are values that are close, find the smallest one
                            if len(prev_wave_y_diff_pos)>0:
                                smallest_y_diff = min(prev_wave_y_diff_pos)
                                smallest_y_ind = prev_wave_y_diff.index(smallest_y_diff)
                            else:
                                smallest_y_diff = 1
                        else:
                            smallest_y_diff = 1

                        # if there is a wave in the previous few frames close in time and space, assign new wave the same number
                        if smallest_y_diff <=0.1:
                            idnum = last_idnums[smallest_y_ind]
                        #otherwise, increase the wave counter and assign a new number
                        else:                            
                            wave_count+=1
                            #print(wave_count)
                            idnum = wave_count
                                              
                    #print(idnum)   
                    #append new values to existing list                         
                    center_y_list.append(center_y) 
                    idnum_list.append(idnum)
                    count_list.append(count)
                    label_list.append(int(class_id))
                    cur_frame_data.append((int(class_id), center_x, center_y, box_width, box_height, round(score,2), idnum, count))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    #cv2.putText(frame, results.names[int(class_id)].upper()+ '=' + str(round(score,2)) + ', wave ' + str(idnum), (int(x1), int(y1 - 10)),
                        #cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, results.names[int(class_id)].upper()+ '=' + str(round(score,2)), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    #write the label information to a .txt file
                    write_file = label_folder_out + image_name +'.txt'
                    f = open(write_file, "a") 
                    f.write(str(int(class_id)) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' +  str(box_width) + ' '+  str(box_height) +'\n')
                    f.close() 
                    
            #store data
            cur_frame_data_archive.append(cur_frame_data) #append current frame data to the list
            prev_frame_data = cur_frame_data.copy() #current frame data becomes previous frame data

           
        #append to the test video with labels (test_folder)
        recorder_test.write(frame)

        #Set wait key to 0 in order to pause video each frame and advance with any key
        #escape key exits
        #set to any number besides 0 to run continously 
        if cv2.waitKey(1) == 27:
            main_capture.release()
            recorder_test.release()
            cv2.destroyAllWindows()
            break

        #Displays Frames  (comment for speed)
        #cv2.namedWindow('video', cv2.WINDOW_GUI_NORMAL)
        #cv2.imshow('video', frame)
        #cv2.resizeWindow('video',int(width/2),int(height/2))

        ret, frame = main_capture.read()

    #end video capture
    main_capture.release()
    recorder_test.release()
    cv2.destroyAllWindows()

    # end of main loop                     
        

    
# run script
if __name__=='__main__':
    videocnn(test_folder, model_folder, file)