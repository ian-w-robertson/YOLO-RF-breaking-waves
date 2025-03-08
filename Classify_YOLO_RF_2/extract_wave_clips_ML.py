'''
#Written by Ian Robertson (ARL at UH), adapted from Alejandro Alvaro (FAU)
#python extract_wave_clips_ML.py

This program 
   1. Crops input videos to a square resolution.
   2. Uses a trained CNN model to identify wave features and create a list of wave frames associated with each unique breaking event.
   3. Output a sequence of feature labels corresponding to each unique wave in a txt file

Input: 
    - a .mp4 video file containing wave breaking (in the raw_videos_folder)
    - an appropriately trained YOLO CNN model (in the model_folder)

Outputs include:
    in the clips_folder:
    - a cropped square video (same length as original)
    - video clips (.mp4) for each unique wave event
    - corresponding .txt file of all the labels associated with each video clip (one per clip)

    in the test_folder:
    - a cropped square video with feature labels drawn on the video (same length as orginal)
    - frames (.jpgs) with features labeled on them
    - .txt files associated with each labeled frame that includes label info (coordinates, confidence, etc.)

Note: materials in the test_folder are for testing the CNN. This can be commented to save space on the hard drive if you have high confidence in CNN performance.
'''

import os
from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
from os.path import join
import operator

#import glob, ntpath

#define output type. options are 'clips' or 'frames'. 'clips' exports video clips of individual wave events. 'frames' exports folders with all the frames from a single wave event
output_type = 'clips' #use clips unless altering the architecture of the other programs
file = '002H30T33irr_camF_3'
base_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
#home_folder = r".\\Classify_YOLO_RF_2" #project folder
train_home_folder = base_folder + r"\\Train_YOLO_RF_2" #project folder for training
test_home_folder = base_folder + r"\\Test_YOLO_RF_2"
raw_videos_folder = base_folder + r'\\raw_videos' #for videos on external drive (saves space on local drive)
#model_folder = train_home_folder + '\\runs\\detect\\train3\\weights\\best.pt' #choose YOLO model here
model_folder = r'.\\models\\YOLO_train3_best.pt' #choose YOLO model to use, use model in current folder if you have not trained your own model

# don't change unless altering path structure 
clips_folder = '.\\videos\\'  # if doing step 1, use this path
test_folder = '.\\videos_test\\' # if doing step 1, use this path
#clips_folder = test_home_folder + '\\videos\\' # if step 1 already completed in testing, use test path to get videos
#test_folder = test_home_folder + '\\videos_test\\' # if step 1 already completed in testing, use test path to get videos


def extract_waves_func(output_type, raw_videos_folder, clips_folder, test_folder, model_folder, file): #output type is 'clips' or 'frames'. 
    #user inputs
    #Video path and name
    path = raw_videos_folder
    save_path = clips_folder
    test_path = test_folder
    ext = '.mp4'
    if not os.path.exists(join(test_path, file)): #create test_folder for video
        os.makedirs(join(test_path, file))

    #choose how many pixels to crop from each side of the image (initial guess)
    #keep these settings for 1024x1024 video
    crop_resolution = 1024
    crop_right = 150
    crop_left = 1920-crop_resolution-crop_right
    crop_bottom = 0
    crop_top = 1080-crop_resolution-crop_bottom

    ##############################################################################################################################

    #Set up save path and video capture
    save_name = file + '_processed' + ext
    video = join(path, file + ext)
    main_capture = cv2.VideoCapture(video)
    fps = main_capture.get(cv2.CAP_PROP_FPS)
    main_capture.set(cv2.CAP_PROP_POS_FRAMES, fps*10)

    #Get dimensions to utilize if needed.
    width  = main_capture.get(3)  # float `width`
    height = main_capture.get(4)  # float `height`

    # define resolution and crop frame
    RES = int(width-(crop_left+crop_right)), int(height - (crop_top+crop_bottom)) #output video resolution (width, height)
    ret, frame = main_capture.read()
    frame = frame[crop_top:int(height)-crop_bottom,crop_left:int(width)-crop_right]

    i=0
    proceed = 'n'

    # view first frame of video and make crop adjustments
    while proceed != 'y':
        #redefine video
        main_capture = cv2.VideoCapture(video)
        fps = main_capture.get(cv2.CAP_PROP_FPS)
        main_capture.set(cv2.CAP_PROP_POS_FRAMES, fps)
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


    #Get frames to save at same rate as original
    fps = main_capture.get(cv2.CAP_PROP_FPS)

    #Resolution is needed to set up the recording object to save the edited video. Add desired path for video to be saved
    recorder = cv2.VideoWriter(join(save_path,save_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, RES) #save to folder for future labeling
    recorder_test = cv2.VideoWriter(join(test_path,save_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, RES) #save to folder for CNN verification/testing

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
        print(count)

        #Save Frames
        recorder.write(frame)

        #Set wait key to 0 in order to pause video each frame and advance with any key
        #escape key exits
        #set to any number besides 0 to run continously 
        if cv2.waitKey(1) == 27:
            print('failed')
            main_capture.release()
            recorder.release()
            cv2.destroyAllWindows()
            break

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
                    cv2.putText(frame, results.names[int(class_id)].upper()+ '=' + str(round(score,2)) + ', wave ' + str(idnum), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    #write the label information to a .txt file
                    write_file = test_path+'\\'+ Path(file).stem + '\\' + Path(file).stem+'_frame'+str(count).zfill(6)+'.txt'
                    f = open(write_file, "a") 
                    f.write(str(int(class_id)) + ' ' + str(center_x) + ' ' + str(center_y) + ' ' +  str(box_width) + ' '+  str(box_height)+  ' ' + str(round(score,2)) + ' ' + str(idnum) + ' ' + str(count) +'\n')
                    f.close() 

            #store data
            cur_frame_data_archive.append(cur_frame_data) #append current frame data to the list
            prev_frame_data = cur_frame_data.copy() #current frame data becomes previous frame data

            #save the image with labels (test_folder)
            cv2.imwrite(test_path+'\\'+ Path(file).stem + '\\' + Path(file).stem+'_frame'+str(count).zfill(6)+'.jpg',frame) 

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
    recorder.release()
    recorder_test.release()
    cv2.destroyAllWindows()

    # end of main loop                     
        

    # Go back through video and save clips and label sequences based on label information

    # Find breakpoints for wave clips
    idnum_array = np.array(idnum_list)

    if len(idnum_array) > 0:
        #initialize arrays to store beginning and end frames of each breaking wave event
        min_frame_count_list = np.zeros(max(idnum_array)+1)
        max_frame_count_list = np.zeros(max(idnum_array)+1)
        wave_count_index_list = []
        #build a list of start and end array indices corresponding to each unique wave event
        for i in range(0, len(max_frame_count_list)):
            idx_list, = np.where(idnum_array==i) #find indices of the array that correspond to wave number "i"
            wave_count_index_list.append(idx_list) #append to master list
            min_idx = min(idx_list)
            max_idx = max(idx_list)
            min_frame_count_list[i] = count_list[min_idx] #start frame
            max_frame_count_list[i] = count_list[max_idx] #end frame

        # Extract videos using start and end points for each wave
        for i in range(0, len(min_frame_count_list)):
            clip_name = 'wave' + "{:03}".format(i+1)
            input_filename = join(save_path,save_name)
            wave_label_list = [label_list[idx]+1 for idx in wave_count_index_list[i]]

            #calculate number of "breaking" labels to later exclude non-breaking waves with erroneous breaking labels or very short break times
            breaking_number = ([label!=1 for label in wave_label_list].count(True)) #count the number of labels that are not "1" ("breaking labels")
            if not all([label==1 for label in wave_label_list]) and breaking_number > fps/4: #excludes non-breaking waves and very short waves (1/4 second)
                start_frame = min_frame_count_list[i]
                end_frame = max_frame_count_list[i]
                #create folder
                if not os.path.exists(input_filename[:-4]):
                    os.makedirs(input_filename[:-4])
                #save frames in a folder
                if output_type == 'frames': 
                    output_filename = join(save_path, save_name[:-4], clip_name, file + '_' + clip_name)
                    if not os.path.exists(join(input_filename[:-4], clip_name)):
                        os.makedirs(join(input_filename[:-4], clip_name))
                    get_frame(input_filename, output_filename,  start_frame, end_frame)
                #save video clips in a folder
                elif output_type == 'clips':
                    output_filename = join(save_path, save_name[:-4], file + '_' + clip_name)
                    #print(output_filename)
                    get_clip(input_filename, output_filename,  start_frame, end_frame, fps, RES)
                # write list of wave labels to a text file
                write_file = output_filename + '.txt'
                f = open(write_file, "a") 
                f.write(str(wave_label_list))
                f.close()
    else:
        print('No waves were detected')
        

# functions

# extract video clip given start and end frames
def get_clip(input_filename, output_filename,  start_frame, end_frame, fps, RES):

    #prepare to extract frames
    vidcap = cv2.VideoCapture(input_filename) # begin capture
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,start_frame) #set start_frame
    output_filename_vid = output_filename + '.mp4' 
    
    # open video writer
    vidwrite = cv2.VideoWriter(output_filename_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, RES)
    success, image = vidcap.read()
    frame_count = start_frame

    #write each frame to a video clip
    while success and (frame_count < end_frame):
        vidwrite.write(image)  # write frame into video
        success, image = vidcap.read()  # read frame from video
        frame_count+=1

    #close video writers
    vidwrite.release()
    vidcap.release()

# extract specified frames given start and end frames
def get_frame(input_filename, output_filename,  start_frame, end_frame):
    #prepare to extract frames
    cap = cv2.VideoCapture(input_filename)
    #clip_fps = cap.get(cv2.CAP_PROP_FPS)
    j=0
    frame_count = start_frame

    #write each frame to an image file
    while (start_frame <= frame_count) and (frame_count <= end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        output_filename_image = output_filename + '_' + "{:03}".format(j+1) + '.jpg'
        cv2.imwrite(output_filename_image, frame)
        j+=1
        frame_count+=1
        
    #close image writer
    cap.release()

# run script
if __name__=='__main__':
    extract_waves_func(output_type, raw_videos_folder, clips_folder, test_folder, model_folder, file)
