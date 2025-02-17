#This program uses computer vision to detect breaking waves. Then, it saves video clips of each wave as separate video clips as .mp4 files or a series of jpgs. 
#Written by Ian Robertson (ARL at UH), adapted from Alejandro Alvaro (FAU Advanced Computing Lab)
#python extract_wave_frames.py


import cv2
import numpy as np
import os
from os.path import join
import json

#import glob, ntpath

#define output type. options are 'clips' or 'frames'. 'clips' exports video clips of individual wave events. 'frames' exports folders with all the frames from a single wave event
output_type = 'frames'
#output_type = 'clips'

#other inputs
home_folder = r".\\Train_YOLO_RF_2"
input_path = r'.\\raw_videos'
save_path = home_folder + '\\cnn_frames\\'
file = '000_011H030T45irr_camF_2'


def extract_wave_frames_func(output_type, input_path, save_path, file): #output type is 'clips' or 'frames'. 

    #user inputs
    ext = '.mp4'
    output_fps = 15 #only save 15 images per second
    start_time = 10 #how many seconds into the video to start (accounts for camera movement at the beginning of the video)

    #choose how many pixels to crop from each side of the image (initial guess)
    #keep these settings to output a 1024x1024 video
    crop_resolution = 1024
    crop_right = 150
    crop_left = 1920-crop_resolution-crop_right
    crop_bottom = 0
    crop_top = 1080-crop_resolution

    '''
    #Define corner points of ROI(Region of Interest) (across, down) (comment if defining below)
    pt_A = [540, 100] #upper left
    pt_B = [540, 100] #upper right
    pt_C = [1345, 700] #lower right
    pt_D = [580, 700] #lower left
    '''

    ##############################################################################################################################

    #Set up save path and video capture
    save_name = file + '_processed' + ext #output cropped video name
    video = join(input_path, file + ext) #input video path
    main_capture = cv2.VideoCapture(video) # start video capture
    fps = main_capture.get(cv2.CAP_PROP_FPS) # get frame rate
    main_capture.set(cv2.CAP_PROP_POS_FRAMES, fps*start_time) #use the frame at the 10 second mark to decide the region of interest (ROI)

    #Get dimensions to utilize if needed.
    width  = main_capture.get(3)  # float `width`
    height = main_capture.get(4)  # float `height`

    # define resolution and crop frame
    RES = int(width-(crop_left+crop_right)), int(height - (crop_top+crop_bottom)) #output video resolution (width, height)
    ret, frame = main_capture.read()
    frame = frame[crop_top:int(height)-crop_bottom,crop_left:int(width)-crop_right]

    #Define corner points of ROI(Region of Interest) (across, down) (comment if already defined above)
    pt_A = (175, 300) #upper left
    pt_B = pt_A #upper right
    #pt_C = (RES[0]-1, RES[1]-1) #lower right
    pt_C = (RES[0]-50, RES[1]-1) #lower right
    pt_D = (350, RES[1]-1) #lower left

    i=0
    proceed = 'n' #set initial value to run through the while loop the first time

    # view first frame of video and make crop and ROI adjustments
    while proceed != 'y':
        #redefine video
        main_capture = cv2.VideoCapture(video)
        fps = main_capture.get(cv2.CAP_PROP_FPS)
        main_capture.set(cv2.CAP_PROP_POS_FRAMES, fps*10) #start 10 seconds into video to adjust ROI accurately
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

            RES = int(width-(crop_left+crop_right)), int(height - (crop_top+crop_bottom)) #for 1080p resolution input
            ret, frame = main_capture.read()
            frame = frame[crop_top:int(height)-crop_bottom,crop_left:int(width)-crop_right] #crop the frame

            # manually change ROI
            if proceed == 'r':
                #Define corner points of ROI(Region of Interest) (across, down)
                pt_A = input("Insert coordinates of top point. Previous value was " + str(int(pt_A[0]))+', ' + str(int(pt_A[1])) + ": ") 
                pt_A = list(map(int, pt_A.split(', ')))
                pt_B = pt_A #upper right
                pt_C = input("Insert coordinates of lower right point. Previous value was " + str(int(pt_C[0]))+', ' + str(int(pt_C[1])) + ": ") 
                pt_C = list(map(int, pt_C.split(', ')))
                pt_D = input("Insert coordinates of lower left point. Previous value was " + str(int(pt_D[0]))+', ' + str(int(pt_D[1])) + ": ") 
                pt_D = list(map(int, pt_D.split(', ')))


        #Draw ROI onto Frame
        color = (0, 255, 0)
        thickness = 9
        cv2.line(frame, pt_A, pt_D, color, thickness)
        cv2.line(frame, pt_B, pt_C, color, thickness)
        cv2.line(frame, pt_D, pt_C, color, thickness)

        # show resulting image
        cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('image', frame)
        cv2.resizeWindow('image',int(width/2),int(height/2)) #may need to be edited depending on size of computer screen

        if cv2.waitKey(1) == 27:
                main_capture.release()
                cv2.destroyAllWindows()

        # go back and change crop/ROI settings or continue to making video
        proceed = input ('Press c to crop image, press r to change ROI, or press y to continue making video with current settings: ')
        i+=1

    fps = main_capture.get(cv2.CAP_PROP_FPS)


    #Set up Knn for motion detection
    subtractor = cv2.createBackgroundSubtractorKNN()

    #Resolution is needed to set up the recording object to save the edited video. Add desired path for video to be saved
    recorder = cv2.VideoWriter(join(save_path,save_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, RES)

    #initialize counter variables 
    count = 0 #frame count
    idnum = 0 #wave count
    wave_count = 0 #will be idnum+1 for accurate wave count
    
    #initialize blank arrays
    prev_frame_data = []#empty array to hold previous array data in order to identify when a new wave breaks
    break_frames = [] #empty array to keep track of frames with wave breaking
    wave_num = [] #empty array to keep track of which wave event is associated with which break frame

    # main loop (loops through frames, identifies and saves break frames and associated wave index, outputs cropped video)
    while True:
        ret, frame = main_capture.read()
        if type(frame)==type(None):
            break
        #crop frame to desired size to reduce computation cost and reduce possible inaccuracies
        frame = frame[crop_top:int(height)-crop_bottom,crop_left:int(width)-crop_right]
        
        #Draw ROI onto Frame
        #color = (0, 255, 0)
        #thickness = 9
        #cv2.line(frame, pt_A, pt_D, color, thickness)
        #cv2.line(frame, pt_B, pt_C, color, thickness)
        #cv2.line(frame, pt_D, pt_C, color, thickness)
        
        #Create mask to ignore data outside ROI 
        mask2 = np.zeros(frame.shape, dtype=np.uint8) 
        myROI = np.array([[pt_A, pt_B, pt_C, pt_D]], dtype=np.int32)
        channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask2, myROI, ignore_mask_color)
        frame2 = cv2.bitwise_and(frame, mask2)
        

        #Create Motion Mask to Detect Waves
        mask = subtractor.apply(frame2, 1)
        countours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #Increase Frame Count
        count +=1
        #Create Arrays to hold important Data
        center_points_cur_frame = []
        cur_frame_data = []
        #Set up Text to Display
        font                   = cv2.FONT_HERSHEY_SIMPLEX 
        bottomLeftCornerOfText = (10,500)
        fontScale              = 1
        fontColor              = (0,0,255)
        thickness              = 3
        lineType               = 2
        for cnt in countours:
            #Get the area of motion
            area = cv2.contourArea(cnt)
            #Set Criteria for Identifying wave
            if area > 15000 and area < 150000: #might need to be adjusted (IR)
                #Draw the contour of detected wave
                #cv2.drawContours(frame,[cnt],-1,(0,255,0),2)
                x,y,w,h = cv2.boundingRect(cnt)
                #Add data to current frame array
                cur_frame_data.append((x,y,w,h,idnum,count))
                #Draw the Rectangular bound
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0, 255),2)
                num = 0
                #Criteria for determining if a new wave is present
                for num in range(len(cur_frame_data)):
                    #print(str(cur_frame_data[num])+ "\n")
                    #checks if we have previous wave stored
                    if not prev_frame_data:
                        pass
                    #If we have previous wave we compare with current wave 
                    else:
                        #Compare the number of waves
                        #print(len(cur_frame_data)-1)
                        #If the number of waves is different skip
                        if len(prev_frame_data) < len(cur_frame_data):
                            pass
                        else:
                            #Check distance between the wave front from last frame and current frame also allows for frame to be untracked for a few frames to account for error
                            if (prev_frame_data[num][1]+prev_frame_data[num][3]) - (cur_frame_data[num][1]+cur_frame_data[num][3])> 25 or cur_frame_data[num][5] - prev_frame_data[num][5] >= fps*1.5: #might need to be changed (IR)
                                #checks to see if the wave is breaking apart and that is why there are multiple waves present
                                if (prev_frame_data[num][1]) - (cur_frame_data[num][1] + cur_frame_data[num][3])> 25 or cur_frame_data[num][5]-prev_frame_data[num][5] >= fps*1.5: #might need to be changed (IR)
                                        idnum +=1
                                        #print((prev_frame_data[num][1]+prev_frame_data[num][3]) - cur_frame_data[num][1])
                                        #print((prev_frame_data[num][1]) - (cur_frame_data[num][1] + cur_frame_data[num][3]))
                                        #print(cur_frame_data[num][5] - prev_frame_data[num][5])

                    num += 1
                    
                wave_count = idnum+1 #count which wave we are on

                # create arrays to track breaking waves
                break_frames.append(count) #add the current frame to array of frames that have wave breaking
                wave_num.append(wave_count) #add the corresponding wave number to array of wave numbers
                prev_frame_data = cur_frame_data.copy() #copy current frame data to refer to on next iteration
                

        #Displays Frames  (comment for speed)
        #cv2.namedWindow('video', cv2.WINDOW_GUI_NORMAL)
        #cv2.imshow('video', frame)
        #cv2.resizeWindow('video',int(width/2),int(height/2))

        #Saves Frames
        recorder.write(frame) #writes cropped frames to a video (just crops outputs square video, no other filtering is applied here)

        #Set wait key to 0 in order to pause video each frame and advance with any key
        #escape key exits
        #set to any number besides 0 to run continously 
        if cv2.waitKey(1) == 27:
            main_capture.release()
            recorder.release()
            cv2.destroyAllWindows()
            break

    #release capture and recorder once video is completely cycled through
    main_capture.release()
    recorder.release()
    cv2.destroyAllWindows()
    print("Wave Count = "+ str(wave_count))

    #end of main loop


    # Use the data collected on breaking frames to extract useful frames (or clips) from the cropped video

    # save breaking frame data if needed later
    with open(save_path + file + '_wave_num.json', 'w') as f:
        json.dump(wave_num, f)

    with open(save_path + file + '_break_frames.json', 'w') as f:
        json.dump(break_frames, f)

    # load data if needed
    #with open(save_path + file + '_wave_num.json', 'r') as f:
        #wave_num = json.load(f)

    #with open(save_path + file + '_break_frames.json', 'r') as f:
        #break_frames = json.load(f)

    # Go back through the video and save frames that have breaking

    # Find breakpoints for wave clips
    wave_change, = np.where(np.diff(wave_num) == 1)
    break_points_begin = [break_frames[index+1] for index in wave_change]
    break_points_begin.insert(0, break_frames[0])
    break_points_end = [break_frames[index] for index in wave_change]
    break_points_end.append(break_frames[len(break_frames)-1])
    
    #make and save a classes list as a txt file in each folder (in preparation for labeling for CNN training)
    classes = str('prebreaking' + '\n' + 'curling' + '\n' + 'splashing' + '\n' +'whitewash' + '\n' + 'crumbling')
    input_filename = join(save_path, save_name)
    output_dir = join(save_path, input_filename[:-4])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)         
    f = open(output_dir + '\\classes.txt', "a") 
    f.write(classes)
    f.close()

    #go through each wave event and save corresponding images
    for i in range(len(break_points_begin)):
        clip_name = 'wave' + "{:03}".format(i+1)
        output_filename = join(save_path, save_name[:-4], file + '_' + clip_name)
        start_frame = break_points_begin[i]
        if start_frame > fps*1.5:
            start_frame = int(start_frame - (fps*1.5))  # save images 1.5s prior to breaking to capture the start of the wave
        else:
            start_frame = 0
        end_frame = break_points_end[i]
        if output_type == 'frames':
            #if not os.path.exists(join(input_filename[:-4], clip_name)):
                #os.makedirs(join(input_filename[:-4], clip_name))
            get_frame(input_filename, output_filename,  start_frame, end_frame, output_fps)
        elif output_type == 'clips':
            get_clip(input_filename, output_filename,  start_frame, end_frame, fps, RES)
        

##################### additional functions ######################################

# extract video clip given start and end frames
def get_clip(input_filename, output_filename,  start_frame, end_frame, fps, RES):
    # input and output videos are probably mp4
    vidcap = cv2.VideoCapture(input_filename)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    output_filename_vid = output_filename + '.mp4'
    
    # open video writer
    vidwrite = cv2.VideoWriter(output_filename_vid, cv2.VideoWriter_fourcc(*'h256'), fps, RES)
    
    success, image = vidcap.read()
    frame_count = start_frame
    while success and (frame_count < end_frame):
        vidwrite.write(image)  # write frame into video
        success, image = vidcap.read()  # read frame from video
        frame_count+=1
    vidwrite.release()
    vidcap.release()

# extract specified frames given start and end frames

def get_frame(input_filename, output_filename,  start_frame, end_frame, output_fps):
    cap = cv2.VideoCapture(input_filename) #start video capture
    clip_fps = cap.get(cv2.CAP_PROP_FPS) #get fps

    #initialize counters
    j=0
    frame_count = start_frame 

    # loop through each wave event and save frames
    while (start_frame <= frame_count) and (frame_count <= end_frame): #look at frames from a unique wave event
        if j%(int(round(clip_fps/output_fps)))==0: #only save a subsample of frames (based on 'output_fps')
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read() # get frame
            output_filename_image = output_filename + '_' + "{:03}".format(j+1) + '.jpg' #image name
            cv2.imwrite(output_filename_image, frame) #save image
        j+=1
        frame_count+=1
    cap.release()


# run script   
if __name__=='__main__':
    extract_wave_frames_func(output_type, input_path, save_path, file)



