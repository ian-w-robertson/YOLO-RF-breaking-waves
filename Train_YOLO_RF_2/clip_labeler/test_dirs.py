import tkinter as tk
from tkinter import *
from tkinter import messagebox
import os
from threading import Thread

from tkVideoPlayer import TkinterVideo
from queueHandler import QueueHandler
from decisionLogApplier import apply_changes
#debug
#from test import reorganize

#Could use TOML to manage config but might be a hassle
#User config for paths

home_folder = r".\\YOLO_RF\\Train_YOLO_RF_2"
file = '000_008H050T40reg_camF_processed'
save_path = home_folder + '\\labeled_wave_sequences'
path = home_folder + ".\\YOLO_RF\\Train_YOLO_RF_2\\clip_labeler\\clips\\"
todo_path = path + file #path where input videos are stored


# Get the path two directories up
two_up_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

print(two_up_dir)