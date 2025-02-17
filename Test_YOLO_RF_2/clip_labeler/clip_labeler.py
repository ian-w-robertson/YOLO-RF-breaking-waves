# labels wave clips as plunging or spilling
# written by Ayden Malahoff-Kamei 
# python clip_labeler.py 


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

home_folder = os.path.abspath(os.path.join(os.getcwd(), ".."))
file = 'test_1'
save_path = home_folder + '\\labeled_wave_sequences'
path = home_folder + ".\\YOLO_RF\\Train_YOLO_RF_2\\clip_labeler\\clips\\"
todo_path = path + file #path where input videos are stored

#Main function
def clip_labeler_func(todo_path, home_folder, file, save_path):
    plunging_path = save_path + '\\plunging'
    unclear_path = save_path + '\\unclear'
    spilling_path = save_path + '\\spilling'
    #QueueHandler object
    queueHandler = QueueHandler(todo_path, plunging_path, 
                        unclear_path, spilling_path, home_folder)


    #Creates tkinter window
    root = tk.Tk()
    root.title("Clip Labeler")
    root.geometry('1000x800')


    #Displays working directory
    top_frame = tk.Frame(master=root, bg="#a9a9a9")
    directory_label = Label(top_frame, text=queueHandler.todo_path, font="Bold",
                            padx=5, bg='#a9a9a9')
    directory_label.grid(column=0, row=0)
    top_frame.pack(fill=X, side=TOP)


    #Displays queue
    queue = queueHandler.get_queue_list()

    queue_frame = tk.Frame(master=root, bg='#d3d3d3')
    queue_frame.grid_rowconfigure(0, weight=1)
    queue_frame.grid_columnconfigure(0, weight=1)

    prev_label = Label(queue_frame, bg='#d3d3d3', text=queue[0])
    current_label = Label(queue_frame, bg="#d3d3d3", text=queue[1])
    next_label = Label(queue_frame, bg='#d3d3d3', text=queue[2])

    prev_label.grid(row=0, sticky=W)
    current_label.grid(row=0, sticky=N)
    next_label.grid(row=0, sticky=E)

    queue_frame.pack(side=TOP, fill=X)


    #Video Frame
    video_frame = tk.Frame(master=root)
    video_player = TkinterVideo(master=video_frame, scaled=True)
    video_player.pack(expand=True, fill='both')
    video_player.load(queueHandler.get_clip_path())
    video_player.play()
    video_frame.pack(side=TOP, expand=True, fill='both')


    #Disable buttons
    def disable(*args):
        plung_bt['state'] = tk.DISABLED
        unclear_bt['state'] = tk.DISABLED
        spill_bt['state'] = tk.DISABLED
        null_btn['state'] = tk.DISABLED

    #Enable buttons
    def enable(*args):
        play_btn['text'] = "Play"
        plung_bt['state'] = tk.NORMAL
        unclear_bt['state'] = tk.NORMAL
        spill_bt['state'] = tk.NORMAL
        null_btn['state'] = tk.NORMAL
        
    #Exit function
    def exit_func():
        root.destroy()
        if queueHandler.queue_position == 1:
            exit()
        result = messagebox.askquestion(title="Exiting", message="You are exiting midway through your queue"
                                "\nWould you like to apply these decisions using the current decision log? (You can always do this later)")
        if result == "yes":
            apply_changes(queueHandler.log)
        else:
            exit()

    #Organize button function
    def catagorize(choice):
        if queueHandler.queue_position+2 == len(queueHandler.clip_list):
            queueHandler.catagorize(choice)
            root.destroy()
            result = messagebox.askquestion(title="Good job!", message="You have completed all items in your queue!"
                                "\nWould you like to apply these decisions using the current decision log? (You can always do this later)")
            if result == "yes":
                apply_changes(queueHandler.log)
            else:
                exit()
        else:
            Thread(target=disable).start()
            video_player.load(queueHandler.get_clip_path(1))
            video_player.play()
            queueHandler.catagorize(choice)
            queueHandler.next()
            Thread(target=update_queue).start()
                
    #Update queue bar function
    def update_queue():
        queue = queueHandler.get_queue_list()
        prev_label.config(text=queue[0])
        current_label.config(text=queue[1])
        next_label.config(text=queue[2])

    #Play/pause func
    def play_pause(*args):
        Thread(target=disable).start()
        if video_player.is_paused():
            video_player.play()
            play_btn['text'] = "Pause"
        else:
            video_player.pause()
            play_btn['text'] = "Play"


    #Button frame
    organize_button_frame = tk.Frame(master=root, padx=15, pady= 10)
    organize_button_frame.grid_rowconfigure(0, weight=1)
    organize_button_frame.grid_columnconfigure(0, weight=1)


    #Create buttons, bind keys and bind to grid and frame
    plung_bt = Button(organize_button_frame, command=lambda: catagorize("plunging"),
                       text="Plunging Wave", font=("24"), padx=15, pady= 10, state="disabled")
    plung_bt.grid(row=0, sticky=W)
    root.bind("<Left>", lambda *args: plung_bt.invoke())

    unclear_bt = Button(organize_button_frame, command=lambda: catagorize("unclear"), 
                        text="Unclear Wave", font=('24'), padx=15, pady= 10, state="disabled")
    unclear_bt.grid(row=0, sticky=N)
    root.bind("<Up>", lambda *args: unclear_bt.invoke())

    spill_bt = Button(organize_button_frame, command=lambda: catagorize("spilling"), 
                      text="Spilling Wave", font=('24'), padx=15, pady= 10, state="disabled")
    spill_bt.grid(row=0, sticky=E)
    root.bind("<Right>", lambda *args :spill_bt.invoke())

    #Pack buttons
    organize_button_frame.pack(side=BOTTOM, fill=X)

    #Control button frame
    control_button_frame = tk.Frame(master=root, padx=10, pady=5, bg='#d3d3d3')
    control_button_frame.grid_rowconfigure(0, weight=1)
    control_button_frame.grid_columnconfigure(0, weight=1)

    #Create control buttons, bind to grid and frame
    play_btn = Button(control_button_frame, command=play_pause, text="Pause", font="12", padx=10, pady=5)
    play_btn.grid(row=0, sticky=N)

    exit_btn = Button(control_button_frame, command=exit_func, text="Exit", font="12", padx=10, pady=5)
    exit_btn.grid(row=0, column=0, sticky=W)

    null_btn = Button(control_button_frame, command=lambda: catagorize("null"), text="Null", font="12", padx=10, pady=5, state="disabled")
    null_btn.grid(row=0, column=0, sticky=E)

    #Bind space bar to pause play func
    root.bind("<space>", lambda *args :play_btn.invoke())

    #Pack the control button frame
    control_button_frame.pack(side=BOTTOM, fill=X)

    #Lets user click buttons following video end
    video_player.bind("<<Ended>>", enable)

    #Mainloop
    root.mainloop()

if __name__ == "__main__":
    if os.listdir(todo_path) == []:
        print("No clips in clips folder")
        exit()
    else:
        clip_labeler_func(todo_path, home_folder, file, save_path)
