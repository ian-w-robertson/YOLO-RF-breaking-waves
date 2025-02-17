import os
from datetime import datetime

class QueueHandler:
    def __init__(self, todo_path: str, plunging_path: str, unclear_path: str, spilling_path: str, home_folder: str):
        #Save paths
        self.todo_path = os.path.abspath(todo_path)
        self.plunging_path = os.path.abspath(plunging_path)
        self.unclear_path = os.path.abspath(unclear_path)
        self.spilling_path = os.path.abspath(spilling_path)

        #Create Decision log, YYYY-MM-DD HH:MM:SS
        #self.log = home_folder + '\\clip_labeler\\logs\\{datetime.now().strftime("%Y-%m-%d %H_%M_%S")}.txt'
        self.log = home_folder + '\\clip_labeler\\logs\\' + os.path.basename(todo_path)+'.txt'
        if not os.path.exists(self.log):
            open(self.log, 'x')
        else:
            print('Log file already exists for this dataset. To continue adding labels to the log, press y. To cancel, press n')
            cont = str(input())
            if cont == 'y':
                print('Proceeding')
            else:
                print('Operation terminated')
                return

        
        
        #Get list of clip names
        try:
            files = os.listdir(todo_path)
            self.clip_list = [file for file in files if file.endswith('.mp4')]
            self.clip_list.insert(0, "None")
            self.clip_list.append("Finished!")
        except Exception as e:
            print(f"Failed to load clips with exception: {e}")
        self.queue_position = 1
    
    def check_load(self) -> str:
        if not os.path.exists(self.todo_path):
            return "Todo path invalid"
        if not os.path.exists(self.plunging_path):
            return "Plunging path invalid"
        if not os.path.exists(self.unclear_path):
            return "Unclear path invalid"
        if not os.path.exists(self.spilling_path):
            return "Spilling path invalid"
        return "Successfully loaded all paths"

    #Return path of clip in queue, default is current
    def get_clip_path(self, displacement = 0) -> str:
        return self.todo_path + fr'\\{self.clip_list[self.queue_position + displacement]}'
    
    #Move queue up one
    def next(self) -> None:
        self.queue_position += 1

    #Returns the previous, current and next clip in a list
    def get_queue_list(self) -> list:
        return [self.clip_list[self.queue_position-1], self.clip_list[self.queue_position], self.clip_list[self.queue_position+1]]
    
    #Choose either plunging, unclear, or spilling and categorize the current clip by moving it to it's respective location
    def catagorize(self, catagory) -> None:
        try:
            src_vid = os.path.abspath(self.get_clip_path())
            src_name = src_vid[:-4]
            src = src_name + '.txt'
            vid_out = self.clip_list[self.queue_position]
            out_name = vid_out[:-4]
            out_file = out_name + '.txt'
            with open(self.log, 'a') as f:
                if catagory == "plunging":
                    f.write(f"{src} > {self.plunging_path}\\{out_file}\n")
                    f.write(f"{src_vid} > {self.plunging_path}\\{vid_out}\n")
                if catagory == "unclear":
                    f.write(f"{src} > {self.unclear_path}\\{out_file}\n")
                    f.write(f"{src_vid} > {self.unclear_path}\\{vid_out}\n")
                if catagory == "spilling":
                    f.write(f"{src} > {self.spilling_path}\\{out_file}\n")
                    f.write(f"{src_vid} > {self.spilling_path}\\{vid_out}\n")
                if catagory == "null":
                    f.write(f"{src} > delete\n")
                    f.write(f"{src_vid} > delete\n")
                f.close()
        except Exception as e:
            print(f"Failed with exception: {e}")