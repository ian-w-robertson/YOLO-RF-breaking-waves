# analyze labels from videos
# written by Ian Robertson (ARL at UH)

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join

#inputs
#home_folder = '.\\Classify_YOLO_RF_2'
data_folder  = r'.\\results_by_condition\\'
results_folder = data_folder
test_case = 'H025T36irr'

def analyze_breaking_func(results_folder, test_case):

    # find directories of files that have the wave conitions specified in test_case
    dir_list = []
    for root, dirs, files in os.walk(results_folder):
        for dir in dirs:
            if test_case in dir:
                dir_list.append(os.path.join(root, dir))

    dir_list_edit = dir_list[:] # exclude duplicate cases if desired

    # get list of results.txt files for desired wave cases
    file_list = []
    for dir in dir_list_edit:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('esults.txt'):
                    file_list.append(os.path.join(root, file)) # get directory list of txt files 

    # make a master list of all data
    df_list = []
    for file in file_list:
        data = pd.read_fwf(file) # read in results text file
        df_list.append(data) # append to list

    master_list = pd.concat(df_list, ignore_index = True) #concatenate lists
    master_list.to_csv(results_folder + test_case + "_combined_results.csv", index=False) #save combined list

    #extract layout numbers from filenames
    names = master_list.name.tolist()
    layout_num = [name[0:3] for name in names]
    layout_0_idx = [i for i, x in enumerate(layout_num) if x == '000']
    layout_1_idx = [i for i, x in enumerate(layout_num) if x == '01b']
    layout_3_idx = [i for i, x in enumerate(layout_num) if x == '003']
    layout_4_idx = [i for i, x in enumerate(layout_num) if x == '004']

    # make list of label values for each filename
    layout_0_values = master_list.value[layout_0_idx].tolist()
    layout_1_values = master_list.value[layout_1_idx].tolist()
    layout_3_values = master_list.value[layout_3_idx].tolist()
    layout_4_values = master_list.value[layout_4_idx].tolist()

    # tally up 
    layout_0_plunging = layout_0_values.count(0.0)
    layout_0_spilling = layout_0_values.count(1.0)

    layout_1_plunging = layout_1_values.count(0.0)
    layout_1_spilling = layout_1_values.count(1.0)

    layout_3_plunging = layout_3_values.count(0.0)
    layout_3_spilling = layout_3_values.count(1.0)

    layout_4_plunging = layout_4_values.count(0.0)
    layout_4_spilling = layout_4_values.count(1.0)

    layout_0_total = layout_0_plunging+layout_0_spilling
    layout_1_total = layout_1_plunging+layout_1_spilling
    layout_3_total = layout_3_plunging+layout_3_spilling
    layout_4_total = layout_4_plunging+layout_4_spilling

    if layout_1_total == 0:
        layout_1_total = 1

    #save results
    d = {'Layout': ["Empty Flume", "Layout 1", "Layout 3", "Layout 4"], 
         'Plunging': [layout_0_plunging, layout_1_plunging, layout_3_plunging, layout_4_plunging], 
         'Spilling': [layout_0_spilling, layout_1_spilling, layout_3_spilling, layout_4_spilling] }
    
    df = pd.DataFrame(d)
    df.to_csv(results_folder + test_case + "_breaker_stats.csv", index=False)

    # plot results

    layouts = ("Empty Flume", "Layout 1", "Layout 3", "Layout 4")
    breaker_type = {
        'Plunging': (layout_0_plunging, layout_1_plunging, layout_3_plunging, layout_4_plunging),
        'Spilling': (layout_0_spilling, layout_1_spilling, layout_3_spilling, layout_4_spilling),
    }

    breaker_type_normalized = {
        'Plunging': (round(layout_0_plunging/layout_0_total,2), round(layout_1_plunging/layout_1_total,2), 
                     round(layout_3_plunging/layout_3_total,2), round(layout_4_plunging/layout_4_total,2)),
        'Spilling': (round(layout_0_spilling/layout_0_total,2), round(layout_1_spilling/layout_1_total,2), 
                     round(layout_3_spilling/layout_3_total,2), round(layout_4_spilling/layout_4_total,2)),
    }

    x = np.arange(len(layouts))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    max_y = max(layout_0_plunging, layout_1_plunging, layout_3_plunging, layout_4_plunging, 
                layout_0_spilling, layout_1_spilling, layout_3_spilling, layout_4_spilling)

    fig, ax = plt.subplots(layout='constrained')

    # plot results
    for attribute, measurement in breaker_type.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Occurences')
    ax.set_xticks(x + width/2, layouts)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.1*max_y)
    plt.title(test_case)

    plt.savefig(results_folder + test_case + "_breaker_stats_plot.png")
    plt.show()

    # plot normalized results
    x = np.arange(len(layouts))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in breaker_type_normalized.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percent Occurence')
    ax.set_xticks(x + width/2, layouts)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)
    plt.title(test_case)

    plt.savefig(results_folder + test_case + "_breaker_stats__norm_plot.png")
    plt.show()

def analyze_breaking_func_single(results_folder, test_case):

    # find directories of files that have the wave conitions specified in test_case
    dir_list = []
    for root, dirs, files in os.walk(results_folder):
        for dir in dirs:
            if test_case in dir:
                dir_list.append(os.path.join(root, dir))

    dir_list_edit = dir_list[:] # exclude duplicate cases if desired

    # get list of results.txt files for desired wave cases
    file_list = []
    for dir in dir_list_edit:
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith('esults.txt'):
                    file_list.append(os.path.join(root, file)) # get directory list of txt files 

    # make a master list of all data
    df_list = []
    for file in file_list:
        data = pd.read_fwf(file) # read in results text file
        df_list.append(data) # append to list

    master_list = pd.concat(df_list, ignore_index = True) #concatenate lists
    master_list.to_csv(results_folder + test_case + "_results.csv", index=False) #save combined list



    # tally up 
    counts = master_list['value'].value_counts()
    plunging_count = counts[0.0]
    spilling_count = counts[1.0]

    total_count = spilling_count + plunging_count

    plunging_norm = round(plunging_count/total_count,2)
    spilling_norm = round(spilling_count/total_count,2)
  
    if total_count == 0:
        total_count = 1




    max_y = max(plunging_count, spilling_count)
    fig, ax = plt.subplots(layout='constrained')

    barplot1 = plt.bar(['Plunging', 'Spilling'], [plunging_count,spilling_count])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Occurences')
    ax.set_ylim(0, 1.1*max_y)
    plt.title(test_case)
    ax.bar_label(barplot1)

    plt.savefig(results_folder + test_case + "_breaker_stats_plot.png")
    plt.show()



    fig, ax = plt.subplots(layout='constrained')
    barplot2 = plt.bar(['Plunging', 'Spilling'], [plunging_norm,spilling_norm])
    ax.set_ylim(0, 1)
    plt.title(test_case)
    ax.bar_label(barplot2)

    plt.savefig(results_folder + test_case + "_breaker_stats_norm_plot.png")
    plt.show()

#run script     
if __name__=='__main__':
    analyze_breaking_func(results_folder, test_case)