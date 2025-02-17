#counts the number of input files for each class

import os

dir = r'.\\val'   #train or val
text_list = []
for filename in os.listdir(dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(dir, filename)
        with open(filepath, "r") as file:
            text_list.append(file.read(1))

print(text_list.count('0'))  #prebreaking
print(text_list.count('1'))  #curling
print(text_list.count('2'))  #splashing
print(text_list.count('3'))  #whitewash
print(text_list.count('4'))  #crumbling
