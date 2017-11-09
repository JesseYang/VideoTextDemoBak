import os
from scipy import misc
import json
import shutil
import numpy as np
import numpy.ma as ma
import random

from demo_neo import segment_lines, extract_lines

quit()
raw_data_dir = 'output'
target_data_dir = 'data_new'
dataset_name = 'data_new'

# clear
if os.path.exists(target_data_dir):
    shutil.rmtree(target_data_dir)
os.mkdir(target_data_dir)

# collect raw data
raw_data_path_collections = []
files = os.listdir(raw_data_dir)
for file in files:
    if file.endswith('txt'):
        shutil.copy(os.path.join(raw_data_dir, file), os.path.join(target_data_dir, file))
        continue
    name = file
    raw_data_path = os.path.join(raw_data_dir, name)
    raw_data_path_collections.append(raw_data_path)
num_raw_data = len(raw_data_path_collections)
print('Number of raw data:', num_raw_data)
target_data_path_collections = []
for idx, each_raw_data_path in enumerate(raw_data_path_collections):
    mask = 0
    target_path = os.path.join(target_data_dir, each_raw_data_path.split('/')[-1])
    misc.imsave(target_path, img)
    target_data_path_collections.append(target_path+'\n')
    print(idx, num_raw_data, sep='/')

# split into training set and test set
test_ratio = 0.1
total_num = len(target_data_path_collections)
test_num = int(test_ratio * total_num)
train_num = total_num - test_num
train_records = target_data_path_collections[0:train_num]
test_records = target_data_path_collections[train_num:]

# save to text file
all_out_file = open(dataset_name + '_all.txt', 'w')
for record in target_data_path_collections:
    all_out_file.write(record)
all_out_file.close()

train_out_file = open(dataset_name + '_train.txt', 'w')
for record in train_records:
    train_out_file.write(record)
train_out_file.close()

test_out_file = open(dataset_name + '_test.txt', 'w')
for record in test_records:
    test_out_file.write(record)
test_out_file.close()

