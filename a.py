import shutil
import os
all_data = [os.path.join('data_new', i) for i in os.listdir('data_new') if i.endswith('.png')]
for i in range(10000):
    shutil.copy(all_data[i], 'data_1w/')
    shutil.copy(all_data[i].replace('.png', '.txt'), 'data_1w/') 
    print('1W: {}/{}'.format(i+1, 10000))
for i in range(100000):
    shutil.copy(all_data[i], 'data_10w/') 
    shutil.copy(all_data[i].replace('.png', '.txt'), 'data_10w/')
    print('10W: {}/{}'.format(i+1, 100000))
