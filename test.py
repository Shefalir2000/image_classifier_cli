import shutil

import os


def count_files(data_path, count):
    for path in os.listdir(data_path):
        if os.path.isdir( data_path + '/' +path):
            count = count_files(data_path + '/' + path, count)
        elif not "jpeg" in path and not "jpg" in path and not "jfif" in path and not "pjpeg" in path and not "pjp" in path and not "png" in path and not "svg" in path and not "webp" in path:
            return count 
        elif os.path.isfile(data_path + '/' + path):
            count = count + 1
    return count 

print(count_files("testing_data/training_set", 0))