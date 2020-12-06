

"""
    Version 1 2019-06-20 Abhinav Kumar
"""

import os
import glob
import subprocess
import json

SEP = " "

def grab_files(folder_full_path, EXTENSIONS):
    """
        Grabs files in folder with the given extensions
        Returns- list
    """
    files_grabbed = []
    for j in range(len(EXTENSIONS)):
        key = os.path.join(folder_full_path, "*"+EXTENSIONS[j])
        files_grabbed.extend(glob.glob(key))

    return files_grabbed

def copy_files(list_of_files, output_folder_path):
    """
        Copies a list of files to the output folder path
    """
    num_files = len(list_of_files)

   # Iterate over all files
    for j in range(num_files):
        filename = list_of_files[j]

        # Copy the files to new location
        subprocess.call(["cp", filename,  output_folder_path])

def create_folder(output_folder_path):
    """
        Tries to create a folder if folder is not present
    """
    if os.path.exists(output_folder_path):
        print("Directory exists {}".format(output_folder_path))
    else:
        print("Creating directory {}".format(output_folder_path))
        os.makedirs(output_folder_path)

def write_row_to_file(landmark_per_image, output_file_path, num_points= 14, transposed= False):
    """
        Writes row of data to output_file_path in format of 300W
    """
    with open(output_file_path, 'w') as fw:
        fw.write("version: 1\n")
        fw.write("n_points: "+ str(num_points) + "\n")
        fw.write("{\n")

        for i in range(num_points):
            if transposed:
                fw.write(str(landmark_per_image[i, 0]) + SEP + str(landmark_per_image[i, 1]) + "\n")
            else:
                fw.write(str(landmark_per_image[0, i]) + SEP + str(landmark_per_image[1, i]) + "\n")

        fw.write("}\n")

def write_data_to_json(data, output_file_path):
    if len(data) == 0:
        return

    print("\nWriting to {}".format(output_file_path))
    with open(output_file_path, 'w') as outfile:
        json.dump(data, outfile)

def execute(command, print_flag= False):
    if print_flag:
        print(command)
    os.system(command)
