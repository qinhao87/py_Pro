import os

def save_images_names_to_file(folder_path,output_file):
    with open(output_file, 'w') as file:
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                file.write(filename+'\n')

save_images_names_to_file('/home/hao/py_Pro/edges','image_names.lst')