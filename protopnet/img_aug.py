import Augmentor
import os
import glob

dir = '../data/blasto/dataset_icm/train_cropped/'
target_dir = '../../train_cropped_augmented/'
os.makedirs('../data/blasto/dataset_icm/train_cropped_augmented', exist_ok=True)
folders = glob.glob(dir + '/*')
target_folders = [folder.replace("train_cropped","train_cropped_augmented") for folder in glob.glob(dir + '/*')]

print(folders, target_folders)

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
    # rotation
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # shear
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p