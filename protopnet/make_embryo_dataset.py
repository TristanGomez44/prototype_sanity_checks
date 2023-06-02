import os,glob,sys

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import argparse
NO_ANNOT = -1

def preproc_annot(x):
    if x.isdigit():
        return int(x)
    else:
        return NO_ANNOT
  
def make_annot_dict(dataset_path,is_train):

    if is_train:
        annot_filename = "Gardner_train_silver.csv"
    else:
        annot_filename = "Gardner_test_gold_onlyGardnerScores.csv"

    annot_path = os.path.join(dataset_path,annot_filename)
    annot_csv = np.genfromtxt(annot_path,delimiter=";",dtype=str)

    dic = {}

    for row in annot_csv[1:]:
        sub_dic = {"exp":preproc_annot(row[1]),"icm":preproc_annot(row[2]),"te":preproc_annot(row[3])}
        #Verify dic 
        annot_nb = 0 
        for key in sub_dic:
            if sub_dic[key] != NO_ANNOT:
                annot_nb += 1 
        
        assert annot_nb>0,f"Image {row[0]} from dataset {annot_filename} has no annotation: found {annot_nb} annotation for {len(sub_dic.keys())} keys."

        dic[row[0]] = sub_dic

    return dic 

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, metavar='N',default="../../embryo/data/Blastocyst_Dataset")
    parser.add_argument('--dest_path', type=str, metavar='N',default='../data/blasto/dataset')
    parser.add_argument('--key', type=str, metavar='N',default='icm')

    args = parser.parse_args()

    args.dest_path = args.dest_path+"_"+args.key+"/"
    os.makedirs(args.dest_path,exist_ok=True)

    for is_train in [True,False]:
        
        annot_dict = make_annot_dict(args.dataset_path,is_train)
        image_list = sorted(annot_dict.keys())
        image_fold = os.path.join(args.dataset_path,"Images")

        subset_dest_path = os.path.join(args.dest_path,"train_cropped" if is_train else "test_cropped")
        os.makedirs(subset_dest_path,exist_ok=True)

        for image_name in image_list:

            image_path = os.path.join(image_fold,image_name)
            image = Image.open(image_path)

            annot = annot_dict[image_name][args.key]

            if annot != NO_ANNOT:
                
                #create folder if not exist
                class_fold = os.path.join(subset_dest_path,str(annot))

                os.makedirs(class_fold,exist_ok=True)

                #copy image 
                image_dest_path = os.path.join(class_fold,image_name)

                image.save(image_dest_path)

if __name__ == "__main__":
    main()

            

