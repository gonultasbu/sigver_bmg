import os
import numpy as numpy
import pandas as pd  

def clean_mats(input_path):
    for root, dirs, filenames in os.walk(input_path, topdown=False):
        for file in dirs:
            if (file.endswith("extracted_features")):
                os.rmdir(os.path.join(root,file))


if __name__ == "__main__":
    clean_mats("/home/mert/Desktop/sigver_dataset/")
