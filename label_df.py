import pandas as pd  
import numpy as np  
import os  

def detect_types(path):
    work_csv = os.path.join(path,"extracted_features.csv")
    sig_df = pd.read_csv(work_csv)
    return 0




if __name__ == "__main__":
    path = "/home/mert/Desktop/sigver_dataset/GPDSSyntheticSignatures4k"
    detect_types(path)