import numpy as np 
import pandas as pd 

df=pd.read_csv("/home/mert/Desktop/sigver_dataset/GPDSSyntheticSignatures4k/extracted_features.csv")
df.rename(columns={'0': 'path', '1': 'person', '2':'sig_num', '3':'fakeness'}, inplace=True)
df.to_csv("/home/mert/Desktop/sigver_dataset/GPDSSyntheticSignatures4k/extracted_features.csv")