""" This example extract features for all signatures in a folder,
    using the CNN trained on the GPDS dataset. Results are saved in a matlab
    format.

    Usage: python process_folder.py <signatures_path> <save_path>
                                    <model_path> [canvas_size]

    Example:
    python process_folder.py signatures/ features/ models/signet.pkl

    This example will process all signatures in the "signatures" folder, using
    the SigNet model, and saving the results to the features folder

"""
#%%

from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import tf_signet
from tf_cnn_model import TF_CNNModel
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
import scipy.io
from find_largest_image import find_largest
import tqdm 
import dask.dataframe as dd

def tf_process(signatures_path, model_path):

    canvas_size = [2078, 3307] 
    print('Using model %s' % model_path)
    print('Using canvas size: %s' % (canvas_size,))
    sig_df = pd.DataFrame()
    # Load the model
    model_weight_path = 'models/signet.pkl'
    model = TF_CNNModel(tf_signet, model_weight_path)
    image_list = list()
    for root, dirs, filenames in os.walk(signatures_path, topdown=False):
        if (not root[-1].isdigit()) and not (root == signatures_path):
            continue
        print ("Working on " + (root))
        for file in filenames:
            if (file.endswith(".jpg")):
                image_list.append(os.path.join(root,file))

    image_ndarray = np.zeros((len(image_list), 150, 220),dtype='uint8')
    for counter, full_file_name in enumerate(tqdm.tqdm(image_list)):
        image_ndarray[counter,:,:] = np.expand_dims(
                preprocess_signature(imread(full_file_name, flatten=True),canvas_size),axis=0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    image_list = np.array(image_list)
    image_list = np.expand_dims(image_list,axis=1)

    i_slices = np.split(image_ndarray,500)
    f_slices = np.split(image_list,500)

    for c_var, nd_slice in enumerate(tqdm.tqdm(i_slices)):
        feature_vector = model.get_feature_vector_multiple(sess, nd_slice)
        temp_df = pd.DataFrame(np.hstack((f_slices[c_var],feature_vector)))
        sig_df = sig_df.append(temp_df, ignore_index=True)

    

    sig_df.to_csv(os.path.join(signatures_path,"extracted_features.csv"))



if (__name__ == "__main__"):
    tf_process("/home/mert/Desktop/sigver_dataset/GPDSSyntheticSignatures4k/","models/signet.pkl")
