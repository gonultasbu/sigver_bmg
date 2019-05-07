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

def tf_process(signatures_path, model_path):

    canvas_size = [1338, 2973] 
    print('Using model %s' % model_path)
    print('Using canvas size: %s' % (canvas_size,))
    sig_df = pd.DataFrame()
    # Load the model
    model_weight_path = model_path
    model = TF_CNNModel(tf_signet, model_weight_path)
    image_list = list()
    person_list = list()
    sig_num_list = list()
    fakeness_list = list()
    for root, dirs, filenames in tqdm.tqdm(os.walk(signatures_path, topdown=False)):
        if (not root[-1].isdigit()) and not (root == signatures_path) : continue
        for file in filenames:
            if (file.endswith(".jpg")):
                image_list.append(os.path.join(root,file))
                person_list.append(int(root.split('\\')[-1]))
                sig_num_list.append(int(file.split('-')[2].split('.')[0]))
                if (file.startswith('cf')) : fakeness_list.append(True)
                elif (file.startswith('c')) : fakeness_list.append(False)
                else:
                    print("invalid character found in image file, quitting!")
                    quit() 
    image_list = np.asarray(image_list)
    person_list = np.asarray(person_list, dtype='uint16')
    sig_num_list = np.asarray(sig_num_list, dtype='uint16')
    fakeness_list = np.asarray(fakeness_list, dtype='bool_')
    image_list = np.expand_dims(image_list, axis=1)
    person_list = np.expand_dims(person_list, axis=1)
    sig_num_list = np.expand_dims(sig_num_list, axis=1)
    fakeness_list = np.expand_dims(fakeness_list, axis=1)
    image_ndarray = np.zeros((len(image_list), 150, 220),dtype='uint8')
    data_ndarray = np.hstack((image_list, person_list, sig_num_list, fakeness_list))
    for counter, full_file_name in enumerate(tqdm.tqdm(image_list[:,0])):
        image_ndarray[counter,:,:] = np.expand_dims(
                preprocess_signature(imread(full_file_name, flatten=True),canvas_size),axis=0)
        # if counter > 3 : break
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    splitter_coeff = 500
    i_slices = np.split(image_ndarray,splitter_coeff)
    vf_slices = np.split(np.zeros((len(image_list),2048)),splitter_coeff)
    del image_ndarray, image_list, person_list, sig_num_list, fakeness_list

    for c_var, nd_slice in enumerate(tqdm.tqdm(i_slices)):
        feature_vector = model.get_feature_vector_multiple(sess, nd_slice)
        vf_slices[c_var] = feature_vector
    sig_v_df = pd.DataFrame(np.vstack(vf_slices))

    del i_slices, vf_slices
    sig_d_df = pd.DataFrame(data_ndarray)
    sig_d_df.rename(columns={'0': 'path', '1': 'person', '2':'sig_num', '3':'fakeness'}, inplace=True)
    sig_d_df.to_csv(os.path.join(signatures_path,"data_features.csv"), index=False) 
    sig_v_df.to_csv(os.path.join(signatures_path,"visual_features.csv"), index=False)
if (__name__ == "__main__"):
    tf_process("C:\\Users\\Mert\\Documents\\GitHub\\sigver_bmg\\data\\GPDSSyntheticSignatures4k","C:\\Users\\Mert\\Documents\\GitHub\\sigver_bmg\\models\\signet.pkl")
