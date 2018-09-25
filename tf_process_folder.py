""" This example extract features for all signatures in a folder,
    using the CNN trained on the GPDS dataset. Results are saved in a matlab
    format.

    Usage: python process_folder.py <signatures_path> <save_path>
                                    <model_path> [canvas_size]

    Example:
    python process_folder.py signatures/ features/ models/signet.pkl

    This example will process all signatures in the "signatures" folder, using
    the SigNet model, and saving the resutls to the features folder

"""
from scipy.misc import imread
from preprocess.normalize import preprocess_signature
import tf_signet
from tf_cnn_model import TF_CNNModel
import tensorflow as tf
import numpy as np
import sys
import os
import scipy.io
from find_largest_image import find_largest

if len(sys.argv) not in [4, 6]:
    print('Usage: python process_folder.py <signatures_path> <save_path> '
          '<model_path> [canvas_size]')
    exit(1)

# Let's fix the code below later.
signatures_path = sys.argv[1]
save_path = sys.argv[2]
model_path = sys.argv[3]
if len(sys.argv) == 4:
    canvas_size = find_largest(signatures_path)  # Maximum signature size
else:
    canvas_size = (int(sys.argv[4]), int(sys.argv[5]))

print('Using model %s' % model_path)
print('Using canvas size: %s' % (canvas_size,))

# Load the model
model_weight_path = 'models/signet.pkl'
model = TF_CNNModel(tf_signet, model_weight_path)
person_counter=0
for root, dirs, filenames in os.walk(signatures_path, topdown=False):

    #Only work in directories ending with numbers.
    if (not root[-1].isdigit()) and not (root == signatures_path):
        continue

    # Go to a directory and collect image names in a list.
    image_list = list()
    print ("Working on " + (root))
    #image_root=root
    for file in filenames:
        if (file.endswith(".jpg")):
            image_list.append(file)

    #Extend the image names to full directory names
    full_file_name_list = [root + "/" + image for image in image_list]

    for full_file_name in full_file_name_list:

        if (full_file_name_list.index(full_file_name)==0):
            # Colors are flattened into grayscale layer
            image_ndarray = np.expand_dims(
                preprocess_signature(imread(full_file_name, flatten=True),canvas_size),axis=0)
        else:
            image_ndarray = np.append(arr=image_ndarray, values=
                np.expand_dims(preprocess_signature(imread(full_file_name, flatten=True), canvas_size), axis=0), axis=0)


    #original_list = [imread(full_file_name, flatten=True) for full_file_name in full_file_name_list]
    #processed_list = [preprocess_signature(original, canvas_size) for original in original_list]

    # Note: it there is a large number of signatures to process, it is faster to
    # process them in batches (i.e. use "get_feature_vector_multiple")
    # Use the CNN to extract features
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feature_vector = model.get_feature_vector_multiple(sess, image_ndarray)

    feature_save_folder_name = os.path.join(root, "extracted_features")
    # Create new folder if it does not already exist
    try:
        os.makedirs(feature_save_folder_name)
    except:
        pass

    #Save in the MATLAB format
    #name needs to be written according to list above
    person_counter+=1
    print str(person_counter) + " people are processed!"
    for iterator_1 in range (0,feature_vector.shape[0]):
        save_filename = os.path.join(feature_save_folder_name, os.path.splitext(image_list[iterator_1])[0] + '.mat')
        scipy.io.savemat(save_filename, {'feature_vector': feature_vector[iterator_1,:]})
else:
    pass
