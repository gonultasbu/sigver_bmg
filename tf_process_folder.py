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

if len(sys.argv) not in [4,6]:
    print('Usage: python process_folder.py <signatures_path> <save_path> '
          '<model_path> [canvas_size]')
    exit(1)

#Let's fix the code below later.
signatures_path = sys.argv[1]
save_path = sys.argv[2]
model_path = sys.argv[3]
if len(sys.argv) == 4:
    canvas_size = (1338, 2973)  # Maximum signature size
else:
    canvas_size = (int(sys.argv[4]), int(sys.argv[5]))

print('Using model %s' % model_path)
print('Using canvas size: %s' % (canvas_size,))

# Load the model
model_weight_path = 'models/signet.pkl'
model = TF_CNNModel(tf_signet, model_weight_path)

# Note: it there is a large number of signatures to process, it is faster to
# process them in batches (i.e. use "get_feature_vector_multiple")

for root, dirs, filenames in os.walk(signatures_path, topdown=False):
    # Load and pre-process the signature
    print ("Working on " + (root))
    for file in filenames:
        if (file.endswith(".jpg")):
            full_file_name=os.path.join(root, file)
            original = imread(full_file_name, flatten=True)
            processed = preprocess_signature(original, canvas_size)

            # Use the CNN to extract features
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            feature_vector = model.get_feature_vector(sess,processed)

            # Save in the matlab format
            feature_save_folder_name=os.path.join(root,"extracted_features")

            # Create new folder if it does not already exist
            try:
                os.makedirs(feature_save_folder_name)
            except:
                pass

            save_filename = os.path.join(feature_save_folder_name, os.path.splitext(file)[0] + '.mat')
            scipy.io.savemat(save_filename, {'feature_vector':feature_vector})
        else:
            pass
