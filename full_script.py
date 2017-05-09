from data_package import data_fns
from exec_functions import execute

# Make this folder if it doesnt exist. Path to save the pkled datasets and captions
inpainting_root = '/usr/local/data/sejacob/lifeworld_release/data/inpainting/'
save_path = inpainting_root + 'pkls/'

# This is where the training images are
path = inpainting_root + 'train2014/'
train_or_val = 'train'

data_fns.save_dataset(path, save_path, train_or_val)
data_fns.make_input_output_set(save_path, train_or_val)

# This is where the validation images are
path = inpainting_root + 'val2014/'
train_or_val = 'val'

data_fns.save_dataset(path, save_path, train_or_val)
data_fns.make_input_output_set(save_path, train_or_val)

# Path to the trained google gensim model
# http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
gensimpath = inpainting_root+'word2vec/GoogleNews-vectors-negative300.bin'

trainpath = inpainting_root+'train2014/'
valpath = inpainting_root+'val2014/'

# This is where the unprocessed captions are
captionpath = inpainting_root+'dict_key_imgID_value_caps_train_and_valid.pkl'
pklpath = save_path

data_fns.captionsetup(gensimpath, trainpath, valpath, captionpath, pklpath)

# This is where the training losses will be plotted
plt_path = inpainting_root+'predictions/dcg_losses.png'

# Predictions made on one batch of the training set after each 100th iteration
wr_path = inpainting_root+'predictions/train_test_run_train_batch'

# Path to the directory where predictions will be made on the whole test and train set after each run
wr_predictions_path = inpainting_root+'predictions/train_test_run'

# Path to the pkl directory where datasets are stored
pkl_path = save_path

# Path to the directory where the model will be saved
model_path = inpainting_root+'models/'

execute.DCGAN_captions_LSTM_train_both(plt_path, wr_path, pkl_path, model_path, wr_predictions_path, 20, 1)
