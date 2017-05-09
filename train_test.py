from exec_functions import execute

inpainting_root = '/usr/local/data/sejacob/lifeworld/data/inpainting/'
# This is where the training losses will be plotted
plt_path = inpainting_root + 'predictions/dcg_losses.png'

# Predictions made on one batch of the training set after each 100th iteration
wr_path = inpainting_root + 'predictions/train_test_run_train_batch'

# Path to the directory where predictions will be made on the whole test and train set after each run
wr_predictions_path = inpainting_root + 'predictions/train_test_run'

# Path to the pkl directory where datasets are stored
pkl_path = inpainting_root + 'pkls/'

# Path to the directory where the model will be saved
model_path = inpainting_root + 'models/'

execute.DCGAN_captions_LSTM_train_both(plt_path, wr_path, pkl_path, model_path, wr_predictions_path, 20, 1)
