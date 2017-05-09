from data_package import data_fns

#Make this folder if it doesnt exist. Path to save the pkled datasets and captions
save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'

#This is where the training images are
inpainting_root = '/usr/local/data/sejacob/lifeworld/data/inpainting/'
path = inpainting_root+'train2014/'
train_or_val = 'train'

data_fns.save_dataset(path, save_path, train_or_val)
data_fns.make_input_output_set(save_path,train_or_val)

#This is where the validation images are
path = inpainting_root+'val2014/'
train_or_val = 'val'

data_fns.save_dataset(path,save_path,train_or_val)
data_fns.make_input_output_set(save_path,train_or_val)

#Path to the trained google gensim model
#http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
gensimpath = inpainting_root+'word2vec/GoogleNews-vectors-negative300.bin'

trainpath = inpainting_root+'train2014/'
valpath =  inpainting_root+'val2014/'

#This is where the unprocessed catpions are
captionpath = inpainting_root+'dict_key_imgID_value_caps_train_and_valid.pkl'
pklpath = save_path


data_fns.captionsetup(gensimpath,trainpath,valpath,captionpath,pklpath)
