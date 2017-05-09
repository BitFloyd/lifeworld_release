import os
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

from data_package import data_fns
from model_pkg import model_zoo

# Get data as numpy mem-map to not overload the RAM
print "GET_DATASETS"
save_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/pkls/'
dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')
shape_img = dset_train[0].shape
rows = shape_img[0]
cols = shape_img[1]
assert (rows == cols)
start = int(round(rows / 4))
end = int(round(rows * 3 / 4))
for i in range(0, len(dset_middle_empty_train)):
    dset_middle_empty_train[i][start:end, start:end, :] = 0.0
for i in range(0, len(dset_middle_empty_val)):
    dset_middle_empty_train[i][start:end, start:end, :] = 0.0

# EXPERIMENT 3
# --------------
# Matching with features instead of pixels.
# Idea : Features from reconstructed image should match the features from a pre-trained model


print "GET MODEL"
model_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/exp03_vgg_mp_T_feature_match.model'

es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=1)
checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00000001, verbose=1)

full_model = model_zoo.feature_comparison_model(shape_img, [64, 128, 256, 512], maxpool=False, highcap=False,
                                                op_only_middle=False)

if (os.path.isfile(model_filepath)):
    full_model = load_model(model_filepath)

feat_extract_model = VGG16(include_top=False, weights='imagenet', input_shape=shape_img)
x = feat_extract_model.output
x = model_zoo.Flatten()(x)
feature_extraction_model_flat = model_zoo.Model(input=feat_extract_model.input, output=x)

#
# print "GET VGG FEATURES"
# train_feats = feature_extraction_model_flat.predict(dset_train, batch_size=200)

# print "START FIT"
# history = full_model.fit(dset_middle_empty_train, train_feats,
#                          batch_size=200, nb_epoch=400,
#                          callbacks=[es, checkpoint, reduce_lr],
#                          validation_split=0.3,
#                          shuffle=True)
layer_index = 0
for i in range(2, len(full_model.layers)):
    layer_shape = full_model.layers[i].output_shape
    if (layer_shape[1:] == shape_img):
        layer_index = i
        break

test_model = model_zoo.Model(input=full_model.input,
                             output=full_model.layers[layer_index].get_output_at(0))

print "MAKE PREDICTIONS"
predictions = test_model.predict(dset_middle_empty_val, batch_size=100)

print "SAVE IMAGES"
write_path = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/exp03_vgg_mp_T_feature_match/'
data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)
