import cPickle as pkl
from tempfile import mkdtemp
import cv2
import gensim
import numpy as np
import os
import string
from tempfile import mkdtemp
import os
import gensim


def save_dataset(path, save_path, train_or_val='train'):
    filelist = sorted(os.listdir(path))
    img = cv2.imread(path + filelist[0], cv2.IMREAD_COLOR)
    shape_img = img.shape
    save_path = save_path + train_or_val + '_dset.npy'
    fname = os.path.join(mkdtemp(), 'newfile.dat')
    dset = np.memmap(fname, dtype='float32', mode='w+',
                     shape=(len(filelist), shape_img[0], shape_img[1], shape_img[2]))

    for i in range(0, len(filelist)):
        filename = path + filelist[i]
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        norm_image = cv2.normalize(image, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        dset[i] = norm_image

    np.save(save_path, dset)
    print ("Save dataset finished")
    return True


def load_dataset(load_path):
    dset = np.load(load_path, mmap_mode='r+')
    return dset


def make_input_output_set(save_path, train_or_val='train'):
    dset_path = save_path + train_or_val + '_dset.npy'
    dset = load_dataset(dset_path)
    img = dset[0]
    assert (img.shape[0] == img.shape[1])
    len_side = img.shape[0]
    start = int(round(len_side / 4))
    end = int(round(len_side * 3 / 4))
    shape_middle = (dset.shape[0], int(dset.shape[1] / 2), int(dset.shape[2] / 2), dset.shape[3])

    save_path_middle = save_path + train_or_val + '_dset_middle.npy'
    save_path_middle_empty = save_path + train_or_val + '_dset_middle_empty.npy'

    fname1 = os.path.join(mkdtemp(), 'newfile2.dat')
    fname2 = os.path.join(mkdtemp(), 'newfile3.dat')

    dset_middle = np.memmap(fname1, dtype='float32', mode='w+', shape=shape_middle)
    dset_empty_middle = np.memmap(fname2, dtype='float32', mode='w+', shape=dset.shape)

    for i in range(0, len(dset)):
        dset_empty_middle[i] = dset[i]
        img = dset[i]
        middle_of_img = img[start:end, start:end, :]
        dset_empty_middle[i][start:end, start:end, :] = 0.0
        dset_middle[i] = middle_of_img

    np.save(save_path_middle, dset_middle)
    np.save(save_path_middle_empty, dset_empty_middle)
    print ("Make I/O set finished")
    return True


def get_datasets(save_path, train_or_val='train'):
    dset_path = save_path + train_or_val + '_dset.npy'
    save_path_middle = save_path + train_or_val + '_dset_middle.npy'
    save_path_middle_empty = save_path + train_or_val + '_dset_middle_empty.npy'

    dset = np.load(dset_path, mmap_mode='r+')
    dset_middle = np.load(save_path_middle, mmap_mode='r+')
    dset_middle_empty = np.load(save_path_middle_empty, mmap_mode='r+')

    return dset, dset_middle, dset_middle_empty


def write_predicted(test_set, test_set_middle_empty, predictions, write_path, middle_only=True):
    rows = test_set[0].shape[0]
    cols = test_set[0].shape[1]
    channels = test_set[0].shape[2]

    assert (rows == cols)

    image = np.zeros((rows, cols * 3 + 20, channels))
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))
    assert (len(test_set) == len(test_set_middle_empty) == len(predictions))

    for i in range(0, len(test_set)):
        image = np.zeros((rows, cols * 3 + 20, channels))
        image[:, 0:cols, :] = test_set[i]
        test_set_middle_empty[i][start:end, start:end, :] = 0.0
        image[:, cols + 10:cols * 2 + 10, :] = test_set_middle_empty[i]
        if (middle_only):
            middle_filled_image = test_set_middle_empty[i]
            middle_filled_image[start:end, start:end, :] = predictions[i]
            image[:, cols * 2 + 20:cols * 3 + 20, :] = middle_filled_image
        else:
            image[:, cols * 2 + 20:cols * 3 + 20, :] = predictions[i]

        # filename
        filename = write_path + str(i).zfill(len(str(len(test_set)))) + '.jpg'
        # imwrite
        image = ((image + 1) * (255 / 2)).astype('uint8')
        cv2.imwrite(filename, image)


def captionsetup(gensimpath,trainpath,valpath,captionpath,pklpath):
    model = gensim.models.KeyedVectors.load_word2vec_format(gensimpath,binary=True)

    with open(captionpath) as f:
        dict = pkl.load(f)

    keys = dict.keys()

    max = 0

    for i in range(0, len(dict)):

        for j in range(0, len(dict[keys[i]])):
            sen = dict[keys[i]][j]
            # Remove punctuation
            sen = sen.translate(None, string.punctuation)
            dict[keys[i]][j] = sen

            len_sen = len(sen.split())
            if (len_sen > max):
                max = len_sen

    print "SAVE TRAIN STUFF"
    filelist = sorted(os.listdir(trainpath))

    print max + 1

    fname = os.path.join(mkdtemp(), 'newfile.dat')
    dset_train = np.memmap(fname, dtype='float32', mode='w+', shape=(len(filelist), 5, max + 1, 300))

    for i in range(0, len(filelist)):
        key = filelist[i][:-4]
        for j in range(0, len(dict[key])):
            if (j > 4):
                continue
            l = dict[key][j].split()
            m = 0
            for k in range(0, len(l)):
                # print l[k]
                try:
                    vector = model[l[k]]
                except KeyError:
                    continue
                dset_train[i, j, m, :] = vector
                m += 1

    np.save(pklpath+'all_words_2_vectors_train.npy', dset_train)

    print "SAVE VAL STUFF"
    filelist = sorted(os.listdir(valpath))

    fname = os.path.join(mkdtemp(), 'newfile.dat')
    dset_val = np.memmap(fname, dtype='float32', mode='w+', shape=(len(filelist), 5, max + 1, 300))

    for i in range(0, len(filelist)):
        key = filelist[i][:-4]
        for j in range(0, len(dict[key])):
            if (j > 4):
                continue
            l = dict[key][j].split()
            m = 0
            for k in range(0, len(l)):
                # print l[k]
                try:
                    vector = model[l[k]]
                except KeyError:
                    continue
                dset_train[i, j, m, :] = vector
                m += 1

    np.save(pklpath+'all_words_2_vectors_val.npy', dset_val)