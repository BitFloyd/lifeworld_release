import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import shuffle
from tqdm import tqdm
from data_package import data_fns
from model_pkg import model_zoo


def plot_loss(losses, plt_path):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(losses["df"], label='disc-full loss')
    plt.plot(losses["dh"], label='disc-half loss')
    plt.plot(losses["g"], label='generative loss')
    plt.plot(losses["m"], label='ms/ssim loss')
    plt.legend()
    plt.savefig(plt_path)
    plt.close(fig)


def train_for_both_captions(dset_train, dset_middle_empty_train, train_captions, adversary_model_full,
                            adversary_model_half, gan_merged_model,
                            full_model, losses, plt_path, wr_path,
                            nb_epoch=50000, plt_frq=100, batch_size=50, predict_frq=1000, run_num=10, m_train=False,
                            cp=0.1):
    rand_disc_train = np.random.randint(0, nb_epoch, size=int(0.01 * nb_epoch))
    for e in tqdm(range(nb_epoch)):
        corrupt = int(cp * batch_size)
        rand_index = np.random.randint(0, dset_train.shape[0], size=batch_size)
        train_batch = dset_train[rand_index]
        middle_empty_train_batch = dset_middle_empty_train[rand_index]
        rand_caption_idx = np.random.randint(0, train_captions.shape[1], len(rand_index))
        captions_train_batch = train_captions[rand_index, rand_caption_idx]

        predictions = gan_merged_model.predict(
            [middle_empty_train_batch, captions_train_batch],
            batch_size=batch_size, verbose=False)

        dset_max = np.vstack((train_batch, predictions))

        # make soft labels with label smoothing
        # False = 0.0-0.3
        # True = 0.7-1.2

        dset_real_false = np.random.rand((predictions.shape[0] + train_batch.shape[0]), 1) * 0.3

        dset_real_false[0:train_batch.shape[0]] = np.random.rand(train_batch.shape[0], 1) * 0.5 + 0.7

        # corrupt labels with low probability for discriminator training.
        # i.e, 10% of batch size, make true labels as false and false labels as true.
        rand_index_label_corrupt = np.random.randint(0, train_batch.shape[0], corrupt)
        dset_real_false[rand_index_label_corrupt] = np.random.rand(len(rand_index_label_corrupt), 1) * 0
        rand_index_label_corrupt = np.random.randint(train_batch.shape[0], dset_real_false.shape[0], corrupt)
        dset_real_false[rand_index_label_corrupt] = np.random.rand(len(rand_index_label_corrupt), 1) * 0.5 + 0.7

        dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

        model_zoo.make_trainable(adversary_model_full, True)
        adversary_loss_full = adversary_model_full.train_on_batch(dset_max, dset_real_false)

        model_zoo.make_trainable(adversary_model_half, True)
        adversary_loss_half = adversary_model_half.train_on_batch(dset_max, dset_real_false)

        losses["df"].append(adversary_loss_full[0])
        losses["dh"].append(adversary_loss_half[0])

        model_zoo.make_trainable(adversary_model_full, False)
        model_zoo.make_trainable(adversary_model_half, False)

        if e in rand_disc_train:
            # randomly train discriminator more twice for stability
            losses["g"].append(losses["g"][-1])
            losses["m"].append(losses["m"][-1])
            continue

        generator_labels = np.random.rand(middle_empty_train_batch.shape[0], 1) * 0.5 + 0.7
        rand_index_label_corrupt = np.random.randint(0, middle_empty_train_batch.shape[0], corrupt)
        generator_labels[rand_index_label_corrupt] = np.random.rand(len(rand_index_label_corrupt), 1) * 0.3

        g_loss = full_model.train_on_batch([middle_empty_train_batch, captions_train_batch],
                                           [generator_labels, generator_labels])
        losses["g"].append(g_loss[0])

        if (m_train):
            m_loss = gan_merged_model.train_on_batch(
                [middle_empty_train_batch, captions_train_batch], train_batch)
            losses["m"].append(m_loss)

        else:
            losses["m"].append(0.0)

        if e % plt_frq == plt_frq - 1:
            plot_loss(losses, plt_path)

        if e % predict_frq == predict_frq - 1:
            rand_index = np.asarray(range(0, batch_size))
            train_batch = dset_train[rand_index]
            middle_empty_train_batch = dset_middle_empty_train[rand_index]
            rand_caption_idx = np.random.randint(0, train_captions.shape[1], len(rand_index))
            captions_train_batch = train_captions[rand_index, rand_caption_idx]
            predictions = gan_merged_model.predict(
                [middle_empty_train_batch, captions_train_batch], batch_size=100)
            write_path = wr_path + str(run_num + 1).zfill(2) + '_' + str(e) + '/'

            if not os.path.exists(write_path):
                os.makedirs(write_path)
            data_fns.write_predicted(train_batch, middle_empty_train_batch, predictions, write_path, middle_only=False)


def DCGAN_captions_LSTM_train_both(plt_path, wr_path, pkl_path, model_path, wr_predictions_path, num_runs=10, n_epoch=1):
    # Get data as numpy mem-map to not overload the RAM
    plt.ioff()
    print "GET_DATASETS"
    save_path = pkl_path
    dset_train, dset_middle_train, dset_middle_empty_train = data_fns.get_datasets(save_path, 'train')
    dset_val, dset_middle_val, dset_middle_empty_val = data_fns.get_datasets(save_path, 'val')

    del (dset_middle_val)
    del (dset_middle_train)

    shape_img = dset_train[0].shape
    rows = shape_img[0]
    cols = shape_img[1]
    assert (rows == cols)
    start = int(round(rows / 4))
    end = int(round(rows * 3 / 4))

    # EXPERIMENT 8
    # --------------
    # DC-GAN WITH ALL THE GAN-HACKS and inception modules for inpainting
    # Takes captions as well.
    # For each batch training, any 1 out of the 5 captions are passed
    # For prediction, any of the captions may be used.
    # Train discriminator on the middle and the full.

    # dset_middle_train_p = np.load(pkl_path + 'dset_middle_train_p.npy')
    # dset_middle_val_p = np.load(pkl_path + 'dset_middle_val_p.npy')

    for i in range(0, len(dset_middle_empty_train)):
        # dset_middle_empty_train[i][start:end, start:end, :] = dset_middle_train_p[i] * 2.0 - 1
        dset_middle_empty_train[i][start:end, start:end, :] = np.random.rand(32,32,3)*2.0 - 1
    for i in range(0, len(dset_middle_empty_val)):
        # dset_middle_empty_val[i][start:end, start:end, :] = dset_middle_val_p[i] * 2.0 - 1
        dset_middle_empty_val[i][start:end, start:end, :] = np.random.rand(32, 32, 3) * 2.0 - 1

    # print "DELETE SCRAP DATASETS"
    # del (dset_middle_train_p)
    # del (dset_middle_val_p)

    print "##################"
    print "GET CAPTIONS"
    print "##################"
    train_captions = np.load(pkl_path + 'all_words_2_vectors_train.npy',
                             mmap_mode='r+')
    val_captions = np.load(pkl_path + 'all_words_2_vectors_val.npy',
                           mmap_mode='r+')

    print "GET GAN MODEL"
    gan_merged_model, adversary_model_full, adversary_model_half = model_zoo.DC_caption_LSTM_inception_only(shape_img,
                                                                                                            [32, 64,
                                                                                                             128, 256],
                                                                                                            train_both=True,
                                                                                                            noise=True)

    losses = {"df": [], "dh": [], "g": [], "m": []}

    sgd = model_zoo.SGD(lr=0.0002, momentum=0.5, nesterov=True)
    adam = model_zoo.Adam(lr=0.0002, beta_1=0.5, beta_2=0.9, epsilon=1e-8, decay=0.0)

    print "COMPILE GAN MERGED MODEL"
    # The loss is reduced so as to not completely constrain the filters to overfit to the training data
    model_zoo.alpha = 0.0 / 10  # SSIM and MSE loss
    model_zoo.beta = 0.01  # MSE loss kept higher to avoid patchy reconstructions

    gan_merged_model.compile(optimizer=adam, loss=model_zoo.loss_DSSIM_theano)
    gan_merged_model.summary()
    print "####################################"

    print "COMPILE ADVERSARY MODEL FULL"
    adversary_model_full.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
    adversary_model_full.summary()

    print "COMPILE ADVERSARY MODEL FULL"
    adversary_model_half.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
    adversary_model_half.summary()
    print "####################################"

    print "COMPILE FULL MODELS"
    full_model_tensor_full = adversary_model_full(gan_merged_model.output)
    full_model_tensor_half = adversary_model_half(gan_merged_model.output)
    full_model = model_zoo.Model(outputs=[full_model_tensor_full, full_model_tensor_half],
                                 inputs=gan_merged_model.input)
    full_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    full_model.summary()

    model_weights = model_path + 'full_model_weights.h5'
    print "####################################"

    print "PARAMETERS GMM"
    print gan_merged_model.count_params()
    print "PARAMETERS AM FULL"
    print adversary_model_full.count_params()
    print "PARAMETERS AM HALF"
    print adversary_model_half.count_params()
    print "PARAMETERS FM"
    print full_model.count_params()

    dset_max = np.vstack((dset_train, dset_middle_empty_train))
    dset_real_false = np.zeros(((dset_middle_empty_train.shape[0] + dset_train.shape[0]), 1))
    dset_real_false[0:dset_train.shape[0]] = 1
    dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

    print "Start Adversary Training INITIAL"
    model_zoo.make_trainable(adversary_model_full, True)
    model_zoo.make_trainable(adversary_model_half, True)
    # adversary_model.summary()

    print "#########################"
    print "Start ADVERSARY Fit FULL INITIAL"
    print "#########################"

    adversary_model_full.fit(dset_max, dset_real_false,
                             batch_size=100,
                             nb_epoch=n_epoch,
                             shuffle=True,
                             verbose=True)
    print "#########################"
    print "Start ADVERSARY Fit HALF INITIAL"
    print "#########################"

    adversary_model_half.fit(dset_max, dset_real_false,
                             batch_size=100,
                             nb_epoch=n_epoch,
                             shuffle=True,
                             verbose=True)

    del (dset_max)
    del (dset_real_false)

    print "START SLOW CONVERGE BATCH TRAINING"

    for i in range(0, num_runs):

        print "###########################"
        print i + 1
        print "###########################"
        cp = 0.15 - 0.01 * (i)

        if (cp < 0.0):
            cp = 0.0

        print "###########################"
        print "CP", cp
        print "###########################"

        train_for_both_captions(dset_train,
                                dset_middle_empty_train,
                                train_captions,
                                adversary_model_full,
                                adversary_model_half,
                                gan_merged_model,
                                full_model,
                                losses,
                                plt_path,
                                wr_path,
                                nb_epoch=8000,
                                plt_frq=100,
                                batch_size=40,
                                predict_frq=100,
                                run_num=i, m_train=True,
                                cp=cp)

        print "#########################"
        print "MAKE PREDICTIONS TRAIN"
        print "#########################"
        captions = np.zeros((train_captions.shape[0], train_captions.shape[2], train_captions.shape[3]))
        for j in range(0, len(train_captions)):
            captions[j] = train_captions[j, np.random.randint(0, train_captions.shape[1], 1)]

        predictions = gan_merged_model.predict([dset_middle_empty_train, captions],
                                               batch_size=32)

        # Strengthen adversary to stabilize.
        print "ADVERSARY FULL STRENGTHEN"
        strengthen_adversary(dset_train, predictions, adversary_model_full, strength=0.01)
        print "ADVERSARY HALF STRENGTHEN"
        strengthen_adversary(dset_train, predictions, adversary_model_half, strength=0.01)

        print "SAVE IMAGES"
        write_path = wr_predictions_path + '_train' + str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_train, dset_middle_empty_train, predictions, write_path, middle_only=False)

        print "#########################"
        print "MAKE PREDICTIONS TEST"
        print "#########################"
        captions = np.zeros((val_captions.shape[0], val_captions.shape[2], val_captions.shape[3]))
        for j in range(0, len(val_captions)):
            captions[j] = val_captions[j, np.random.randint(0, val_captions.shape[1], 1)]
        predictions = gan_merged_model.predict([dset_middle_empty_val, captions], batch_size=32)

        print "SAVE IMAGES"
        write_path = wr_predictions_path + '_test' + str(i + 1).zfill(2) + '/'

        if not os.path.exists(write_path):
            os.makedirs(write_path)
        data_fns.write_predicted(dset_val, dset_middle_empty_val, predictions, write_path, middle_only=False)

        del (captions)
        del (predictions)
        full_model.save_weights(model_weights)
        full_model.save(model_path+'caption_full_model_inception_only.h5')

    print "FINISHED ALL RUNS"


def strengthen_adversary(dset_train, predictions, adversary_model, strength=0.01):
    dset_max = np.vstack((dset_train, predictions))
    dset_real_false = np.zeros(((predictions.shape[0] + dset_train.shape[0]), 1))
    dset_real_false[0:dset_train.shape[0]] = 1
    dset_max, dset_real_false = shuffle(dset_max, dset_real_false)

    print "Start Adversary Training INITIAL"
    model_zoo.make_trainable(adversary_model, True)
    model_zoo.weights_kick(adversary_model, kick=0.1)
    print "#########################"
    print "Start ADVERSARY WORKOUT Fit "
    print "#########################"

    loss = 1.0
    count = 0

    while (loss >= strength):

        if count > 9:
            break

        hist = adversary_model.fit(dset_max, dset_real_false,
                                   batch_size=100,
                                   nb_epoch=1,
                                   shuffle=True,
                                   verbose=True)

        loss = hist.history['loss'][-1]
        count += 1
