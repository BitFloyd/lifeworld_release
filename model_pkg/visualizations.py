from keras.utils.vis_utils import plot_model


def make_model_visualization(model, filepath):
    plot_model(model, to_file=filepath, show_shapes=True, show_layer_names=True)
    return True
