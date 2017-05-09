from model_pkg import model_zoo
from model_pkg import visualizations

print "GET MODELS"

gan_merged_model, adversary_model_full, adversary_model_half = model_zoo.DC_caption_LSTM_inception_only((64, 64, 3),
                                                                                  [32, 64, 128,
                                                                                   256],
                                                                                  train_both=True,
                                                                                  noise=True)

full_model_tensor_full = adversary_model_full(gan_merged_model.output)
full_model_tensor_half = adversary_model_half(gan_merged_model.output)
full_model = model_zoo.Model(outputs=[full_model_tensor_full, full_model_tensor_half],
                             inputs=gan_merged_model.input)
full_model.summary()

gmm_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/generator.png'
adm_filepath_full = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/discriminator_full.png'
adm_filepath_half = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/discriminator_half.png'
full_filepath = '/usr/local/data/sejacob/lifeworld/data/inpainting/models/full_model.png'

visualizations.make_model_visualization(gan_merged_model, gmm_filepath)
visualizations.make_model_visualization(adversary_model_full, adm_filepath_full)
visualizations.make_model_visualization(adversary_model_half, adm_filepath_half)
visualizations.make_model_visualization(full_model, full_filepath)
