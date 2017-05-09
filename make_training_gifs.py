import os
import imageio

batch_size = 40
root_name = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/dcgan_eurus/'
dst_folder = '/usr/local/data/sejacob/lifeworld/data/inpainting/predictions/training_gifs/'
dirlist = os.listdir(root_name)
order_list = []
os.chdir(root_name)
dirs = filter(os.path.isdir, os.listdir(root_name))
# files = [os.path.join(search_dir, f) for f in files] # add path to each file
dirs.sort(key=lambda x: os.path.getmtime(x))
print dirs
for i in range(0, batch_size):
    images = []
    with imageio.get_writer(dst_folder + str(i).zfill(len(str(batch_size))) + ".gif", mode='I',
                            duration=20.0 / (len(dirs))) as writer:
        for j in range(0, len(dirs)):
            filename = root_name + dirs[j] + '/' + str(i).zfill(len(str(batch_size))) + ".jpg"

            image = imageio.imread(filename)

            if j == 0:
                for k in range(0, 20):
                    writer.append_data(image)
            elif (j == len(dirs) - 1):
                for k in range(0, 20):
                    writer.append_data(image)
            else:
                writer.append_data(image)
