# lifeworld

Repository for IFT 6266 H2017 project.

Steps to run:
Get datasets from : http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/

Extract the inpainting directory.
cd into the inpainting directory and create the following directories:
pkls,
models,
predictions,
word2vec,

Download the google trained gensim model binaries from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit 
into the word2vec directory


Install all required dependencies.

numpy,
scipy,
sklearn,
keras,
theano,
openCV,
tqdm

In the full_script.py, change the inpainting_root variable to the folder where you extracted the data.

Run full_script.py

