# lifeworld

Repository for IFT 6266 H2017 project.

Steps to run:
1. Get datasets from : http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/

2. Extract the inpainting directory. cd into the inpainting directory and create the following directories:
      1. pkls,
      2. models,
      3. predictions,
      4. word2vec,

3. Download the google trained gensim model binaries from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit into the word2vec directory


4. Install all required dependencies.

     1. numpy,
     2. scipy,
     3. sklearn,
     4. keras,
     5. theano,
     6. openCV,
     7. tqdm

5. In the full_script.py, change the inpainting_root variable to the folder where you extracted the data.

6. Run full_script.py

