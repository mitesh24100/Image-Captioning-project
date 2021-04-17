import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
import cv2
from tensorflow.keras.applications.xception import Xception,preprocess_input,decode_predictions
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,Dense,Dropout,Embedding,LSTM
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Add


max_len = 35


model = load_model("model.h5")

model_temp = Xception(weights='imagenet', input_shape=(299,299,3))

model_new = Model(model_temp.input,model_temp.layers[-2].output)



def preprocess_img(img):
    img = image.load_img(img,target_size=(299,299))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img


def encode_img(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector




with open("word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)
    
with open("idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)





def predict_caption(img):
    in_text = 'startseq'
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([img,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text += (' ' +  word)
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = " ".join(final_caption)
    return final_caption



def caption_this_image(image):
    enc = encode_img(image)
    caption = predict_caption(enc)
    return caption


