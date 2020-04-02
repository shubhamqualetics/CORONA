#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Necessary Libraries.
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from cv2 import *
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout


# In[16]:
#from __future__ import division, print_function
# coding=utf-8
#import sys
import os
#import glob
#import re

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
import flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer




# Define a flask app
#import flask
#from flask import Flask, redirect, url_for, request, render_template
#from keras.models import load_model
#from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer


#app = Flask(__name__)

# Model saved with Keras model.save()
# Load your trained model
#import numpy as np
#from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#model = load_model(model1.pkl')
model = pickle.load(open('my_model.pkl', 'rb'))

#model = pickle.load(open('model1.pkl', 'rb'))
model._make_predict_function()          # Necessary


# In[17]:



print('Model loaded. Check http://127.0.0.1:5000/')


# In[18]:


def model_predict(img_path,model):
    image = cv2.imread(img_path)
    data1=[]
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50, 50))
    data1.append(np.array(size_image))
    ready=np.array(data1)
    test_pred2= model.predict(ready)
    return test_pred2


# In[19]:


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# In[20]:


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        result = np.argmax(preds, axis=1)
        result=result[0]
        result=('BacterialPneumonia' if result == 0 else ('COVID' if  result == 1 else( 'Normal'  if result == 2 else 'ViralPneumonia' )))
        return result
    return None


# In[21]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




