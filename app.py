# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, flash, abort
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np
import pandas as pd
from flask import jsonify
from flask import render_template
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from py import util
from py import metreE
from py import metricQc
import configFile
# from util import base64_to_pil, pil2datauri

import os
import sys
import io

from PIL import Image
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        # Currently, memory growth needs to be the same across GPUs
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#    except RuntimeError as e:
#        # Memory growth must be set before GPUs have been initialized
#        print(e)


DEVICE = "/cpu:0"
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
config = configFile.MsMaskConfig()
MODEL_PATH = 'static/h5/msDet.h5'


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1
    ALLOW_GROWTH = True
    PER_PROCESS_GPU_MEMORY_FRACTION = 0.9


config = InferenceConfig()
# config.display()
with tf.device(DEVICE):
    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_PATH, config=config)

model.load_weights(MODEL_PATH, by_name=True)
model.keras_model._make_predict_function()
print('Loading Weights')

# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    # img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    # If has an alpha channel, remove it for consistency
    if x.shape[-1] == 4:
        x = x[..., :3]

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!

    preds = model.detect([x], verbose=1)
    r = preds[0]

    return r['rois']


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploadFolder'
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024
app.secret_key = "msy"

FILETYPES = set(['png', 'jpg', 'jpeg'])


def uzanti_kontrol(dosyaadi):
    return '.' in dosyaadi and \
        dosyaadi.rsplit('.', 1)[1].lower() in FILETYPES


@app.route('/')
def index():
    title = "Homepage"
    cap = "Homepage - Test"
    return render_template('main.html', title=title, cap=cap)


@app.route('/msDetection')
def msDetection():
    title = "MS Detection"
    cap = "MS Detection - Test"
    return render_template('detection.html', title=title, cap=cap)


@app.route('/msDetection/<fname>')
def detecFile1():
    if request.method == 'POST':
        q = dict(request.files)
        img = q["image"]
        img = Image.open(io.BytesIO(img.read()))
        # img = Image.open(io.BytesIO(img))
        # img = base64_to_pil(request.json)
        preds = model_predict(img, model)
        if preds.size > 0:
            print(preds)
            responsejson = {
                'x1': int(preds[0][0]),
                'x2': int(preds[0][1]),
                'y1': int(preds[0][2]),
                'y2': int(preds[0][3]),
                }
            json_resp = jsonify(responsejson)
            json_resp.status_code = 200
            print(json_resp)
            return json_resp
        else:
            json_resp = {
                'x1': 'None',
                'x2': 'None',
                'y1': 'None',
                'y2': 'None',
                }
            json_resp = jsonify(json_resp)
            json_resp.status_code = 200
    return json_resp
    
    #return render_template('detection.html', title=title, cap=cap)


def detecFile(filename):
    #title = "MS Detection"
    #cap = "MS Detection - Test"
    #return render_template('detection.html', title=title, cap=cap+" "+fname, filename=fname)
    return redirect(url_for('static', filename='uploadFolder/'+filename), code=301)


@app.route('/upload1File', methods=['POST'])
def upload1File():
    if request.method == 'POST':

		# formdan dosya gelip gelmediğini kontrol edelim
      if 'fileMRi' not in request.files:
         flash('Dosya seçilmedi')
         return redirect('msDetection')

		# kullanıcı dosya seçmemiş ve tarayıcı boş isim göndermiş mi
      f = request.files['fileMRi']
      if f.filename == '':
         flash('Dosya seçilmedi')
         return redirect('msDetection')

		# gelen dosyayı güvenlik önlemlerinden geçir
      if f and uzanti_kontrol(f.filename):
          filename = secure_filename(f.filename)
          f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          flash('Dosya başarıyla yüklendi.')
          title = "MS Detection"
          cap = "MS Detection - Test"
          return render_template('detection.html', title=title, cap=cap+" "+filename, filename=filename)

      else:
         flash('İzin verilmeyen dosya uzantısı')
         return redirect('msDetection')
    else:
        abort(401)


@app.route('/followup')
def followup():
   
    title = "MS FollowUp "
    cap = "MS FollowUp - Test"
    return render_template('main.html', title=title, cap=cap)



@app.route('/about')
def about():
    title = "About"
    cap = "About - Test"
    return render_template('main.html', title=title, cap=cap)



if __name__=="__main__":
   app.run(debug=True) 
