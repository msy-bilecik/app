# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn.visualize import display_images
from mrcnn import visualize
from mrcnn import utils
import configFile
from py import metricQc
from py import metreE
from py import util
from keras.preprocessing import image
from keras.models import load_model
from tensorflow import keras
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, flash, abort
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np
import pandas as pd

import os
import sys
import io
import cv2

import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
print(tf.__version__)

# from util import base64_to_pil, pil2datauri


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


DEVICE = "/gpu:0"
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
config = configFile.MsMaskConfig()
MODEL_PATH = 'static/h5/msDet.h5'


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    ALLOW_GROWTH = True
    PER_PROCESS_GPU_MEMORY_FRACTION = 0.9


config = InferenceConfig()
config.display()
with tf.device(DEVICE):
    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_PATH, config=config)


model.load_weights(MODEL_PATH, by_name=True)
model.keras_model._make_predict_function()
print('Loading Weights')

# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploadFolder'
UPLOAD_PRED_PATH = app.config['UPLOAD_FOLDER']
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


@app.route('/msDetection/<filename>')
def detecFile(filename):
    return redirect(url_for('static', filename='uploadFolder/'+filename), code=301)


@app.route('/upload1File', methods=['POST'])
def upload1File():
    if request.method == 'POST':
        # formdan dosya gelip gelmediğini kontrol edelim
        if 'fname' not in request.files:
            flash('Dosya seçilmedi')
            return redirect('msDetection')

            # kullanıcı dosya seçmemiş ve tarayıcı boş isim göndermiş mi
        f = request.files['fname']
        if f.filename == '':
            flash('Dosya seçilmedi')
            return redirect('msDetection')

            # gelen dosyayı güvenlik önlemlerinden geçir
        if f and uzanti_kontrol(f.filename):

            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)
            flash('Dosya başarıyla yüklendi.')
            image = cv2.imread(filepath)
            results = model.detect([image], verbose=1)

            class_names = ['BG', 'msMask']
            r = results[0]
            predFileName = "pre_"+filename.split('.')[0]+".jpg"
            print(predFileName)
            pred_path = UPLOAD_PRED_PATH+"/"+predFileName
            visualize.save_instances(
                image, r['rois'], r['masks'], r['class_ids'], class_names,  r['scores'], path=pred_path)
            flash("Tespit edilen lezyon adedi: "+str(len(r['class_ids'])))
            flash("Lezyon Tespit Başarımı: " +
                  str(sum(r['scores'])/len(r['class_ids'])))
            
            title = "MS Detection"
            cap = "MS Detection - Test"
            return render_template('detection.html', title=title, cap=cap+" PRE "+filename, filename=predFileName)

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


if __name__ == "__main__":
    app.run(debug=True)
