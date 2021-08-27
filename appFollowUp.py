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
from skimage.metrics import structural_similarity as ssim


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


def maskCompound(mArr):
    col = mArr.shape
    s = col[2]
    if s > 1:
        mArr1 = mArr[:, :, 0]
        i = 1
        while i < s:
            mArr2 = mArr[:, :, i]
            mArr1 = np.logical_or(mArr1, mArr2)
            i = i+1
        m = mArr1
        m2 = m[:, :, np.newaxis]

        return m2
    else:
        return mArr


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compareMatrix(masks1, masks2):
    message = ""
    ix = masks1.shape[2]
    iy = masks2.shape[2]
    cMatrix = np.zeros((ix, iy))
    bMatrix = np.zeros((ix, iy))
    likeC = 0
    tinyC = 0
    bigC = 0
    i = 0
    for i in range(ix):
        mask1 = masks1[:, :, i]
        for t in range(iy):
            mask2 = masks2[:, :, t]
            mask1Norm = mask1 / np.sqrt(np.sum(mask1**2))
            mask2Norm = mask2 / np.sqrt(np.sum(mask2**2))
            simScore = np.sum(mask2Norm*mask1Norm)
            if(simScore > 0):
                bMatrix[i, t] = mask2.sum()/mask1.sum()
                if (bMatrix[i, t] > 0.9 and bMatrix[i, t] < 1.1):
                    likeC = likeC+1
                elif (bMatrix[i, t] <= 0.9):
                    tinyC = tinyC+1
                elif (bMatrix[i, t] >= 1.1):
                    bigC = bigC+1

            cMatrix[i, t] = simScore
            t = t+1
        i = i+1

    zNew = cMatrix.sum(axis=0)
    zOld = cMatrix.sum(axis=1)
    exC = zOld.size-np.count_nonzero(zOld)
    newC = zNew.size-np.count_nonzero(zNew)

    if(likeC == 0 and bigC == 0 and tinyC == 0):
        message = "değerlendirme için yeterli benzerlik bulunamadı. "
    elif(likeC == iy and likeC == ix):
        message = "lezyonlarda değişim olmamıştır."
    else:
        if(likeC > 0):
            message = message + \
                "{:.0f} plakda değişim olmamıştır.".format(likeC)
        if(tinyC > 0):
            message = message + " {:.0f} plak küçülmüştür.".format(tinyC)
        if(bigC > 0):
            message = message + \
                " {:.0f} plakda büyüme gözlenmiştir.".format(bigC)
        if(exC > 0):
            message = message + \
                " {: .0f} plak gözlenmemiştir.".format(exC)
        if(newC > 0):
            message = message + \
                " {: .0f} yeni plak tespit edilmiştir.".format(newC)

    print(cMatrix)
    print(bMatrix)
    return message, cMatrix, bMatrix


def compareMasks(masks1, masks2):
    print(masks1.shape)
    print(masks2.shape)
    # print(maskCompound(masks1).shape)
    # print(maskCompound(masks2).shape)
    if(masks1.shape[0] == masks2.shape[0] and masks1.shape[1] == masks2.shape[1]):
        message, cMatrix, bMatrix = compareMatrix(masks1, masks2)

    else:
        message = "uyumsuz boyut"

    return message


@app.route('/')
def index():
    title = "Homepage"
    cap = "Homepage - Test"
    content = "Multiple Sclerosis Detection And Follow-up System Test Page"
    return render_template('main.html', title=title, cap=cap, content=content)


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
            flash('Dosya seçilmedi', 'danger')
            return redirect('msDetection')

            # kullanıcı dosya seçmemiş ve tarayıcı boş isim göndermiş mi
        f = request.files['fname']
        if f.filename == '':
            flash('Dosya seçilmedi', 'danger')
            return redirect('msDetection')

            # gelen dosyayı güvenlik önlemlerinden geçir
        if f and uzanti_kontrol(f.filename):

            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)
            flash('Dosya başarıyla yüklendi.', 'success')
            image = cv2.imread(filepath)
            results = model.detect([image], verbose=1)

            class_names = ['BG', 'msMask']
            r = results[0]
            predFileName = "pre_"+filename.split('.')[0]+".jpg"
            print(predFileName)
            pred_path = UPLOAD_PRED_PATH+"/"+predFileName
            visualize.save_instances(
                image, r['rois'], r['masks'], r['class_ids'], class_names,  r['scores'], path=pred_path)
            flash("Tespit edilen lezyon adedi: " +
                  str(len(r['class_ids'])), 'light')
            flash("Lezyon Tespit Başarımı: %{:.2f} ".format(+
                  (sum(r['scores'])/len(r['class_ids']))*100), 'warning')

            title = "MS Detection"
            cap = "MS Detection - Test"
            return render_template('detection.html', title=title, cap=cap+" PRE "+filename, filename=predFileName)

        else:
            flash('İzin verilmeyen dosya uzantısı', 'danger')
            return redirect('msDetection')
    else:
        abort(401)


@app.route('/followup')
def followup():
    title = "MS FollowUp "
    cap = "MS FollowUp - Test"
    return render_template('followup.html', title=title, cap=cap)


@app.route('/upload2File', methods=['POST'])
def upload2File():
    if request.method == 'POST':
        # formdan dosya gelip gelmediğini kontrol edelim
        if ('firstMR' and 'secondMR') not in request.files:
            flash('Dosya seçilmedi', 'danger')
            return redirect('followup')

        f1 = request.files['firstMR']
        f2 = request.files['secondMR']

        if f1.filename == '' or f2.filename == '':
            flash('Dosya seçilmedi', 'danger')
            return redirect('followup')

        if((f1 and uzanti_kontrol(f1.filename)) and (f2 and uzanti_kontrol(f2.filename))) != True:
            flash('Dosya tipinde hata var. ', 'danger')
            return redirect('followup')
        else:
            filename1 = secure_filename(f1.filename)
            filename2 = secure_filename(f2.filename)

            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            f1.save(filepath1)
            # flash('ilk MR görüntüsü başarıyla yüklendi.', 'success')

            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
            f2.save(filepath2)
            # flash('ikinci MR görüntüsü başarıyla yüklendi.', 'success')

            class_names = ['BG', 'msMask']

            image1 = cv2.imread(filepath1)
            results1 = model.detect([image1], verbose=1)
            r1 = results1[0]
            predFileName1 = "pre_"+filename1.split('.')[0]+".jpg"
            pred_path1 = UPLOAD_PRED_PATH+"/"+predFileName1
            visualize.save_instances(
                image1, r1['rois'], r1['masks'], r1['class_ids'], class_names,  r1['scores'], path=pred_path1)

            image2 = cv2.imread(filepath2)
            results2 = model.detect([image2], verbose=1)
            r2 = results2[0]
            predFileName2 = "pre_"+filename2.split('.')[0]+".jpg"
            pred_path2 = UPLOAD_PRED_PATH+"/"+predFileName2
            visualize.save_instances(
                image2, r2['rois'], r2['masks'], r2['class_ids'], class_names,  r2['scores'], path=pred_path2)

            compareResult = compareMasks(r1['masks'], r2['masks'])
            flash(compareResult, 'warning')
            # print(r1['masks'].shape)
            # print(r2['masks'].shape)

    title = "Follow-Up Pre"
    cap = "Follow-Up Preview - Test"

    return render_template('followupPre.html', title=title, cap=cap, orgFilename1=filename1, orgFilename2=filename2, filename1=predFileName1, filename2=predFileName2)


@app.route('/about')
def about():
    title = "About"
    cap = "About - Test"
    return render_template('main.html', title=title, cap=cap)


if __name__ == "__main__":
    app.run(debug=True)
