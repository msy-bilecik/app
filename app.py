# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
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
from util import base64_to_pil, pil2datauri

import os
import sys

from PIL import Image
import io
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploadFolder'
app.secret_key = "msy"

FILETYPES = set([ 'pdf', 'png', 'jpg', 'jpeg'])

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
def detecFile():
    title = "MS Detection"
    cap = "MS Detection - Test"

    return render_template('detection.html', title=title, cap=cap)


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
         fname = secure_filename(f.filename)
         f.save(os.path.join(app.config['UPLOAD_FOLDER'], fname))
         # return redirect(url_for('dosyayukleme',dosya=dosyaadi))
         return redirect('msDetection/' + fname)
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
