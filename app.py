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
import uuid
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
ColorSet = [(1.0, 1.0, 0.0), (0.5, 1.0, 0.0),  (1.0, 0.0, 0.0),
            (0.0, 0.5, 1.0), (1, 1, 1)]

# (1.0, 1.0, 0.0) sarı
# (0.5, 1.0, 0.0) yeşil
# (1.0, 0.0, 0.0) kırmızı
# (0.0, 0.5, 1.0) mavi
# (0.25, 0.25, 0.25) gri


def uzanti_kontrol(dosyaadi):
    return '.' in dosyaadi and \
        dosyaadi.rsplit('.', 1)[1].lower() in FILETYPES


def uzanti_kontrolJson(dosyaadi):
    return '.' in dosyaadi and \
        dosyaadi.rsplit('.', 1)[1].lower() in ['json']


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


def compareMasks(r1, r2):
    masks1 = r1['masks']
    masks2 = r2['masks']
    return compareM(masks1, masks2)


def compareM(masks1, masks2):
    if(masks1.shape[0] == masks2.shape[0] and masks1.shape[1] == masks2.shape[1]):

        message = ""
        ix = masks1.shape[2]
        iy = masks2.shape[2]
        zN = np.zeros(iy).astype(int)+4
        zO = np.zeros(ix).astype(int)
        ratesR1 = np.zeros(ix)
        ratesR2 = np.zeros(iy)
        cMatrix = np.zeros((ix, iy))
        bMatrix = np.zeros((ix, iy))
        likeC = 0
        tinyC = 0
        bigC = 0
        i = 0
        r1 = []
        r2 = []
        for i in range(ix):
            mask1 = masks1[:, :, i]
            for t in range(iy):
                mask2 = masks2[:, :, t]
                mask1Norm = mask1 / np.sqrt(np.sum(mask1**2))
                mask2Norm = mask2 / np.sqrt(np.sum(mask2**2))
                simScore = np.sum(mask2Norm*mask1Norm)
                if(simScore > 0):
                    bMatrix[i, t] = mask2.sum()/mask1.sum()
                    if (bMatrix[i, t] > 0.98 and bMatrix[i, t] < 1.02):
                        likeC = likeC+1
                        zN[t] = 3
                        zO[i] = 3
                        ratesR1[i] = 0
                        ratesR2[t] = 0
                    elif (bMatrix[i, t] <= 0.98):
                        tinyC = tinyC+1
                        zN[t] = 1
                        zO[i] = 1
                        rate = (1 - bMatrix[i, t])*100
                        ratesR1[i] = rate
                        ratesR2[t] = rate
                        flash(" 1 plak  %{:.2f} küçülmüştür. ".format(
                            rate), "success")
                    elif (bMatrix[i, t] >= 1.02):
                        bigC = bigC+1
                        zN[t] = 2
                        zO[i] = 2
                        rate = (bMatrix[i, t] - 1)*100
                        ratesR1[i] = rate
                        ratesR2[t] = rate
                        flash(" 1 plakda %{:.2f} büyüme gözlenmiştir. ".format(
                            rate), "danger")

                cMatrix[i, t] = simScore
                t = t+1
            i = i+1

        print(zO)
        print(zN)

        colorsR2 = colorSetting(zN, ColorSet)
        colorsR1 = colorSetting(zO, ColorSet)

        zNew = bMatrix.sum(axis=0)
        # print("zNew")
        # print(zNew)

        zOld = bMatrix.sum(axis=1)
        # print("zOld")
        # print(zOld)

        exC = zOld.size-np.count_nonzero(zOld)
        newC = zNew.size-np.count_nonzero(zNew)

        for i in range(ix):
            if (zOld[i] == 0):
                ratesR1[i] = 0
        for i in range(iy):
            if (zNew[i] == 0):
                ratesR2[i] = 0

        if(likeC == 0 and bigC == 0 and tinyC == 0):
            message = "değerlendirme için yeterli benzerlik bulunamadı. "
            flash("değerlendirme için yeterli benzerlik bulunamadı. ", "info")
        elif(likeC == iy and likeC == ix):
            message = "lezyonlarda değişim olmamıştır."
            flash("lezyonlarda değişim olmamıştır.", "success")
        else:
            if(likeC > 0):
                message = message + \
                    "{:.0f} plakda değişim olmamıştır.".format(likeC)
                flash("{:.0f} plakda değişim olmamıştır.".format(likeC), "info")
            if(tinyC > 0):
                message = message + " {:.0f} plak küçülmüştür.".format(tinyC)
            if(bigC > 0):
                message = message + \
                    " {:.0f} plakda büyüme gözlenmiştir.".format(bigC)
            if(exC > 0):
                message = message + \
                    " {: .0f} plak gözlenmemiştir.".format(exC)
                flash(" {: .0f} plak gözlenmemiştir.".format(exC), "warning")
            if(newC > 0):
                message = message + \
                    " {: .0f} yeni plak tespit edilmiştir.".format(newC)
                flash(" {: .0f} yeni plak tespit edilmiştir.".format(
                    newC), "light")

    else:
        message = "uyumsuz boyut"
        flash("uyumsuz boyut", "danger")
        zN = zO = 0

    return message, colorsR1, colorsR2, ratesR1, ratesR2, zO, zN


def colorSetting(colorM, ColorSet):
    colors = []
    for i in range(len(colorM)):
        colors.append(ColorSet[int(colorM[i])])

    return colors


@app.route('/')
def index():
    title = "MS-DAF"
    cap = "Homepage - Test"
    content = "Multiple Sclerosis Detection And Follow-up System Test Page"
    return render_template('main.html', title=title, cap=cap, content=content)


@app.route('/showPic/<filename>')
def detecFile(filename):
    return redirect(url_for('static', filename='uploadFolder/'+filename), code=301)


@app.route('/msShow')
def msShow():
    title = "MS-DAF - MS Görüntüleme"
    cap = "Uzman Hekimin Belirlediği MS Plaklarının Görüntülenmesi"
    abstract = "Hekimlerin işaretledikleri MS plaklarını sergileyen uygulama sayfamızdır. Bunun için MR kesiti ve VGG 1.0.6 formatında segmentasyon dosyanızı yüklemelisiniz. "
    fxUrl = url_for("msSliceShow")
    json = True
    return render_template('detection.html', title=title, cap=cap, abstract=abstract, fx=fxUrl, json1=json)


@app.route('/msDetection')
def msDetection():
    title = "MS-DAF - MS Tespiti"
    cap = "MR Görüntüsü Üzerinde Otomatik MS Tespiti"
    abstract = "MR görüntüleri üzerinde otomatik olarak plak tespiti yapan uygulama sayfamızdır. Bunun için MR kesitinizi yüklemeniz yeterlidir. "
    fxUrl = url_for("msFinder")
    return render_template('detection.html', title=title, cap=cap, abstract=abstract, fx=fxUrl)


@app.route('/msDetectionCompare')
def msDetectionCompare():
    title = "MS-DAF - Karşılaştırmalı MS Tespiti"
    cap = "Karşılaştırmalı MS Tespiti"
    abstract = "MR görüntüleri üzerinde otomatik olarak plak tespiti ve uzman hekim görüşü ile karşılaştırmasını yapan uygulama sayfamızdır. Bunun için MR kesiti ve VGG 1.0.6 formatında segmentasyon dosyanızı yüklemelisiniz. "
    fxUrl = url_for("msFinderCompare")
    json = True
    return render_template('detection.html', title=title, cap=cap, abstract=abstract, fx=fxUrl, json1=json)

################# Follow -Up Links##########


@app.route('/msFollowUp')
def msFollowUp():
    title = "MS-DAF - MS FollowUp "
    cap = "MS Takibi"
    abstract = "Hekimlerin işaretledikleri MS plaklarını sergileyen, farklı iki periyotta alınmış görüntüleri karşılaştıran uygulama sayfamızdır. Bunun için 2 MR kesiti ve VGG 1.0.6 formatında segmentasyon dosyanızı yüklemelisiniz. "
    fxUrl = url_for("msFollowUpShow")
    json = True
    return render_template('followup.html',  title=title, cap=cap, abstract=abstract, fx=fxUrl, json1=json)


@app.route('/msOtoFollowUp')
def msOtoFollowUp():
    title = "MS-DAF - MS FollowUp "
    cap = "Otomatik MS Takibi"
    abstract = "Farklı iki periyotta alınmış MR görüntüleri üzerindeki lezyonları tespit ederek karşılaştıran uygulama sayfamızdır. Bunun için 2 MR kesiti dosyanızı yüklemelisiniz. "
    fxUrl = url_for("msFollowUpFinder")
    return render_template('followup.html',  title=title, cap=cap, abstract=abstract, fx=fxUrl)


@app.route('/msFollowUpCompare')
def msFollowUpCompare():
    title = "MS-DAF - MS FollowUp "
    cap = "Otomatik Karşılaştırmalı MS Takibi"
    abstract = "Hekimlerin işaretledikleri MS plaklarını sergileyen, farklı iki periyotta alınmış görüntüleri karşılaştıran uygulama sayfamızdır. Bunun için MR kesiti ve VGG 1.0.6 formatında segmentasyon dosyanızı yüklemelisiniz. "
    fxUrl = url_for("msFollowUpCompareShow")
    json = True
    return render_template('followup.html',  title=title, cap=cap, abstract=abstract, fx=fxUrl, json1=json)


@app.route('/msSliceShow', methods=['POST'])
def msSliceShow():
    if request.method == 'POST':
        # formdan dosya gelip gelmediğini kontrol edelim
        if 'fname' not in request.files:
            flash('Dosya seçilmedi', 'danger')
            return redirect('msShow')

            # kullanıcı dosya seçmemiş ve tarayıcı boş isim göndermiş mi
        f = request.files['fname']
        fJson = request.files['jsonfname']
        if f.filename == '':
            flash('Dosya seçilmedi', 'danger')
            return redirect('msShow')

            # gelen dosyayı güvenlik önlemlerinden geçir
        if f and uzanti_kontrol(f.filename) and fJson and uzanti_kontrolJson(fJson.filename):

            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)

            flash('Dosya başarıyla yüklendi.', 'success')
            image = cv2.imread(filepath)
            class_names = ['BG', 'msMask']
            jsonFile = str(uuid.uuid4())+".json"
            filenameJ = secure_filename(jsonFile)
            filepathJ = os.path.join(app.config['UPLOAD_FOLDER'], filenameJ)
            fJson.save(filepathJ)
            dataset = configFile.MsMaskDataset()
            dataset.sload_msMask(app.config['UPLOAD_FOLDER'], filepathJ)
            dataset.prepare()
            image1, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(
                    dataset, config, 0, use_mini_mask=False)
            info = dataset.image_info[0]

            GTFileName = "GT_"+filename.split('.')[0]+".jpg"
            GTFilePath = UPLOAD_PRED_PATH+"/"+GTFileName
            visualize.save_instances(
                image1, gt_bbox, gt_mask, gt_class_id, class_names, path=GTFilePath)

            title = "MS-DAF - MS Görüntüleme"
            cap = "MS Görüntüleme"
            abstract = filename.split('.')[
                0]+" dosyasının uzman hekim görüşleri ile belirtilen plak(ları) aşağıda detaylı olarak görülmektedir."

            return render_template('detectionPre.html', title=title, cap=cap, abstract=abstract,
                                   GTFileName=GTFileName, orjFile=filename)

        else:
            flash('İzin verilmeyen dosya uzantısı', 'danger')
            return redirect('msShow')
    else:
        abort(401)


@app.route('/msFinder', methods=['POST'])
def msFinder():
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
            predFileName = "det_"+filename.split('.')[0]+".jpg"
            print(predFileName)
            pred_path = UPLOAD_PRED_PATH+"/"+predFileName
            class_names = ['BG', 'msPlaque']
            visualize.save_instances(
                image, r['rois'], r['masks'], r['class_ids'], class_names,  r['scores'], path=pred_path)
            flash("Tespit edilen lezyon adedi: " +
                  str(len(r['class_ids'])), 'light')

            title = "MS-DAF - MS Tespiti"
            cap = "MR Görüntüsü Üzerinde Otomatik MS Tespiti"
            abstract = filename.split('.')[
                0]+" dosyasının otomatik olarak tespit edilen MS plak(ları) detaylı olarak görülmektedir."
            return render_template('detectionPre.html', title=title, cap=cap, abstract=abstract, orjFile=filename, predFileName=predFileName)

        else:
            flash('İzin verilmeyen dosya uzantısı', 'danger')
            return redirect('msDetection')
    else:
        abort(401)


@app.route('/msFinderCompare', methods=['POST'])
def msFinderCompare():
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
            predFileName = "det_"+filename.split('.')[0]+".jpg"
            print(predFileName)
            pred_path = UPLOAD_PRED_PATH+"/"+predFileName
            class_names = ['BG', 'msPlaque']
            visualize.save_instances(
                image, r['rois'], r['masks'], r['class_ids'], class_names,  r['scores'], path=pred_path)
            flash("Tespit edilen lezyon adedi: " +
                  str(len(r['class_ids'])), 'light')

            fJson = request.files['jsonfname']
            if fJson and uzanti_kontrolJson(fJson.filename):
                jsonFile = str(uuid.uuid4())+".json"
                filenameJ = secure_filename(jsonFile)
                filepathJ = os.path.join(
                    app.config['UPLOAD_FOLDER'], filenameJ)
                fJson.save(filepathJ)

                dataset = configFile.MsMaskDataset()
                dataset.sload_msMask(app.config['UPLOAD_FOLDER'], filepathJ)
                dataset.prepare()
                image1, image_meta, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(
                        dataset, config, 0, use_mini_mask=False)
                info = dataset.image_info[0]
                GTFileName = "GT_"+filename.split('.')[0]+".jpg"
                GTFilePath = UPLOAD_PRED_PATH+"/"+GTFileName

                visualize.save_instances(
                    image1, gt_bbox, gt_mask, gt_class_id, class_names, path=GTFilePath)

                GTMatchFile = "GT_over_"+filename.split('.')[0]+".jpg"
                GTMatchPath = UPLOAD_PRED_PATH+"/"+GTMatchFile
                visualize.save_differences(image, gt_bbox, gt_class_id, gt_mask,
                                           r['rois'], r['class_ids'], r['scores'], r['masks'],
                                           dataset.class_names, path=GTMatchPath,
                                           show_box=False
                                           )
                result = maskCompound(r['masks'])
                reference = maskCompound(gt_mask)

                dc = metreE.dc(result, reference)
                jc = metreE.jc(result, reference)
                iouX = utils.compute_overlaps_masks(result, reference)
                iou = iouX[0][0]
                print()
                voe = 1-iou
                vol = np.count_nonzero(result)
                if not(vol == 0):
                    asd = metreE.asd(result, reference)
                    assd = metreE.assd(result, reference)
                    flash("DC:{:.2f}, JC:{:.2f}, VOE:{:.2f}, IOU:{:.2f}, ASD:{:.2f}, ASSD:{:.2f} ".format(
                        dc, jc, voe, iou, asd, assd), "light")
                else:
                    flash("DC:{:.2f}, JC:{:.2f}, VOE:{:.2f}, IOU:{:.2f} ".format(
                        dc, jc, voe, iou), "light")

                title = "MS-DAF - Karşılaştırmalı MS Tespiti"
                cap = "MR Görüntüsü Üzerinde Otomatik MS Tespiti ve Uzman Hekim Seçimleri ile karşılaştırması"
                abstract = filename.split('.')[
                    0]+" dosyasının otomatik olarak tespit edilen MS plak(ları) detaylı olarak görülmektedir."
                return render_template('detectionPre.html', title=title, cap=cap, abstract=abstract,
                                       orjFile=filename,   predFileName=predFileName,
                                       GTOverFileName=GTMatchFile,  GTFileName=GTFileName)

            else:
                flash("Ground Truth file not exist or wrong", "danger")

            title = "MS-DAF - MS Tespiti"
            cap = "MR Görüntüsü Üzerinde Otomatik MS Tespiti"
            abstract = filename.split('.')[
                0]+" dosyasının otomatik olarak tespit edilen MS plak(ları) detaylı olarak görülmektedir. Uzman Hekim seçimlerini yüklemediğiniz için bu bölümden devam edilmiştir. "
            return render_template('detectionPre.html', title=title, cap=cap, abstract=abstract, orjFile=filename, predFileName=predFileName)

        else:
            flash('İzin verilmeyen dosya uzantısı', 'danger')
            return redirect('msDetection')
    else:
        abort(401)


@app.route('/msFollowUpShow', methods=['POST'])
def msFollowUpShow():
    if request.method == 'POST':
        # formdan dosya gelip gelmediğini kontrol edelim
        if ('firstMR' and 'secondMR') not in request.files:
            flash('Dosya seçilmedi', 'danger')
            return redirect('msFollowUp')

        f0 = request.files['firstMR']
        f1 = request.files['secondMR']
        fJson = request.files['jsonfname']

        if f0.filename == '' or f1.filename == '' or fJson.filename == '':
            flash('Dosya seçilmedi', 'danger')
            return redirect('msFollowUp')

        if((f0 and uzanti_kontrol(f0.filename)) and (f1 and uzanti_kontrol(f1.filename)) and (fJson and uzanti_kontrolJson(fJson.filename))) != True:
            flash('Dosya tipinde hata var. ', 'danger')
            return redirect('msFollowUp')
        else:
            filename0 = secure_filename(f0.filename)
            filename1 = secure_filename(f1.filename)

            filepath0 = os.path.join(app.config['UPLOAD_FOLDER'], filename0)
            f0.save(filepath0)
            # flash('ilk MR görüntüsü başarıyla yüklendi.', 'success')

            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            f1.save(filepath1)
            # flash('ikinci MR görüntüsü başarıyla yüklendi.', 'success')
            # flash('Dosyalar başarıyla yüklendi.', 'success')
            jsonFile = str(uuid.uuid4())+".json"
            filenameJ = secure_filename(jsonFile)
            filepathJ = os.path.join(app.config['UPLOAD_FOLDER'], filenameJ)
            fJson.save(filepathJ)
            dataset = configFile.MsMaskDataset()
            print("Json:", filepathJ)
            dataset.sload_msMask(app.config['UPLOAD_FOLDER'], filepathJ)
            dataset.prepare()
            info = dataset.image_info[0]
            sifir = 0
            bir = 1
            if(info["id"] != f0.filename):
                bir = 0
                sifir = 1

            image0, image_meta0, gt_class_id0, gt_bbox0, gt_mask0 =\
                modellib.load_image_gt(
                    dataset, config, sifir, use_mini_mask=False)

            GTFileName0 = "GT_"+filename0.split('.')[0]+".jpg"
            GTFilePath0 = UPLOAD_PRED_PATH+"/"+GTFileName0

            image1, image_meta1, gt_class_id1, gt_bbox1, gt_mask1 =\
                modellib.load_image_gt(
                    dataset, config, bir, use_mini_mask=False)

            GTFileName1 = "GT_"+filename1.split('.')[0]+".jpg"
            GTFilePath1 = UPLOAD_PRED_PATH+"/"+GTFileName1

            # class_names = ['BG', 'msMask']

            message, colorsR0, colorsR1, ratesR0, ratesR1, classIDs0, classIDs1 = compareM(
                gt_mask0, gt_mask1)

            print(ratesR0, ratesR1, classIDs0, classIDs1)

            class_names = ['old', 'smaller', 'bigger', 'same', 'new']

            visualize.save_instances(
                image0, gt_bbox0, gt_mask0, classIDs0, class_names, ratesR0,
                path=GTFilePath0, colors=colorsR0)
            visualize.save_instances(
                image1, gt_bbox1, gt_mask1, classIDs1, class_names, ratesR1,
                path=GTFilePath1, colors=colorsR1)

            title = "Follow-Up"
            cap = "Follow-Up Görüntüleme"
            abstract = "Yükelenen dosyaların uzman hekim görüşleri ile \
                        belirtilen plak(ları) ve bu plakların değişimleri aşağıda detaylı olarak görülmektedir."

            return render_template('followupPre.html', title=title, cap=cap, abstract=abstract,
                                   orjfile0=filename0, orjfile1=filename1,
                                   GTFileName0=GTFileName0, GTFileName1=GTFileName1
                                   )
    else:
        flash("bir hata oluştu", "danger")
        return redirect('msFollowUpmsFollowUp')


@app.route('/msFollowUpFinder', methods=['POST'])
def msFollowUpFinder():
    if request.method == 'POST':
        # formdan dosya gelip gelmediğini kontrol edelim
        if ('firstMR' and 'secondMR') not in request.files:
            flash('Dosya seçilmedi', 'danger')
            return redirect('msOtoFollowUp')

        f0 = request.files['firstMR']
        f1 = request.files['secondMR']

        if f0.filename == '' or f1.filename == '':
            flash('Dosya seçilmedi', 'danger')
            return redirect('msOtoFollowUp')

        if((f0 and uzanti_kontrol(f0.filename)) and (f1 and uzanti_kontrol(f1.filename))) != True:
            flash('Dosya tipinde hata var. ', 'danger')
            return redirect('msOtoFollowUp')
        else:
            filename0 = secure_filename(f0.filename)
            filename1 = secure_filename(f1.filename)

            filepath0 = os.path.join(app.config['UPLOAD_FOLDER'], filename0)
            f0.save(filepath0)
            # flash('ilk MR görüntüsü başarıyla yüklendi.', 'success')

            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            f1.save(filepath1)
            # flash('ikinci MR görüntüsü başarıyla yüklendi.', 'success')
            # flash('Dosyalar başarıyla yüklendi.', 'success')

            class_names = ['BG', 'msMask']

            image1 = cv2.imread(filepath1)
            results1 = model.detect([image1], verbose=1)
            r1 = results1[0]
            predFileName1 = "pre_"+filename1.split('.')[0]+".jpg"
            pred_path1 = UPLOAD_PRED_PATH+"/"+predFileName1

            image0 = cv2.imread(filepath0)
            results0 = model.detect([image0], verbose=1)
            r0 = results0[0]
            predFileName0 = "pre_"+filename0.split('.')[0]+".jpg"
            pred_path0 = UPLOAD_PRED_PATH+"/"+predFileName0

            message, colorsR0, colorsR1, ratesR0, ratesR1, classIDs0, classIDs1 = compareMasks(
                r0, r1)

            print(ratesR0, ratesR1, classIDs0, classIDs1)

            class_names = ['old', 'smaller', 'bigger', 'same', 'new']

            visualize.save_instances(
                image0, r0['rois'], r0['masks'], classIDs0, class_names,  ratesR0,
                path=pred_path0, colors=colorsR0)
            visualize.save_instances(
                image1, r1['rois'], r1['masks'], classIDs1, class_names, ratesR1,
                path=pred_path1, colors=colorsR1)

            title = "Otomatik Follow-Up"
            cap = "Otomatik Follow-Up"
            abstract = "Yükelenen dosyaların uzman hekim görüşleri ile \
                        belirtilen plak(ları) ve bu plakların değişimlerinin otomatik tespiti\
                        aşağıda detaylı olarak görülmektedir."

            return render_template('followupPre.html', title=title, cap=cap, abstract=abstract,
                                   orjFile0=filename0, orjFile1=filename1,
                                   predFileName0=predFileName0, predFileName1=predFileName1
                                   )
    else:
        flash("bir hata oluştu","danger")
        return redirect('msOtoFollowUp')


@app.route('/msFollowUpCompareShow', methods=['POST'])
def msFollowUpCompareShow():
    if request.method == 'POST':
        # formdan dosya gelip gelmediğini kontrol edelim
        if ('firstMR' and 'secondMR') not in request.files:
            flash('Dosya seçilmedi', 'danger')
            return redirect('msFollowUpCompare')

        f0 = request.files['firstMR']
        f1 = request.files['secondMR']
        fJson = request.files['jsonfname']

        if f0.filename == '' or f1.filename == '' or fJson.filename=='':
            flash('Dosya seçilmedi', 'danger')
            return redirect('msFollowUpCompare')

        if((f0 and uzanti_kontrol(f0.filename)) and (f1 and uzanti_kontrol(f1.filename)) and (fJson and uzanti_kontrolJson(fJson.filename))) != True:
            flash('Dosya tipinde hata var. ', 'danger')
            return redirect('msFollowUpCompare')
        else:
            filename0 = secure_filename(f0.filename)
            filename1 = secure_filename(f1.filename)

            filepath0 = os.path.join(app.config['UPLOAD_FOLDER'], filename0)
            f0.save(filepath0)
            # flash('ilk MR görüntüsü başarıyla yüklendi.', 'success')

            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            f1.save(filepath1)
            # flash('ikinci MR görüntüsü başarıyla yüklendi.', 'success')
            # flash('Dosyalar başarıyla yüklendi.', 'success')

            class_names = ['BG', 'msMask']

            image1 = cv2.imread(filepath1)
            results1 = model.detect([image1], verbose=1)
            r1 = results1[0]
            predFileName1 = "pre_"+filename1.split('.')[0]+".jpg"
            pred_path1 = UPLOAD_PRED_PATH+"/"+predFileName1

            image0 = cv2.imread(filepath0)
            results0 = model.detect([image0], verbose=1)
            r0 = results0[0]
            predFileName0 = "pre_"+filename0.split('.')[0]+".jpg"
            pred_path0 = UPLOAD_PRED_PATH+"/"+predFileName0

            message, colorsR0, colorsR1, ratesR0, ratesR1, classIDs0, classIDs1 = compareMasks(
                r0, r1)

            print(ratesR0, ratesR1, classIDs0, classIDs1)

            class_names = ['old', 'smaller', 'bigger', 'same', 'new']

            visualize.save_instances(
                image0, r0['rois'], r0['masks'], classIDs0, class_names,  ratesR0,
                path=pred_path0, colors=colorsR0)
            visualize.save_instances(
                image1, r1['rois'], r1['masks'], classIDs1, class_names, ratesR1,
                path=pred_path1, colors=colorsR1)
            if fJson and uzanti_kontrolJson(fJson.filename):
                jsonFile = str(uuid.uuid4())+".json"
                filenameJ = secure_filename(jsonFile)
                filepathJ = os.path.join(
                    app.config['UPLOAD_FOLDER'], filenameJ)
                fJson.save(filepathJ)
                dataset = configFile.MsMaskDataset()
                print("Json:", filepathJ)
                dataset.sload_msMask(app.config['UPLOAD_FOLDER'], filepathJ)
                dataset.prepare()
                info = dataset.image_info[0]
                sifir = 0
                bir = 1
                if(info["id"] != f0.filename):
                    bir = 0
                    sifir = 1

                image0, image_meta0, gt_class_id0, gt_bbox0, gt_mask0 =\
                    modellib.load_image_gt(
                        dataset, config, sifir, use_mini_mask=False)

                GTFileName0 = "GT_"+filename0.split('.')[0]+".jpg"
                GTFilePath0 = UPLOAD_PRED_PATH+"/"+GTFileName0

                image1, image_meta1, gt_class_id1, gt_bbox1, gt_mask1 =\
                    modellib.load_image_gt(
                        dataset, config, bir, use_mini_mask=False)

                GTFileName1 = "GT_"+filename1.split('.')[0]+".jpg"
                GTFilePath1 = UPLOAD_PRED_PATH+"/"+GTFileName1

                # class_names = ['BG', 'msMask']
                flash("!!Uzman görüşlerinden alınan sonuclar!! ==> ","light")

                message, colorsR0, colorsR1, ratesR0, ratesR1, classIDs0, classIDs1 = compareM(
                    gt_mask0, gt_mask1)

                print(ratesR0, ratesR1, classIDs0, classIDs1)

                class_names = ['old', 'smaller', 'bigger', 'same', 'new']

                visualize.save_instances(
                    image0, gt_bbox0, gt_mask0, classIDs0, class_names, ratesR0,
                    path=GTFilePath0, colors=colorsR0)
                visualize.save_instances(
                    image1, gt_bbox1, gt_mask1, classIDs1, class_names, ratesR1,
                    path=GTFilePath1, colors=colorsR1)
                
                GTMatchFile0 = "GT_over_"+filename0.split('.')[0]+".jpg"
                GTMatchPath0 = UPLOAD_PRED_PATH+"/"+GTMatchFile0
                visualize.save_differences(image0, gt_bbox0, gt_class_id0, gt_mask0,
                                           r0['rois'], r0['class_ids'], r0['scores'], r0['masks'],
                                           dataset.class_names, path=GTMatchPath0,
                                           show_box=False
                                           )
                result = maskCompound(r0['masks'])
                reference = maskCompound(gt_mask0)

                dc = metreE.dc(result, reference)
                jc = metreE.jc(result, reference)
                iouX = utils.compute_overlaps_masks(result, reference)
                iou = iouX[0][0]
                print()
                voe = 1-iou
                vol = np.count_nonzero(result)
                if not(vol == 0):
                    asd = metreE.asd(result, reference)
                    assd = metreE.assd(result, reference)
                    flash("First MRI: DC:{:.2f}, JC:{:.2f}, VOE:{:.2f}, IOU:{:.2f}, ASD:{:.2f}, ASSD:{:.2f} ".format(
                        dc, jc, voe, iou, asd, assd), "light")
                else:
                    flash("First MRI: DC:{:.2f}, JC:{:.2f}, VOE:{:.2f}, IOU:{:.2f} ".format(
                        dc, jc, voe, iou), "light")

                GTMatchFile1 = "GT_over_"+filename1.split('.')[0]+".jpg"
                GTMatchPath1 = UPLOAD_PRED_PATH+"/"+GTMatchFile1
                visualize.save_differences(image1, gt_bbox1, gt_class_id1, gt_mask1,
                                           r1['rois'], r1['class_ids'], r1['scores'], r1['masks'],
                                           dataset.class_names, path=GTMatchPath1,
                                           show_box=False
                                           )
                result = maskCompound(r1['masks'])
                reference = maskCompound(gt_mask1)

                dc = metreE.dc(result, reference)
                jc = metreE.jc(result, reference)
                iouX = utils.compute_overlaps_masks(result, reference)
                iou = iouX[0][0]
                print()
                voe = 1-iou
                vol = np.count_nonzero(result)
                if not(vol == 0):
                    asd = metreE.asd(result, reference)
                    assd = metreE.assd(result, reference)
                    flash("Second MRI: DC:{:.2f}, JC:{:.2f}, VOE:{:.2f}, IOU:{:.2f}, ASD:{:.2f}, ASSD:{:.2f} ".format(
                        dc, jc, voe, iou, asd, assd), "light")
                else:
                    flash("Second MRI: DC:{:.2f}, JC:{:.2f}, VOE:{:.2f}, IOU:{:.2f} ".format(
                        dc, jc, voe, iou), "light")
                
            

            title = "Karşılaştırmalı Otomatik Follow-Up "
            cap = "Karşılaştırmalı Otomatik Follow-Up"
            abstract = "Yükelenen dosyaların uzman hekim görüşleri ile belirtilen plak(ları)\
                        sistemin otomatik tespit ettiği plak(lar) ve bu plakların değişimlerinin otomatik tespiti\
                        ,  aşağıda detaylı olarak görülmektedir."

            return render_template('followupPre.html', title=title, cap=cap, abstract=abstract,
                                   orjFile0=filename0, orjFile1=filename1,
                                   predFileName0=predFileName0, predFileName1=predFileName1,
                                   GTFileName0=GTFileName0, GTFileName1=GTFileName1,
                                   GTOverFileName0=GTMatchFile0, GTOverFileName1=GTMatchFile1
                                   )
    else:
        flash("bir hata oluştu","danger")
        return redirect('msFollowUpCompare')

@app.route('/about')
def about():
    title = "About"
    cap = "About - Test"
    return render_template('main.html', title=title, cap=cap)


if __name__ == "__main__":
    app.run(debug=True)
