# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import utils
from py import metricQc
from py import metreE
#from py import deepmswebFx
import configFile
from flask import Flask, url_for, request, render_template, Response, jsonify, redirect, flash, abort
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np
import uuid

import os
import cv2

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
app.config['TEMPLATES_AUTO_RELOAD'] = True
UPLOAD_PRED_PATH = app.config['UPLOAD_FOLDER']
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024
app.secret_key = "msy"

FILETYPES = set(['png', 'jpg', 'jpeg'])
ColorSet = [(1.0, 1.0, 0.0), (0.5, 1.0, 0.0),  (1.0, 0.0, 0.0),
            (0.0, 0.5, 1.0), (1, 1, 1)]
olcekMetin = {}
olcekMetin["DC"] = "DC is a criterion calculated according to the overlap amount of the region placed on the pre-selected regions"
olcekMetin["VOE"] = "The VOE metric shows the error rate between the expert opinion and the masked region"
olcekMetin["LTPR"] = "LTPR"
olcekMetin["LFPR"] = "LFPR"
# (1.0, 1.0, 0.0) sar??
# (0.5, 1.0, 0.0) ye??il
# (1.0, 0.0, 0.0) k??rm??z??
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


def compareMasks(r1, r2):
    masks1 = r1['masks']
    masks2 = r2['masks']
    return compareM(masks1, masks2)


def compareM(masks1, masks2):
    messagesX = {}
    if(masks1.shape[0] == masks2.shape[0] and masks1.shape[1] == masks2.shape[1]):

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
                        messagesX["kuculmus"+str(tinyC)] = {
                            "message": "1 plak  %{:.2f} k??????lm????t??r.  ".format(rate), "type": "success"}
                        #"message": "1 plaque is %{: .2f} smaller than before. ".format(rate), "type": "success"}
                        # flash(" 1 plak  %{:.2f} k??????lm????t??r. ".format(rate), "success")
                    elif (bMatrix[i, t] >= 1.02):
                        bigC = bigC+1
                        zN[t] = 2
                        zO[i] = 2
                        rate = (bMatrix[i, t] - 1)*100
                        ratesR1[i] = rate
                        ratesR2[t] = rate
                        messagesX["buyumus"+str(bigC)] = {
                            "message": " 1 plakda %{:.2f} b??y??me g??zlenmi??tir.".format(rate), "type": "danger"}
                        #"message": " 1 plaque is %{:.2f} larger than before.".format(rate), "type": "danger"}
                        # flash(" 1 plakda %{:.2f} b??y??me g??zlenmi??tir. ".format(rate), "danger")

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
            messagesX["notEvulate"] = {
                "message": "De??erlendirme i??in yeterli benzerlik bulunamad??.", "type": "info"}
            #"message": "Not enough similarity found for evaluation.", "type": "info"}
           # flash("de??erlendirme i??in yeterli benzerlik bulunamad??. ", "info")
        elif(likeC == iy and likeC == ix):
            messagesX["degismemis"] = {
                "message": "Lezyonlarda de??i??im olmam????t??r.", "type": "success"}
           # "message": "There is no change in the plaque(s).", "type": "success"}
           # flash("lezyonlarda de??i??im olmam????t??r.", "success")
        else:
            if(likeC > 0):
                messagesX["benzerSayisi"] = {
                    "message": "{:.0f} plakda de??i??im olmam????t??r.".format(likeC), "type": "info"}
               # "message": "There was no change in {:.0f} plaque(s).".format(likeC), "type": "info"}
            #    flash("{:.0f} plakda de??i??im olmam????t??r.".format(likeC), "info")
            if(tinyC > 0):
                #message = " {:.0f} plak k??????lm????t??r.".format(tinyC)
                messagesX["kuculensayisi"] = {
                    "message": " {:.0f} plak k??????lm????t??r.".format(tinyC), "type": "success"}
               # "message": " {:.0f} plaque(s) smaller than before.".format(tinyC), "type": "success"}
            if(bigC > 0):
              #  message = message +  " {:.0f} plakda b??y??me g??zlenmi??tir.".format(bigC)
                messagesX["buyuyenSayisi"] = {
                    "message": " {:.0f} plakda b??y??me g??zlenmi??tir.".format(bigC), "type": "danger"}
                #"message": " {:.0f} plaque(s) bigger than before.".format(bigC), "type": "danger"}

            if(exC > 0):
               # "message": " {: .0f} plaque(s) not observed".format(exC), "type": "warning"}
                messagesX["yokOlan"] = {
                    "message": " {: .0f} plak g??zlenmemi??tir.".format(exC), "type": "warning"}
             #   flash(" {: .0f} plak g??zlenmemi??tir.".format(exC), "warning")
            if(newC > 0):
               # message = message + \                    " {: .0f} yeni plak tespit edilmi??tir.".format(newC)
                messagesX["yeniler"] = {
                    "message": " {: .0f} yeni plak(lar) tespit edildi.".format(newC), "type": "light"}
             #   flash(" {: .0f} yeni plak tespit edilmi??tir.".format(newC), "light")

    else:
        #message = "mismatched size was detected."
        messagesX["yokOlan"] = {
            "message": "uyumsuz boyut tespit edildi.", "type": "danger"}
        #flash("uyumsuz boyut", "danger")
        zN = zO = 0

    return messagesX, colorsR1, colorsR2, ratesR1, ratesR2, zO, zN


def colorSetting(colorM, ColorSet):
    colors = []
    for i in range(len(colorM)):
        colors.append(ColorSet[int(colorM[i])])

    return colors


@app.route('/')
def index():
    t = "DeepMSWeb"
    cap = "Homepage - Test"
    content = "Multiple Sclerosis Detection And Follow-up System Test Page"
    return render_template('main.html', title=t, cap=cap, content=content)


@app.route('/showPic/<filename>')
def detecFile(filename):
    return redirect(url_for('static', filename='uploadFolder/'+filename), code=301)


@app.route('/msDetectionCompare')
def msDetectionCompare():
    title = "DeepMSWeb - Otomatik MS Tespiti"
    cap = "Otomatik MS Tespiti"
    abstract = "MR g??r??nt??lerindeki MS plaklar??n?? otomatik olarak alg??layan ve bunlar?? uzman hekim g??r??????\
         ile kar????la??t??ran bu uygulama sayfas??. Bunun i??in MR g??r??nt??lerinin b??l??tleme bilgilerini\
              VGG 1.0.6 format??nda y??klemelisiniz. "
    fxUrl = url_for("msFinderCompare")
    json = True
    return render_template('detection.html', title=title, cap=cap, abstract1=abstract, fx=fxUrl, json1=json)


@app.route('/msFinderCompare', methods=['POST'])
def msFinderCompare():
    messages = {}
    if request.method == 'POST':
        # formdan dosya gelip gelmedi??ini kontrol edelim
        if 'fname' not in request.files:
            messages["fileNotSelected"] = {
                "message": "Dosya se??ilmemi??tir", "type": "danger"}
            return redirect('msDetectionCompare')

            # kullan??c?? dosya se??memi?? ve taray??c?? bo?? isim g??ndermi?? mi
        f = request.files['fname']
        if f.filename == '':
            messages["fileNotSelected"] = {
                "message": "Dosya se??ilmemi??tir", "type": "danger"}
            return redirect('msDetectionCompare')

            # gelen dosyay?? g??venlik ??nlemlerinden ge??ir
        if f and uzanti_kontrol(f.filename):

            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            f.save(filepath)

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

                olcekler1 = {}
                result = maskCompound(r['masks'])
                reference = maskCompound(gt_mask)

                olcekler1["DC"] = metreE.dc(result, reference)
                #olcekler1["JC"] = metreE.jc(result, reference)
                #olcekler1["IOU"] = utils.compute_overlaps_masks(result, reference)[0][0]
                iou = utils.compute_overlaps_masks(result, reference)[0][0]
                olcekler1["VOE"] = 1 - iou
                olcekler1["LTPR"] = metricQc.ltpr(result, reference)
                olcekler1["LFPR"] = metricQc.lfpr(result, reference)
                # vol = np.count_nonzero(result)
                # if not(vol == 0):
                #     olcekler1["ASD"] = metreE.asd(result, reference)
                #     olcekler1["ASSD"] = metreE.assd(result, reference)

                title = "DeepMSWeb - Otomatik MS Tespiti"
                cap = "Otomatik MS Tespiti"
                abstract = filename.split('.')[0]+" dosyas??nda incelemeler sonucunda; otomatik olarak\
                             alg??lanan MS plak(lar)??n??n t??m detaylar?? g??r??lmektedir."
                return render_template('detectionPre.html', title=title, cap=cap, abstract=abstract,
                                       orjFile=filename,   predFileName=predFileName,
                                       GTOverFileName=GTMatchFile,  GTFileName=GTFileName,
                                       olcekler1=olcekler1, olcekMetin=olcekMetin, messages=messages)

            else:
                messages["jsonEx"] = {
                    "message": "MS lezyonlar?? sadece otomatik olarak tespit edilmi??tir. \
                         Uzman g??r?????? dosyas?? y??klenmedi yada hatal??.", "type": "danger"}

            title = "DeepMSWeb - Otomatik MS Tespiti"
            cap = "Otomatik MS Tespiti"
            abstract = "??ncelemeler sonucunda;" + filename.split('.')[0]+" dosyas??n??n otomatik olarak alg??lanan \
                        MS plakalar?? ve ayr??nt??lar?? g??r??nt??lenmektedir."
            return render_template('detectionPre.html', title=title, cap=cap, abstract=abstract,
                                   orjFile=filename, predFileName=predFileName, messages=messages)

        else:
            messages["notAllowedFile"] = {
                "message": "Dosya izin verilen t??rde de??il", "type": "danger"}
            return redirect('msDetectionCompare')
    else:
        abort(401)


################# Follow -Up Links##########


@app.route('/msFollowUpCompare')
def msFollowUpCompare():
    title = "DeepMSWeb - Otomatik De??i??im Tespiti"
    cap = "Kar????la??t??rmal?? Otomatik De??i??im Tespiti"
    abstract = "Hekimler taraf??ndan i??aretlenen MS plaklar??n??n g??r??nt??lendi??i ve \
         iki farkl?? d??nemde ??ekilen g??r??nt??lerin kar????la??t??r??ld?????? uygulama sayfam??zd??r. \
             Bunun i??in segmentasyon dosyan??z?? MR b??l??m??nde ve VGG 1.0.6 format??nda y??klemelisiniz."
    fxUrl = url_for("msFollowUpCompareShow")
    json = True
    return render_template('followup.html',  title=title, cap=cap, abstract1=abstract, fx=fxUrl, json1=json)


@app.route('/msFollowUpCompareShow', methods=['POST'])
def msFollowUpCompareShow():
    messages = {}
    if request.method == 'POST':
        # formdan dosya gelip gelmedi??ini kontrol edelim
        if ('firstMR' and 'secondMR') not in request.files:
            messages["fileNotSelected"] = {
                "message": "Dosya se??ilmemi??tir", "type": "danger"}
            return redirect('msFollowUpCompare')

        f0 = request.files['firstMR']
        f1 = request.files['secondMR']
        fJson = request.files['jsonfname']

        if f0.filename == '' or f1.filename == '':
            messages["fileNotSelected"] = {
                "message": "Dosya se??ilmemi??tir", "type": "danger"}
            return redirect('msFollowUpCompare')

        # if((f0 and uzanti_kontrol(f0.filename)) and (f1 and uzanti_kontrol(f1.filename)) and (fJson and uzanti_kontrolJson(fJson.filename))) != True:
        #     messages["fileTypeError"] = {
        #        "message": "File type error", "type": "danger"}
        #    return redirect('msFollowUpCompare')
        # else:

        filename0 = secure_filename(f0.filename)
        filename1 = secure_filename(f1.filename)

        filepath0 = os.path.join(app.config['UPLOAD_FOLDER'], filename0)
        f0.save(filepath0)
        # flash('ilk MR g??r??nt??s?? ba??ar??yla y??klendi.', 'success')

        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        f1.save(filepath1)
        # flash('ikinci MR g??r??nt??s?? ba??ar??yla y??klendi.', 'success')
        # flash('Dosyalar ba??ar??yla y??klendi.', 'success')

        class_names = ['BG', 'msMask']

        image1 = cv2.imread(filepath1)
        results1 = model.detect([image1], verbose=1)
        r1 = results1[0]
        predFileName1 = "FU_pre_"+filename1.split('.')[0]+".jpg"
        pred_path1 = UPLOAD_PRED_PATH+"/"+predFileName1

        image0 = cv2.imread(filepath0)
        results0 = model.detect([image0], verbose=1)
        r0 = results0[0]
        predFileName0 = "FU_pre_"+filename0.split('.')[0]+".jpg"
        pred_path0 = UPLOAD_PRED_PATH+"/"+predFileName0

        messages1, colorsR0, colorsR1, ratesR0, ratesR1, classIDs0, classIDs1 = compareMasks(
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

            GTFileName0 = "FU_GT_"+filename0.split('.')[0]+".jpg"
            GTFilePath0 = UPLOAD_PRED_PATH+"/"+GTFileName0

            image1, image_meta1, gt_class_id1, gt_bbox1, gt_mask1 =\
                modellib.load_image_gt(
                    dataset, config, bir, use_mini_mask=False)

            GTFileName1 = "FU_GT_"+filename1.split('.')[0]+".jpg"
            GTFilePath1 = UPLOAD_PRED_PATH+"/"+GTFileName1

            # class_names = ['BG', 'msMask']
            # flash("Opinion", "light")

            messages2, colorsR0, colorsR1, ratesR0, ratesR1, classIDs0, classIDs1 = compareM(
                gt_mask0, gt_mask1)

            print(ratesR0, ratesR1, classIDs0, classIDs1)

            class_names = ['old', 'smaller', 'bigger', 'same', 'new']

            visualize.save_instances(
                image0, gt_bbox0, gt_mask0, classIDs0, class_names, ratesR0,
                path=GTFilePath0, colors=colorsR0)
            visualize.save_instances(
                image1, gt_bbox1, gt_mask1, classIDs1, class_names, ratesR1,
                path=GTFilePath1, colors=colorsR1)

            # first pic compare score
            GTMatchFile0 = "FU_GT_over_"+filename0.split('.')[0]+".jpg"
            GTMatchPath0 = UPLOAD_PRED_PATH+"/"+GTMatchFile0
            visualize.save_differences(image0, gt_bbox0, gt_class_id0, gt_mask0,
                                       r0['rois'], r0['class_ids'], r0['scores'], r0['masks'],
                                       dataset.class_names, path=GTMatchPath0,
                                       show_box=False
                                       )

            olcekler1 = {}
            result = maskCompound(r0['masks'])
            reference = maskCompound(gt_mask0)

            olcekler1["DC"] = metreE.dc(result, reference)
            # olcekler1["JC"] = metreE.jc(result, reference)
            # olcekler1["IOU"] = utils.compute_overlaps_masks(result, reference)[0][0]
            iou = utils.compute_overlaps_masks(result, reference)[0][0]
            olcekler1["VOE"] = 1 - iou
            olcekler1["LTPR"] = metricQc.ltpr(result, reference)
            olcekler1["LFPR"] = metricQc.lfpr(result, reference)
            # olcekler1["AVD"] = metricQc.avd(result, reference)
            # vol = np.count_nonzero(result)
            # if not(vol == 0):
            #     olcekler1["ASD"] = metreE.asd(result, reference)
            #     olcekler1["ASSD"] = metreE.assd(result, reference)

            # SECOND P??C COMPARE SCORE

            GTMatchFile1 = "FU_GT_over_"+filename1.split('.')[0]+".jpg"
            GTMatchPath1 = UPLOAD_PRED_PATH+"/"+GTMatchFile1
            visualize.save_differences(image1, gt_bbox1, gt_class_id1, gt_mask1,
                                       r1['rois'], r1['class_ids'], r1['scores'], r1['masks'],
                                       dataset.class_names, path=GTMatchPath1,
                                       show_box=False
                                       )

            result = maskCompound(r1['masks'])
            reference = maskCompound(gt_mask1)
            olcekler2 = {}

            olcekler2["DC"] = metreE.dc(result, reference)
            # olcekler2["JC"] = metreE.jc(result, reference)
            # olcekler2["IOU"] = utils.compute_overlaps_masks(result, reference)[0][0]
            iou = utils.compute_overlaps_masks(result, reference)[0][0]
            olcekler2["VOE"] = 1 - iou
            olcekler2["LTPR"] = metricQc.ltpr(result, reference)
            olcekler2["LFPR"] = metricQc.lfpr(result, reference)
            # olcekler2["AVD"] = metricQc.avd(result, reference)
            # vol = np.count_nonzero(result)
            # if not(vol == 0):
            #     olcekler2["ASD"] = metreE.asd(result, reference)
            #     olcekler2["ASSD"] = metreE.assd(result, reference)

            title = "Kar????la??t??rmal?? Otomatik De??i??im Tespiti"
            cap = "Kar????la??t??rmal?? Otomatik De??i??im Tespiti"
            abstract = "Y??kelenen dosyalar??n uzman hekim g??r????leri ile belirtilen plak(lar??)\
                        sistemin otomatik tespit etti??i plak(lar) ve bu plaklar??n de??i??imlerinin otomatik tespiti\
                        ,  a??a????da detayl?? olarak g??r??lmektedir."

            return render_template('followupPre.html', title=title, cap=cap, abstract=abstract,
                                   orjFile0=filename0, orjFile1=filename1,
                                   predFileName0=predFileName0, predFileName1=predFileName1,
                                   GTFileName0=GTFileName0, GTFileName1=GTFileName1,
                                   GTOverFileName0=GTMatchFile0, GTOverFileName1=GTMatchFile1,
                                   olcekler1=olcekler1, olcekler2=olcekler2, olcekMetin=olcekMetin,
                                   messages1=messages1, messages2=messages2, messages=messages
                                   )
        title = "Kar????la??t??rmal?? Otomatik De??i??im Tespiti"
        cap = "Kar????la??t??rmal?? Otomatik De??i??im Tespiti"
        abstract = "Y??klenen dosyalar??n sistem taraf??ndan otomatik olarak alg??lanan plak(lar)?? \
            ve bu plaklardaki de??i??ikliklerin otomatik tespiti a??a????da detayl?? olarak g??r??lmektedir."

        return render_template('followupPre.html', title=title, cap=cap, abstract=abstract,
                               orjFile0=filename0, orjFile1=filename1,
                               predFileName0=predFileName0, predFileName1=predFileName1,
                               messages=messages1
                               )

    else:
        flash("Bir Hata", "danger")
        return redirect('msFollowUpCompare')


@app.route('/about')
def about():
    title = "About"
    cap = "About - Test"
    return render_template('main.html', title=title, cap=cap)


if __name__ == "__main__":
    app.run(debug=True)
