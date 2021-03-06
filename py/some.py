import numpy as np


def compareM(masks1, masks2):
    if(masks1.shape[0] == masks2.shape[0] and masks1.shape[1] == masks2.shape[1]):
        message = ""
        ix = masks1.shape[2]
        iy = masks2.shape[2]

        bMatrix = np.zeros((ix, iy))
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
                    sScore = mask2.sum()/mask1.sum()
                    bMatrix[i, t] = sScore
                t = t+1
            i = i+1
    return bMatrix


ColorSet = [(1.0, 1.0, 0.0), (0.5, 1.0, 0.0),  (1.0, 0.0, 0.0),
            (0.0, 0.5, 1.0), (1, 1, 1)]


def colorSetting(colorM, ColorSet):
    colors = []
    for i in range(len(colorM)):
        colors.append(ColorSet[int(colorM[i])])

    return colors


def compareMessage(bMatrix, treshold):
    ix = bMatrix.shape[1]
    iy = bMatrix.shape[2]
    zN = np.zeros(iy).astype(int)+4
    zO = np.zeros(ix).astype(int)

    ratesR1 = np.zeros(ix)
    ratesR2 = np.zeros(iy)

    zNew = bMatrix.sum(axis=0)
    zOld = bMatrix.sum(axis=1)
    exC = zOld.size-np.count_nonzero(zOld)
    newC = zNew.size-np.count_nonzero(zNew)
    altesik = 1-treshold
    ustesik = 1+treshold

    message = {"message": "", "type": ""}

    messages = {}
    for i in range(ix):
        for t in range(iy):
            sScore = bMatrix[i, t]
            if (sScore > altesik and sScore < ustesik):
                likeC = likeC+1
                zN[t] = 3
                zO[i] = 3
                ratesR1[i] = 0
                ratesR2[t] = 0
            elif (sScore <= altesik):
                tinyC = tinyC+1
                zN[t] = 1
                zO[i] = 1
                rate = (1 - sScore)*100
                ratesR1[i] = rate
                ratesR2[t] = rate
                message = {
                    "message": " 1 plak  %{:.2f} k??????lm????t??r. ".format(rate), "type": "success"}
                messages = messages+{"kuculen"+tinyC: message}
            elif (sScore >= ustesik):
                bigC = bigC+1
                zN[t] = 2
                zO[i] = 2
                rate = (sScore - 1)*100
                ratesR1[i] = rate
                ratesR2[t] = rate
                # flash(" 1 plakda %{:.2f} b??y??me g??zlenmi??tir. ".format(rate), "danger")

    for i in range(ix):
        if (zOld[i] == 0):
            ratesR1[i] = 0
    for i in range(iy):
        if (zNew[i] == 0):
            ratesR2[i] = 0

    if(likeC == 0 and bigC == 0 and tinyC == 0):
        message = {
            "message": "de??erlendirme i??in yeterli benzerlik bulunamad??. ", "type": "info"}
        messages = messages+{"benzemiyor": message}
       # flash("de??erlendirme i??in yeterli benzerlik bulunamad??. ", "info")
    elif(likeC == iy and likeC == ix):
        message = "lezyonlarda de??i??im olmam????t??r."
        message = {
            "message": "lezyonlarda de??i??im olmam????t??r. ", "type": "success"}
        messages = {"hicDegismemis": message}
       # flash("lezyonlarda de??i??im olmam????t??r.", "success")
    else:
        if(likeC > 0):
            message = {
                "message": "{} plakda de??i??im olmam????t??r.".format(likeC), "type": "info"}
            messages = messages+{"toplamBenzer": message}
        #    flash("{} plakda de??i??im olmam????t??r.".format(likeC), "info")
        if(tinyC > 0):
            message = {
                "message": " {} plak k??????lm????t??r.".format(tinyC), "type": "info"}
            messages = messages+{"toplamK??c??len": message}
        if(bigC > 0):
            message = {
                "message": " {} plakda b??y??me g??zlenmi??tir.".format(bigC), "type": "info"}
            messages = messages+{"toplamB??y??yen": message}
        if(exC > 0):
            message = {
                "message": " {: .0f} plak g??zlenmemi??tir.".format(exC), "type": "warning"}
            messages = messages+{"toplamYokolan": message}
        if(newC > 0):

            message = {
                "message":  " {: .0f} yeni plak tespit edilmi??tir.".format(newC), "type": "light"}
            messages = messages+{"toplamYeni": message}

    colorsR2 = colorSetting(zN, ColorSet)
    colorsR1 = colorSetting(zO, ColorSet)
    return
