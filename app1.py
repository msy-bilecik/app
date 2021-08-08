import os
from flask import Flask, render_template, url_for, request, redirect, abort, flash
from werkzeug.utils import secure_filename
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploadFolder'
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024
app.secret_key = "msy"

FILETYPES = set([ 'png', 'jpg', 'jpeg'])


def uzanti_kontrol(dosyaadi):
    return '.' in dosyaadi and \
        dosyaadi.rsplit('.', 1)[1].lower() in FILETYPES


def showDetecFile(filename):
    #title = "MS Detection"
    #cap = "MS Detection - Test"
    #fullFName = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #return render_template('detection.html', title=title, cap=cap+" "+filename, fname1=filename)
    return redirect(url_for('static', filename='uploadFolder/'+filename), code=301)


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
    #title = "MS Detection"
    #cap = "MS Detection - Test"
    #return render_template('detection.html', title=title, cap=cap+" "+fname, filename=fname)
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


if __name__ == "__main__":
    app.run(debug=True)
