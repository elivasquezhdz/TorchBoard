from flask import Flask,Response,send_file, render_template, url_for, send_from_directory,request,flash,redirect,make_response
from werkzeug.utils import secure_filename
from sprite_helper import *
from PIL import Image
import numpy as np
import base64
import cv2
import os
import io


app = Flask(__name__)
app.config['SECRET_KEY'] = '9881527bx0b13xr0c602dgde210wa328'

app.config['UPLOAD_FOLDER'] = "UPLOAD_FOLDER"

@app.route('/play')
def play():
    return render_template('play.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/picture')
def picture():
    return render_template('picture.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'bmp','jpg', 'jpeg','png'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_cam', methods=['GET','POST'])
def upload_cam():
    if request.method == 'POST':
        
        imagefile = request.form['file']
        b = str.encode(imagefile)
        z = b[b.find(b'/9'):]
        im = Image.open(io.BytesIO(base64.b64decode(z)))#.save('UPLOAD_FOLDER/captured.jpg')
        img = np.array(im) 
        # Convert RGB to BGR 
        img = img[:, :, ::-1].copy() 

        masked_img = get_masked_image(img)
        croped_image = crop_image(masked_img)
        resized = cv2.resize(croped_image, (150,450))
        board = cv2.imread("sprites/b3.png",cv2.IMREAD_UNCHANGED)
        dst = cv2.addWeighted(board ,1,resized,1,0)

        imgs = []
        imgs.append(dst)
        imgs.append(rotate_image(dst,3))
        imgs.append(rotate_image(dst,5))
        imgs.append(rotate_image(dst,3))
        imgs.append(dst)
        imgs.append(rotate_image(dst,-3))
        imgs.append(rotate_image(dst,-5))
        imgs.append(rotate_image(dst,-3))        
        sprites = np.concatenate(imgs, axis=1)
        cv2.imwrite("static/sprites.png",sprites)
        return redirect(url_for('play'))
    else:
        '''sprites = cv2.imread("sprites.png",cv2.IMREAD_UNCHANGED)
        os.remove("sprites.png")
        retval, buffer = cv2.imencode('.png', sprites)
        response = make_response(buffer.tobytes())
        response.headers["Content-Disposition"] = "attachment; filename=sprites.png"
        response.headers["Content-Type"] = "image/png"
        return response'''
        return redirect(url_for('play'))

@app.route('/download')
def download():
    pass

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Error','danger')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('Error','danger')
            return redirect(request.url)
        if not(allowed_file(file.filename)):
            flash('File type error', 'danger')
            return redirect(url_for('picture'))
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fname = os.path.join('UPLOAD_FOLDER',filename)
            img =cv2.imread(fname)
            masked_img = get_masked_image(img)
            croped_image = crop_image(masked_img)
            resized = cv2.resize(croped_image, (150,450))
            board = cv2.imread("sprites/b3.png",cv2.IMREAD_UNCHANGED)
            dst = cv2.addWeighted(board ,1,resized,1,0)

            imgs = []
            imgs.append(dst)
            imgs.append(rotate_image(dst,3))
            imgs.append(rotate_image(dst,5))
            imgs.append(rotate_image(dst,3))
            imgs.append(dst)
            imgs.append(rotate_image(dst,-3))
            imgs.append(rotate_image(dst,-5))
            imgs.append(rotate_image(dst,-3))        
            sprites = np.concatenate(imgs, axis=1)

            os.remove(fname)
            cv2.imwrite("static/sprites.png",sprites)
            return redirect(url_for('play'))
            '''
            retval, buffer = cv2.imencode('.png', sprites)
            response = make_response(buffer.tobytes())
            response.headers["Content-Disposition"] = "attachment; filename=sprites.png"
            response.headers["Content-Type"] = "image/png"
            return response
            #flash('File uploaded', 'success')
            #return redirect(url_for('index'))'''
    return redirect(url_for('picture'))

if __name__ == '__main__':
    app.run(debug=True)