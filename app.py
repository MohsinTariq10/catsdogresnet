from flask import Flask, render_template, request
import imageio
import numpy as np
import tensorflow.keras.models
import re
import sys
import os
import base64
from PIL import Image
from tensorflow.keras.backend import set_session

sys.path.append(os.path.abspath('./'))
from load import *
app = Flask(__name__)

global model, graph, sess
model, graph, sess = init()

def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)',imgData1).group(1)
    imgdata = base64.b64decode(imgstr)
    with open('output.jpg', 'wb') as output:
        output.write(imgdata)

def compressImage(path):
    im = Image.open(path)
    im = im.convert("RGB")
    new_im = im.resize((150,150),Image.ANTIALIAS)
    tokens = path.split('/')
    name = tokens[len(tokens)-1]
    new_im.save(path, 'JPEG', quality=90)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData.decode())
    compressImage('output.jpg')

    x = imageio.imread('output.jpg')
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    with graph.as_default():
        set_session(sess)
        out = model.predict(x/255)
        print('%.2f' % out[0][0])
        if(out[0][0] <= 0.5):
            print('cat')
            return 'cat ' + str('%.8f' % out[0][0]) 
        else:
            print('dog')
            return 'dog' + str('%.8f' % out[0][0])

if __name__ == '__main__':
    print('main yahan hon ')
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)
