import numpy as np 
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
from scipy.misc import imread, imresize, imshow
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.backend import set_session

def init():
    sess = tf.Session()
    graph = tf.get_default_graph()
    json_file = open('./model/model_adam.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    set_session(sess)
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model/model_adam.h5")
    print('loaded model from disk')

    loaded_model.compile(
        optimizer = RMSprop(lr=0.0001), loss='binary_crossentropy', metrics = ['acc']
    )
    

    return loaded_model, graph, sess
