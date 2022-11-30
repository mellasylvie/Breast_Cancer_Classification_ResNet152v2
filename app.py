from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('model_pruned.h5')
model.make_predict_function()

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def predict_image():
    imagefile = request.files['imagefile']
    image_path = "./static/" + imagefile.filename
    imagefile.save(image_path)

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224,224))
    im_array = np.asarray(image)
    im_array = im_array*(1/255)
    im_input = tf.reshape(im_array, shape = [1, 224, 224, 3])

    predict_label = np.argmax(model.predict(im_input))
    if predict_label == 0:
        predict_class = 'Benign'
    elif predict_label == 1:
        predict_class = 'Malignant'
    else:
        predict_class = 'Normal'

    predict_array = model.predict(im_input)[0]
    max_array = (np.max(predict_array)*100).round(2)

    classification = '%s (%.2f%%)' % (predict_class, max_array)

    return render_template('index.html', prediction=classification, image=image_path)


if __name__ == '__main__':
    app.run(port=3000, debug=True)