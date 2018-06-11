import pandas as pd
import numpy as np
import cv2
from keras import backend
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.models import load_model

import os


##########
#target_model_time = "1527999111"
#target_layer_names = ["conv2d_57", "conv2d_58", "conv2d_59", "conv2d_60"]
#image_file_names = ["IMG_20180126_104147.jpg", "IMG_20180308_114826.jpg", "IMG_20180130_151120.jpg"]
##########
target_model_time = "1528320726"
target_layer_names = ["conv2d_113", "conv2d_114", "conv2d_115", "conv2d_116"]
image_file_names = ["IMG_20180126_104147.jpg", "IMG_20180308_114826.jpg", "IMG_20180130_151120.jpg"]
##########


backend.set_learning_phase(1)

output_dir = "output3"
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
if not os.path.exists(output_dir+"/"+target_model_time):
	os.mkdir(output_dir+"/"+target_model_time)

model = load_model("model/"+target_model_time+".h5")

for file in image_file_names:
	target_image_name = file

	x = img_to_array(load_img("target/"+target_image_name, target_size=(64,64)))
	X = np.expand_dims(x, axis=0)
	X = X.astype('float32')
	preprocessed_input = X / 255.0

	predictions = model.predict(preprocessed_input)
	class_idx = np.argmax(predictions[0])
	class_output = model.output[:, class_idx]

	for layer in target_layer_names:
		print layer
		conv_output = model.get_layer(layer).output
		grads = backend.gradients(class_output, conv_output)[0]
		gradient_function = backend.function([model.input], [conv_output, grads])

		output, grads_val = gradient_function([preprocessed_input])
		output, grads_val = output[0], grads_val[0]

		weights = np.mean(grads_val, axis=(0, 1))
		cam = np.dot(output, weights)
		cam = cv2.resize(cam, (64,64), cv2.INTER_LINEAR)
		cam = np.maximum(cam, 0)
		cam = cam / cam.max()

		jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
		jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
		jetcam = (np.float32(jetcam) + x / 2)
	
		array_to_img(jetcam).save(output_dir+"/"+target_model_time+"/["+layer+"]_"+target_image_name)

