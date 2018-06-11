import pandas as pd
import numpy as np
import cv2
from keras import backend
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.models import load_model

import os


##########
#target_model_time = "1527999111"
#target_layer_name = "conv2d_60"
##########
target_model_time = "1528320726"
target_layer_name = "conv2d_116"
##########


backend.set_learning_phase(1)

output_dir = "output"
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
if not os.path.exists(output_dir+"/"+target_model_time):
	os.mkdir(output_dir+"/"+target_model_time)

model = load_model("model/"+target_model_time+".h5")

f = open("model/"+target_model_time+"_train_list.txt", 'r')
for line in f.readlines():
	line = line.rstrip()
	dir, file = line.split('/')

	target_image_dir = output_dir+"/"+target_model_time+"/"+dir
	if not os.path.exists(target_image_dir):
		os.mkdir(target_image_dir)
	target_image_name = file

	x = img_to_array(load_img(dir+"/"+target_image_name, target_size=(64,64)))
	X = np.expand_dims(x, axis=0)
	X = X.astype('float32')
	preprocessed_input = X / 255.0

	predictions = model.predict(preprocessed_input)
	class_idx = np.argmax(predictions[0])
	class_output = model.output[:, class_idx]

	conv_output = model.get_layer(target_layer_name).output
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
	
	array_to_img(jetcam).save(target_image_dir+"/["+target_layer_name+"]_"+target_image_name)
f.close()

