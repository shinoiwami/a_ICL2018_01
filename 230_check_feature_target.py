import pandas as pd
import numpy as np
import cv2
from keras import backend
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.models import load_model
from PIL import Image

import os
import re


##########
#target_model_time = "1527999111"
#target_layer_name = "conv2d_60"
##########
target_model_time = "1528320726"
target_layer_name = "conv2d_116"
##########


backend.set_learning_phase(1)

output_dir = "output2"
output_map_dir = "output_map"
if not os.path.exists(output_dir):
	os.mkdir(output_dir)
if not os.path.exists(output_dir+"/"+target_model_time):
	os.mkdir(output_dir+"/"+target_model_time)
if not os.path.exists(output_map_dir):
	os.mkdir(output_map_dir)

model = load_model("model/"+target_model_time+".h5")

count_total_1 = 0
count_correct_1 = 0
accuracy_1 = 0.0
embedded_pre = {}

f = open("target.txt", 'r')
for line in f.readlines():
	line = line.rstrip()
	cell = line.split('\t')

	target_image_dir = output_dir+"/"+target_model_time
	target_image_name = cell[0]
	
	# prepare to calculate accuracy and mapping
	id = str(int(float(cell[2])))+"_"+str(int(float(cell[3])))
	embedded_pre.setdefault(id, {})
	embedded_pre[id].setdefault('count_sum', 0)
	embedded_pre[id]['count_sum'] += 1
	embedded_pre[id].setdefault('lat_sum', 0.0)
	embedded_pre[id]['lat_sum'] += float(cell[2])
	embedded_pre[id].setdefault('lng_sum', 0.0)
	embedded_pre[id]['lng_sum'] += float(cell[3])
	embedded_pre[id].setdefault('count', {})

	image = np.array(Image.open("target/"+target_image_name).resize((64, 64)))
	result = model.predict_classes(np.array([image / 255.]))
	res = '{0:02d}'.format(result[0])
	embedded_pre[id]['count'].setdefault(res, 0)
	embedded_pre[id]['count'][res] += 1
		
	count_total_1 += 1
	if result[0] == int(cell[4]):
		count_correct_1 += 1
	print result[0], cell[1]

	# Grad-CAM
	x = img_to_array(load_img("target/"+target_image_name, target_size=(64,64)))
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
accuracy_1 = count_correct_1 / float(count_total_1)
print accuracy_1
accuracy_1 = round(accuracy_1, 4)

# plot results on a map
embedded = []
for id in embedded_pre.keys():
	max_val = max(embedded_pre[id]['count'].values())
	keys_of_max_val = [key for key in embedded_pre[id]['count'] if embedded_pre[id]['count'][key] == max_val]
	lat = embedded_pre[id]['lat_sum'] / embedded_pre[id]['count_sum']
	lng = embedded_pre[id]['lng_sum'] / embedded_pre[id]['count_sum']
		
	if len(keys_of_max_val) > 1:
		spot_str = '["00", '+str(lat)+', '+str(lng)+', image_00]'
	else:
		spot_str = '["'+keys_of_max_val[0]+'", '+str(lat)+', '+str(lng)+', image_'+keys_of_max_val[0]+']'
	embedded.append(spot_str)

fw = open(output_map_dir+"/"+target_model_time+".html", 'w')
f = open("target.tmpl", 'r')
for line in f.readlines():
	line = re.sub(r'<!--tmpl:embedded-->', ",".join(embedded), line)
	fw.write(line)
f.close()
fw.close()

