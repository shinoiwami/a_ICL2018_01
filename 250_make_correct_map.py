import pandas as pd
import numpy as np
import cv2
from keras import backend
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras.models import load_model
from PIL import Image

import os
import re


output_map_dir = "output_map"
if not os.path.exists(output_map_dir):
	os.mkdir(output_map_dir)

embedded_pre = {}

f = open("target.txt", 'r')
for line in f.readlines():
	line = line.rstrip()
	cell = line.split('\t')
	
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

	res = '{0:02d}'.format(int(cell[4]))
	embedded_pre[id]['count'].setdefault(res, 0)
	embedded_pre[id]['count'][res] += 1
f.close()

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

fw = open(output_map_dir+"/correct.html", 'w')
f = open("target.tmpl", 'r')
for line in f.readlines():
	line = re.sub(r'<!--tmpl:embedded-->', ",".join(embedded), line)
	fw.write(line)
f.close()
fw.close()

