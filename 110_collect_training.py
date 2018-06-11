##################################################
# This collects images with URL list from ImageNet
#       ImageNet http://www.image-net.org
##################################################

import os
import requests
import shutil
from time import sleep

#####
# Setting
train_list = {"train_01":"church", "train_02":"mosque"}

#####
for key in train_list:
	if not os.path.exists(key):
		os.mkdir(key)

	i = 1
	f = open(key+".txt", 'r')
	for line in f.readlines():
		print i
		i += 1
		
		line = line.rstrip()
		filename = os.path.basename(line) or 'index.jpg'
		
		if os.path.exists(key+"/"+filename):
			continue

		try:
			r = requests.get(line, stream=True, timeout=5)
			if r.status_code == 200:
				with open(key+"/"+filename, 'wb') as f:
					r.raw.decode_content = True
					shutil.copyfileobj(r.raw, f)
				sleep(2)
		except requests.ConnectionError, e:
			print e


	f.close()
