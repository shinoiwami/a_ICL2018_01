from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import re


#####
# read training data
label_list_init = ["01", "02"]
label_list = []
image_list = []

N = 50
for label in label_list_init:
	i = 0
	for file in os.listdir("train_"+label):
		if file == ".DS_Store":
			continue
		i += 1
		if i > N:
			break

		print file
		label_list.append(label)
		filepath = "train_"+label + "/" + file
		image = np.array(Image.open(filepath).resize((25, 25)))
		image = image.transpose(2, 0, 1)
		image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
		image_list.append(image / 255.)

X = np.array(image_list)
Y = to_categorical(label_list)


#####
# make a model of NN, compile and execute
model = Sequential()
#model.add(Dense(200, input_dim=1875))
model.add(Dense(200, input_dim=1875))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(len(label_list_init)+1))
model.add(Activation("softmax"))

opt = Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(X, Y, nb_epoch=1500, batch_size=100, validation_split=0.1)


#####
# judge
print "####################"

#####
#for file in os.listdir("test"):
#	if file == ".DS_Store":
#		continue

#	filepath = "test" + "/" + file
#	image = np.array(Image.open(filepath).resize((25, 25)))
#	image = image.transpose(2, 0, 1)
#	image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
#	result = model.predict_classes(np.array([image / 255.]))

#	print result, file

#####
embedded = []
f = open("test.txt", 'r')
for line in f.readlines():
	line = line.rstrip()
	cell = line.split('\t')

	filepath = "test" + "/" + cell[0]
	image = np.array(Image.open(filepath).resize((25, 25)))
	image = image.transpose(2, 0, 1)
	image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
	result = model.predict_classes(np.array([image / 255.]))

	spot_str = '["'+cell[1]+'", '+cell[2]+', '+cell[3]+', '
	if result[0] == 1:
		spot_str += 'image_01'
	if result[0] == 2:
		spot_str += 'image_02'
	spot_str += ']'
	embedded.append(spot_str)

f.close()

output_file = "output.html"
fw = open(output_file, 'w')
f = open("test.tmpl", 'r')
for line in f.readlines():
	line = re.sub(r'<!--tmpl:embedded-->', ",".join(embedded), line)
	fw.write(line)
f.close()
fw.close()

