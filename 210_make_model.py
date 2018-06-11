import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
#from keras.utils.np_utils import to_categorical
from keras import optimizers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

import os
import re
import random
import time


trial_log = "210_log.txt"
trial_times = 1000


# https://keras.io/ja/optimizers/
# "rmsprop", "nadam"
opts = {
	"sgd":optimizers.SGD(lr=0.01),
	"rmsprop":optimizers.RMSprop(lr=0.001),
	"adagrad":optimizers.Adagrad(lr=0.01),
	"adadelta":optimizers.Adadelta(lr=1.0),
	"adam":optimizers.Adam(lr=0.001),
	"adamax":optimizers.Adamax(lr=0.002),
	"nadam":optimizers.Nadam(lr=0.002)
}
#opts = {"adamax":optimizers.Adamax(lr=0.002)}			# (adamax)

# https://keras.io/ja/activations/
acts = ["softmax", "elu", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
#acts = ["relu"]


if not os.path.exists("model"):
	os.mkdir("model")

for Z in range(0,trial_times):
	t0 = int(time.time())
	output_time = str(int(time.time()))
	MODEL_PATH = ""
	
	#####
	# tune values
	N = (1+int(random.random()*5)) * 100			# 100-500: number of training data per label
	NEWRON = 2 ** (3+int(random.random()*6))		# 8-256: number of newrons
#	HIDDEN_LAYER = 1+int(random.random()*2)			# 1-3: times of additional hidden layer
	HIDDEN_LAYER = 3
	DROPOUT = int(random.random()*5) / 10.0			# 0-0.5: dropout
	EPOCHS = (1+int(random.random()*4)) * 50		# 50-200
	BATCH_SIZE = (1+int(random.random()*4)) * 50	# 50-200

	i = 0
	opt_num = int(random.random()*len(opts))
	if opt_num == len(opts):
		opt_num = 0
	for OPT_KEY in sorted(opts):
		if i == opt_num:
			OPT = opts[OPT_KEY]
			break
		i += 1


	#####
	print "####################"
	print "[N, newron, hidden_layer, dropout, optimizer, epochs, batch_size, model_path]", N, NEWRON, HIDDEN_LAYER, DROPOUT, OPT_KEY, EPOCHS, BATCH_SIZE, MODEL_PATH


	#####
	# read training data
	label_list_init = ["01", "02"]
	label_list = []
	image_list = []
	nb_classes = len(label_list_init)+1

	fw = open("model/"+output_time+"_train_list.txt", 'w')
	for label in label_list_init:
		i = 0
		for file in list_pictures("train_"+label):
			if ".DS_Store" in file:
				continue
			i += 1
			if i > N:
				break

			img = img_to_array(load_img(file, target_size=(64,64)))
			image_list.append(img)
			label_list.append(int(label))
			fw.write(file+'\n')
	fw.close()

	X = np.asarray(image_list)
	Y = np.asarray(label_list)
	X = X.astype('float32')
	X = X / 255.0
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.0)
	Y_train = np_utils.to_categorical(Y_train, nb_classes)
	Y_test = np_utils.to_categorical(Y_test, nb_classes)


	#####
	# make a model of CNN, compile and execute
	model = Sequential()

	act_num = int(random.random()*len(acts))
	if act_num == len(acts):
		act_num = 0
	ACT = acts[act_num]
	model.add(Conv2D(NEWRON, (3, 3), activation=ACT, input_shape=X_train.shape[1:]))
	MODEL_PATH += "C2D("+ACT+")"
	model.add(MaxPooling2D(pool_size=(2, 2)))
	MODEL_PATH += "->MP2D"
	model.add(Dropout(DROPOUT))
	MODEL_PATH += "->DO"

	for h in range(HIDDEN_LAYER):
		print h
		act_num = int(random.random()*len(acts))
		if act_num == len(acts):
			act_num = 0
		ACT = acts[act_num]
		model.add(Conv2D(NEWRON, (3, 3), activation=ACT))
		MODEL_PATH += "->C2D("+ACT+")"
		model.add(MaxPooling2D(pool_size=(2, 2)))
		MODEL_PATH += "->MP2D"
		model.add(Dropout(DROPOUT))
		MODEL_PATH += "->DO"

	model.add(Flatten())
	MODEL_PATH += "->FL"
	model.add(Dense(nb_classes, activation='softmax'))
	MODEL_PATH += "->DE(softmax)"

	model.compile(loss="binary_crossentropy", optimizer=OPT, metrics=["accuracy"])
	model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data = (X_train, Y_train), verbose=1)

	#####
	# evaluation
	print "####################"
	print "# Evaluating data accuracy"

	loss, accuracy_0 = model.evaluate(X_train, Y_train)
	print 'loss: ', loss, '\naccuracy: ', accuracy_0
	accuracy_0 = round(accuracy_0, 4)

	#####
	# save
	json_string = model.to_json()
	open("model/"+output_time+".json", 'w').write(json_string)
	model.save("model/"+output_time+".h5")


	#####
	# classification
	print "####################"
	print "# Classify real data accuracy"

	count_total_1 = 0
	count_correct_1 = 0
	accuracy_1 = 0.0
	f = open("target.txt", 'r')
	for line in f.readlines():
		line = line.rstrip()
		cell = line.split('\t')

		filepath = "target" + "/" + cell[0]
		image = np.array(Image.open(filepath).resize((64, 64)))
		result = model.predict_classes(np.array([image / 255.]))

		count_total_1 += 1
		if result[0] == int(cell[4]):
			count_correct_1 += 1
		print result[0], cell[1]

	f.close()
	accuracy_1 = count_correct_1 / float(count_total_1)
	print accuracy_1
	accuracy_1 = round(accuracy_1, 4)


	#####
	print "####################"
	print "[N, newron, hidden_layer, dropout, optimizer, epochs, batch_size, model_path]", N, NEWRON, HIDDEN_LAYER, DROPOUT, OPT_KEY, EPOCHS, BATCH_SIZE, MODEL_PATH
	print accuracy_0, accuracy_1


	#####
	# log
	t1 = int(time.time())
	fa = open(trial_log, 'a')
	line = output_time+'\t'+str(N)+'\t'+str(NEWRON)+'\t'+str(HIDDEN_LAYER)+'\t'+str(DROPOUT)+'\t'+str(EPOCHS)+'\t'+str(BATCH_SIZE)+'\t'+str(accuracy_0)+'\t'+str(accuracy_1)+'\t'+str(t1-t0)+'\t'+OPT_KEY+'\t'+MODEL_PATH+'\n'
	fa.write(line)
	fa.close()
	print "####################"
