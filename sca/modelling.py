import sys, argparse
from scipy.spatial.distance import cdist
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os

# models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# CNN 
import sys
from keras.layers import Flatten, Dense, Reshape, Conv1D
from keras.models import Sequential
import keras

# plot_dataset
# INPUTS
#     features - the features of the training data
#     labels   - the labels of the training data
#     test_features - the test features
#     test_labels   - the labels of the test dataset
# OUTPUTS
# FUNCTIONALITY
#     creates a 3d plot of the data
def plot_dataset(features, labels, test_features, test_labels):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(features[:,0], features[:,1], zs=labels, c='pink')
	ax.scatter(test_features[:,0], test_features[:,1], zs=test_labels, c='black')

	plt.xticks(())
	plt.yticks(())

	plt.show()

# extract_data
# INPUTS
#     files - the files to extract data from
# OUTPUTS
#     features - the features of the dataset
#     labels   - the labels of the data
# FUNCTIONALITY
#     this function uses scio to load the mat files and extract the features and labels that 
#     are relevant for the models
def extract_data(feature_set, label_set):
  first = 1
  x = scio.loadmat(feature_set)
  newkeys = x.keys()
  for pop in ['__version__', '__header__', '__globals__']:
    newkeys.remove(pop)
  
  for key in newkeys:
    for item in x[key]:
      if first:
        features = np.array([item])
        first = 0
      else:
        features = np.vstack((features, [item]))

  first = 1
  # load data
  x = scio.loadmat(label_set)
  newkeys = x.keys()
  for pop in ['__version__', '__header__', '__globals__']:
    newkeys.remove(pop)

  print newkeys
  for key in newkeys:
    for item in x[key]:
      for i in range(len(item)):
        if first:
          labels = np.array([item[i]])
          first = 0
        else:
          labels = np.vstack((labels, [item[i]]))
          
  
  if debug:
    print features.shape
    print features
    print labels.shape
    print labels
    
  return features, labels

# train_model_CNN
# INPUTS
#     train        - the features of the training data
#     train_labels - the labels of the training data
#     test         - the values to test the model on
#     test_labels  - the labels of the test dataset to test accuracy of the model
# OUTPUTS
#     score - the score of the models accuracy, generated using the test data and labels
# FUNCTIONALITY
def train_model_CNN(train, train_labels, test, test_labels):
	model = Sequential()
	model.add(Dense(32, activation='relu', input_dim=len(train[0])))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(optimizer='rmsprop',
		      loss='binary_crossentropy',
		      metrics=['accuracy'])

	fitted = model.fit(train, train_labels)
	score = model.evaluate(test, test_labels)

	if debug:
		print 'Mean Accuracy: ', score

	return score


# train_model_RF
# INPUTS
#     train        - the features of the training data
#     train_labels - the labels of the training data
#     test         - the values to test the model on
#     test_labels  - the labels of the test dataset to test accuracy of the model
# OUTPUTS
#     score - the score of the models accuracy, generated using the test data and labels
# FUNCTIONALITY
def train_model_RF(train, train_labels, test, test_labels):
	model = RandomForestClassifier(n_estimators=10, min_samples_leaf=30)
	fitted = model.fit(train, train_labels)
	predicted = fitted.predict(test)
	score = fitted.score(test, test_labels)

	if debug:
		print 'Predicted: ', predicted
		print 'Mean Accuracy: ', score

	return score

# train_model_SVM
# INPUTS
#     train        - the features of the training data
#     train_labels - the labels of the training data
#     test         - the values to test the model on
#     test_labels  - the labels of the test dataset to test accuracy of the model
# OUTPUTS
#     score - the score of the models accuracy, generated using the test data and labels
# FUNCTIONALITY
def train_model_SVM(train, train_labels, test, test_labels):
	model = SVC(kernel='linear')
	fitted = model.fit(train, train_labels)
	predicted = fitted.predict(test)
	score = fitted.score(test, test_labels)

	if debug:
		print 'Predicted: ', predicted
		print 'Mean Accuracy: ', score

	return score

# train_model_KNN
# INPUTS
#     train        - the features of the training data
#     train_labels - the labels of the training data
#     test         - the values to test the model on
#     test_labels  - the labels of the test dataset to test accuracy of the model
# OUTPUTS
#     score - the score of the models accuracy, generated using the test data and labels
# FUNCTIONALITY
def train_model_KNN(train, train_labels, test, test_labels):
	model = KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='ball_tree', p=2)
	fitted = model.fit(train, train_labels)
	predicted = fitted.predict(test)
	score = fitted.score(test, test_labels)

	if debug:
		print 'Predicted: ', predicted
		print 'Mean Accuracy: ', score

	return score
	
# train_model_DTree
# INPUTS
#     train        - the features of the training data
#     train_labels - the labels of the training data
#     test         - the values to test the model on
#     test_labels  - the labels of the test dataset to test accuracy of the model
# OUTPUTS
#     score - the score of the models accuracy, generated using the test data and labels
# FUNCTIONALITY
def train_model_DTree(train, train_labels, test, test_labels):
	model = DecisionTreeClassifier()
	fitted = model.fit(train, train_labels)
	predicted = fitted.predict(test)
	score = fitted.score(test, test_labels)

	if debug:
		print 'Predicted: ', predicted
		print 'Mean Accuracy: ', score

	return score


# MAIN function
def main():
	# set up the argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--training_features', help='the training features; .mat file', required=1)
	parser.add_argument('-l', '--training_labels', help='the training labels; .mat file', required=1)
	parser.add_argument('-e', '--testing_features', help='the testing features; .mat file', required=1)
	parser.add_argument('-a', '--testing_labels', help='the testing labels; .mat file', required=1)
	parser.add_argument('-m', '--method', help='either <CNN>, <SVM>, <KNN>, <DTree>, <RF>', required=1)
	parser.add_argument('-s', '--sound', action='store_true', help='play a sound when its done')
	parser.add_argument('-d', '--debug', action='store_true', help='debug')

	args = parser.parse_args()

	global debug
	debug = args.debug

	# extract the data from the training and testing datasets
	train_features, train_labels = extract_data(args.training_features, args.training_labels)
	test_features, test_labels = extract_data(args.testing_features, args.testing_labels)
  
	if debug:
		print 'training: \n', train_features, '\n', train_labels
		print 'testing: \n', test_features, '\n', test_labels
	
	# use CNN to create a model
	if args.method == 'CNN':
		print 'Using CNN to solve...'

		# train the model
		score = train_model_CNN(train_features, train_labels, test_features, test_labels)
			
    # play a sound so i know it's done and can start a new one
		if args.sound:
		  os.system('play -nq -t alsa synth 1 sine 600')
	
	# use SVM to create a model
	elif args.method == 'SVM':
		print 'Using SVM to solve...'

		# train the model
		score = train_model_SVM(train_features, train_labels, test_features, test_labels)
			
    # play a sound so i know it's done and can start a new one
		if args.sound:
		  os.system('play -nq -t alsa synth 1 sine 600')

	# use KNN to create a model
	elif args.method == 'KNN':
		print 'Using KNN to solve...'
		
		# train the model
		score = train_model_KNN(train_features, train_labels, test_features, test_labels)
			
    # play a sound so i know it's done and can start a new one
		if args.sound:
		  os.system('play -nq -t alsa synth 1 sine 600')

	# use DTree to create a model
	elif args.method == 'DTree':
		print 'Using DTree to solve...'
		
		# train the model
		score = train_model_DTree(train_features, train_labels, test_features, test_labels)
			
    # play a sound so i know it's done and can start a new one
		if args.sound:
		  os.system('play -nq -t alsa synth 1 sine 600')

	# use Other to create a model
	elif args.method == 'RF':
		print 'Using Random Forest to solve...'
		
		# train the model
		score = train_model_RF(train_features, train_labels, test_features, test_labels)
			
    # play a sound so i know it's done and can start a new one
		if args.sound:
		  os.system('play -nq -t alsa synth 1 sine 600')


	# 
	else:
		print 'ERROR: <CNN>, <SVM>, <KNN>, <DTree>, <RF>'


main()
