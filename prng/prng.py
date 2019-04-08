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

# plot_dataset
# INPUTS
#     features - the features of the training data
#     labels   - the labels of the training data
#     test_features - the test features
#     test_labels   - the labels of the test dataset
# OUTPUTS
# FUNCTIONALITY
#     creates a 3d plot of the data
def plot_dataset(train_features, train_labels, test_features, test_labels):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(train_features[:,0], train_features[:,1], zs=train_labels, c='pink')
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
def extract_data(files, bits):
  first = 1

  for dataset in files:
  # load data
    with open(dataset) as csv_file:
      reader = csv.reader(csv_file, delimiter=',')
      for row in reader:
        if first:
          array = []
          for i in range(bits):
            array.append(row[i])
          features = np.array(array)
          labels = np.array([row[bits]])
          first = 0
        else:
          array = []
          for i in range(bits):
            array.append(row[i])
          features = np.vstack((features, array))
          labels = np.append(labels, [row[bits]])

  print features
  print labels
  return features, labels



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
	model = SVC(kernel='linear' degree=5)
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
	parser.add_argument('-t', '--training', metavar='T', nargs='+', help='the training datasets; .txt or .csv file', required=1)
	parser.add_argument('-b', '--bits', help='the number of bits in the prn', type=int, required=1)
	parser.add_argument('-e', '--testing', nargs=1, help='the testing dataset; .mat file', required=1) 
	parser.add_argument('-m', '--method', help='either <CNN>, <SVM>, <KNN>, <DTree>, <Other>', required=1)
	parser.add_argument('-p', '--plot', action='store_true', help='plot the datasets')
	parser.add_argument('-s', '--sound', action='store_true', help='play a sound when its done')
	parser.add_argument('-d', '--debug', action='store_true', help='debug')

	args = parser.parse_args()

	global debug
	debug = args.debug

	# extract the data from the training and testing datasets
	train_features, train_labels = extract_data(args.training, args.bits)
	test_features, test_labels = extract_data(args.testing, args.bits)

	if debug:
		print 'training: \n', train_features, '\n', train_labels
		print 'testing: \n', test_features, '\n', test_labels
	
	# use CNN to create a model
	if args.method == 'CNN':
		print 'Using CNN to solve...'
	
	# use SVM to create a model
	elif args.method == 'SVM':
		print 'Using SVM to solve...'

		# train the model
		score = train_model_SVM(train_features, train_labels, test_features, test_labels)

		# plot the training and testing datasets on a 3d plane
		if args.plot:
			plot_dataset(train_features, train_labels, test_features, test_labels)
			
    # play a sound so i know it's done and can start a new one
		if args.sound:
		  os.system('play -nq -t alsa synth 1 sine 600')

	# use KNN to create a model
	elif args.method == 'KNN':
		print 'Using KNN to solve...'
		
		# train the model
		score = train_model_KNN(train_features, train_labels, test_features, test_labels)

		# plot the training and testing datasets on a 3d plane
		if args.plot:
			plot_dataset(train_features, train_labels, test_features, test_labels)
			
    # play a sound so i know it's done and can start a new one
		if args.sound:
		  os.system('play -nq -t alsa synth 1 sine 600')

	# use DTree to create a model
	elif args.method == 'DTree':
		print 'Using DTree to solve...'
		
		# train the model
		score = train_model_DTree(train_features, train_labels, test_features, test_labels)

		# plot the training and testing datasets on a 3d plane
		if args.plot:
			plot_dataset(train_features, train_labels, test_features, test_labels)
			
    # play a sound so i know it's done and can start a new one
		if args.sound:
		  os.system('play -nq -t alsa synth 1 sine 600')

	# use Other to create a model
	elif args.method == 'Other':
		print 'Using Other to solve...'

	# 
	else:
		print 'ERROR: <CNN>, <SVM>, <KNN>, <DTree>, <Other>'


main()
