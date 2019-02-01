import sys, argparse
from scipy.spatial.distance import cdist
import scipy.io as scio
from sklearn import linear_model
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
def extract_data(files):
	first = 1
	
	for dataset in files:
		# load data
		x = scio.loadmat(dataset)

		for key in x.keys():
			if 'week' in key:
				for item in x[key]:
					if first:
						features = np.array([[item[1],item[2]]])
						labels = np.array([item[4]])
						first = 0
					else:
						features = np.vstack((features, [item[1],item[2]]))
						labels = np.append(labels, [item[4]])

	if debug:
		print features.shape
		print features
		print labels.shape
		print labels
	return features, labels

# test_solvers
# INPUTS
#     features      - the features of the training data
#     labels        - the labels of the training data
#     test_features - the testing features
#     test_labels  - the labels of the test dataset to test accuracy of the model
# OUTPUTS
#     max_solver - the solver that produces the highest accuracy
# FUNCTIONALITY
#     This function uses train_model_regression with the different provided solvers
#     from sklearn to determine which provides the highest value for the dataset
def test_solvers(features, labels, test_features, test_labels):
	max_score = 0
	max_solver = ''

	for solver in ['newton-cg', 'lbfgs', 'sag', 'saga']:
		score = train_model_regression(features, labels, test_features, test_labels, solver)
		if score > max_score:
			max_score = score
			max_solver = solver

	print 'The optimal solver is ', max_solver, ' with a mean accuracy of ', max_score

	return max_solver

# test_elbow
# INPUTS
#     features      - the features of the training data
#     test_features - the testing features
#     test_labels  - the labels of the test dataset to test accuracy of the model
# OUTPUTS
#     min_k - the k number of clusters that should be used determined using the elbow method
# FUNCTIONALITY
#     This function uses the train_model_clustering function and the elbow method to 
#     test values fo k between 2 and 15 to find the elbow on the produced chart visually.
#     it then returns the number of clusters that the elbow method demonstrated is ideal for
#     clustering this data
def test_elbow(features, test_features, test_labels):
	prev_score = 0
	predicted_of_prev = []
	min_k = 5 # derived from looking at the elbow plot

	km = np.array([])
	scores = np.array([])
	distortions = np.array([])

	for k in range(2,15):
		score = train_model_clustering(features, test_features, test_labels, k)
		km = np.append(km, [k])
		scores = np.append(scores, [score])

	plt.plot(km, scores, color='black')

	plt.xticks(())
	plt.yticks(())

	plt.show()

	print 'The optimal clusters is ', min_k, ' with a mean accuracy of ', scores[min_k-2]

	return min_k


# train_model_regression
# INPUTS
#     features - the features of the training data
#     labels   - the labels of the training data
#     test     - the values to test the model on
#     test_labels  - the labels of the test dataset to test accuracy of the model
#     solver       - the solver to use with the logistic regression model
# OUTPUTS
#     score - the score of the models accuracy, generated using the test data and labels
# FUNCTIONALITY
#     supervised learning
#     this will use the linear_model from the sklearn package to fit it to the features with
#     he labels and then predict values using the test dataset. a score produced will tell how 
#     accurately the model classified the dataset.
def train_model_regression(features, labels, test, test_labels, solver):
	regr = linear_model.LogisticRegression(solver=solver, multi_class='multinomial')
	fitted = regr.fit(features, labels)
	predicted = fitted.predict(test)
	score = fitted.score(test, test_labels)

	if debug:
		print 'Predicted: ', type(predicted), predicted
		print 'Mean Accuracy for ', solver, ': ', score

	return score

# train_model_clustering
# INPUTS
#     features - the features of the training data
#     test     - the values to test the model on
#     test_labels  - the labels of the test dataset to test accuracy of the model
#     num_clusters - the number of clusters to use in the model
# OUTPUTS
#     score - the score of the models accuracy, generated using the test data and labels
# FUNCTIONALITY
#     unsupervised learning
#     this will use the KMeans clustering from the sklearn package to fit it to the features 
#     and then predict values using the test dataset. a score produced will tell how accurately
#     the model classified the dataset.
def train_model_clustering(features, test, test_labels, num_clusters):
	clus = cluster.KMeans(n_clusters=num_clusters, random_state=0)
	fitted = clus.fit(features)
	predicted = fitted.predict(test)
	score = abs(fitted.score(test, test_labels))
	distortion = sum(np.min(cdist(features, fitted.cluster_centers_, 'euclidean'), axis=1)) / features.shape[0]

	if debug:
		print 'Predicted: ', type(predicted), predicted
		print 'Mean Accuracy for ', num_clusters, ': ', score

	return score

# MAIN function
def main():
	# set up the argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--training', metavar='T', nargs='+', help='the training datasets; .mat file', required=1)
	parser.add_argument('-e', '--testing', nargs=1, help='the testing dataset; .mat file', required=1) 
	parser.add_argument('-p', '--plot', action='store_true', help='plot the datasets')
	parser.add_argument('-m', '--method', help='either <regression> or <clustering> for part 1 and part 2 of the homework, respectively', required=1)
	parser.add_argument('-d', '--debug', action='store_true', help='debug')

	args = parser.parse_args()

	global debug
	debug = args.debug

	# extract the data from the training and testing datasets
	features, labels = extract_data(args.training)
	test_features, test_labels = extract_data(args.testing)

	if debug:
		print 'training: \n', features, '\n', labels
		print 'testing: \n', test_features, '\n', test_labels
	
	# use regression to create a model
	if args.method == 'regression':
		print 'Using logistic regression to solve...'
		# test the different solvers for logistic regression to determine the one 
		# that produces the most accurate results
		solver = test_solvers(features, labels, test_features, test_labels)

		# use the solver that was determined to produce the best results to train the model
		score = train_model_regression(features, labels, test_features, test_labels, solver)

		# plot the training and testing datasets on a 3d plane
		if args.plot:
			plot_dataset(features, labels, test_features, test_labels)
	
	# use clustering to create a model
	elif args.method == 'clustering':
		print 'Using K-means clustering to solve...'
		# use the elbow method to determine what k is the best for this model
		num_clusters = test_elbow(features, test_features, test_labels)
	
		# using the given number of clusters, train the model
		score = train_model_clustering(features, test_features, test_labels, num_clusters)

		# plot the training and testing datasets on a 3d plane
		if args.plot:
			plot_dataset(features, labels, test_features, test_labels)

	# this only supports regression and clustering
	else:
		print 'ERROR: please use either <regression> or <clustering> as your method'


main()
