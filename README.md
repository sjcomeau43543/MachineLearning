# MachineLearning
Supervised and Unsupervised Machine Learning

## Supervised Learning

Using logistic regression and classification to determine the users location category based on history. The first 4 weeks of data is used as training for the model and week5 is the testing week. week6 is also used as testing for the model.

### Usage
`python27 modelling.py -t data/week1.mat data/week2.mat data/week3.mat data/week4.mat -e data/week5.mat -m regression` 

### Results
Using newton-cg we got a mean accuracy of 97% for the test data.

## Unsupervised Learning
 
Using KMeans clustering and the elbow method to determine the users location based on history. This uses the first 5 weeks of data as training for the model. A 6th week was used to predict the locations.

### Usage
`python27 modelling.py -t data/week1.mat data/week2.mat data/week3.mat data/week4.mat data/week5.mat -e data/week5.mat -m clustering` 

### Results
Using the elbow method we found that 2 categories was the ideal number of clusters.

## Data
The data collected by Google Maps over 5 weeks. The data is taken every half an hour.
These files are located in the data folder and are stored as `.mat` files.

First Column: Timestamp
Second Column: Latitude
Third Column: Longitude
Forth Column: Accuracy
Fifth Column: Label
