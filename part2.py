import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cluster import KMeans # for kmeans clustering
import part2Utils as myUtils

# Training and Testing data and their lablels arrays
trainingData = []
trainingDataLabel = []
testingData = []
testingDataLabel = []

# dct training and testing data
dctReducedTrainingData = []
dctReducedTestingData = []

# PCA training and testing data
pcaTrainingData = []
pcaTestingData = []

# Set printing options to print the entire array (used for debugging and tracing outputs clearly)
np.set_printoptions(threshold=np.inf)

# Loading the data
trainingData, trainingDataLabel, testingData, testingDataLabel = myUtils.loadData(trainingData, trainingDataLabel, testingData, testingDataLabel)

# Applying dct compression
dctReducedTrainingData = myUtils.applyDCTCompression(trainingData, (10000,20,20))
dctReducedTestingData = myUtils.applyDCTCompression(testingData, (2000,20,20))

# Applying PCA with 0.95 variance
pcaTrainingData, pcaTestingData = myUtils.applyPCA(trainingData,testingData, 0.95)

# Applying LDA transformation
ldaTrainingData, ldaTestingData = myUtils.applyLDA(trainingData.reshape((10000,784)), trainingDataLabel, testingData.reshape((2000,784)))

####
####-- K-means clustering scores --###
####
# Number of cluster list
dctNumberOfCluster = [1, 4, 16, 32]

print("\t\t\t-----DCT K-means Test case------\t\t\t")
print("n\ttesting accuracy\ttraining accuracy\tprocessing time(seconds)")
# Looping over number of cluster
for n in dctNumberOfCluster:
    # Apply k-mean clustering on the DCT training data after reshaping to k-mean needed dimension
    testingAccuray, trainingAccuracy, processingTime =myUtils.applyKMeans(dctReducedTrainingData.reshape((10000,400)), trainingDataLabel, dctReducedTestingData.reshape(2000,400), testingDataLabel, n)
    print("{}\t{}\t\t\t{}\t\t\t{}".format(n, testingAccuray, trainingAccuracy, processingTime))

print("\t\t\t-----PCA K-means Test case------\t\t\t")
print("n\ttesting accuracy\ttraining accuracy\tprocessing time(seconds)")
# Looping over number of cluster
for n in dctNumberOfCluster:
    # Apply k-mean clustering on the DCT training data after reshaping to k-mean needed dimension
    testingAccuray, trainingAccuracy, processingTime =myUtils.applyKMeans(pcaTrainingData, trainingDataLabel, pcaTestingData, testingDataLabel, n)
    print("{}\t{}\t\t\t{}\t\t\t{}".format(n, testingAccuray, trainingAccuracy, processingTime))

print("\t\t\t-----LDA K-means Test case------\t\t\t")
print("n\ttesting accuracy\ttraining accuracy\tprocessing time(seconds)")
# Looping over number of cluster
for n in dctNumberOfCluster:
    # Apply k-mean clustering on the DCT training data after reshaping to k-mean needed dimension
    testingAccuray, trainingAccuracy, processingTime =myUtils.applyKMeans(ldaTrainingData, trainingDataLabel, ldaTestingData, testingDataLabel, n)
    print("{}\t{}\t\t\t{}\t\t\t{}".format(n, testingAccuray, trainingAccuracy, processingTime))

####
####-- Linear svm fiting scores --###
####
print("\t\t\t-----DCT linear svm case ------\t\t\t")
print("testing accuracy\ttraining accuracy\tprocessing time(seconds)")
# Apply Svm clustering on the DCT training data after reshaping to k-mean needed dimension
testingAccuray, trainingAccuracy, processingTime =myUtils.applySVM(dctReducedTrainingData.reshape((10000,400)), trainingDataLabel, dctReducedTestingData.reshape(2000,400), testingDataLabel, kernel='linear')
print("{}\t\t\t{}\t\t\t{}".format(testingAccuray, trainingAccuracy, processingTime))

print("\t\t\t-----PCA linear svm case ------\t\t\t")
print("testing accuracy\ttraining accuracy\tprocessing time(seconds)")
# Looping over number of cluster
# Apply k-mean clustering on the DCT training data after reshaping to k-mean needed dimension
testingAccuray, trainingAccuracy, processingTime =myUtils.applySVM(pcaTrainingData, trainingDataLabel, pcaTestingData, testingDataLabel, kernel='linear')
print("{}\t\t\t{}\t\t\t{}".format(testingAccuray, trainingAccuracy, processingTime))

print("\t\t\t-----lda linear svm case ------\t\t\t")
print("testing accuracy\ttraining accuracy\tprocessing time(seconds)")
# Looping over number of cluster
# Apply k-mean clustering on the DCT training data after reshaping to k-mean needed dimension
testingAccuray, trainingAccuracy, processingTime =myUtils.applySVM(ldaTrainingData, trainingDataLabel, ldaTestingData, testingDataLabel, kernel='linear')
print("{}\t\t\t{}\t\t\t{}".format(testingAccuray, trainingAccuracy, processingTime))

####
####-- Non-Linear Radial Basis Function (RBF) svm fiting scores --###
####
print("\t\t\t-----DCT RBF kernel (non-linear) svm case ------\t\t\t")
print("testing accuracy\ttraining accuracy\tprocessing time(seconds)")
# Apply Svm clustering on the DCT training data after reshaping to k-mean needed dimension
testingAccuray, trainingAccuracy, processingTime =myUtils.applySVM(dctReducedTrainingData.reshape((10000,400)), trainingDataLabel, dctReducedTestingData.reshape(2000,400), testingDataLabel, kernel='rbf')
print("{}\t\t\t{}\t\t\t{}".format(testingAccuray, trainingAccuracy, processingTime))

print("\t\t\t-----PCA RBF kernel (non-linear) svm case ------\t\t\t")
print("testing accuracy\ttraining accuracy\tprocessing time(seconds)")
# Looping over number of cluster
# Apply k-mean clustering on the DCT training data after reshaping to k-mean needed dimension
testingAccuray, trainingAccuracy, processingTime =myUtils.applySVM(pcaTrainingData, trainingDataLabel, pcaTestingData, testingDataLabel, kernel='rbf')
print("{}\t\t\t{}\t\t\t{}".format(testingAccuray, trainingAccuracy, processingTime))

print("\t\t\t-----lda RBF kernel (non-linear) svm case ------\t\t\t")
print("testing accuracy\ttraining accuracy\tprocessing time(seconds)")
# Looping over number of cluster
# Apply k-mean clustering on the DCT training data after reshaping to k-mean needed dimension
testingAccuray, trainingAccuracy, processingTime =myUtils.applySVM(ldaTrainingData, trainingDataLabel, ldaTestingData, testingDataLabel, kernel='rbf')
print("{}\t\t\t{}\t\t\t{}".format(testingAccuray, trainingAccuracy, processingTime))

# Plotting Confusion matricses fot the best results of each classifier
km = KMeans(n_clusters=32, n_init=5, max_iter=10000, algorithm='lloyd', random_state=0)
km.fit(ldaTrainingData)
predicted = km.predict(ldaTestingData)
y_hat = myUtils.accCalc(testingDataLabel, predicted, 0)
myUtils.confusionMatrix(testingDataLabel, y_hat)

svm_model = SVC(kernel='rbf')
svm_model.fit(pcaTrainingData, trainingDataLabel)
predicted_svm= svm_model.predict(pcaTestingData)
myUtils.confusionMatrix(testingDataLabel,predicted_svm)
