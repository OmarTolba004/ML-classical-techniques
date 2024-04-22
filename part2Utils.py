import os  # Used in listing files in directories
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from scipy.fftpack import dct #for DCT implementation
from sklearn.decomposition import PCA #for PCA implementation
from sklearn.cluster import KMeans # for kmeans clustering
from scipy.stats import mode # used here to determine the most frequent class label within each cluster,
from sklearn.metrics import accuracy_score # function for measuring accuracy
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


# Training in Testind Data paths
trainingDataPath = r'/home/omar/college/NN/assignments/ReducedMNIST/Reduced MNIST Data/Reduced Trainging data'
testingDataPath = r'/home/omar/college/NN/assignments/ReducedMNIST/Reduced MNIST Data/Reduced Testing data'

def loadData(trainingData, trainingDataLabel, testingData, testingDataLabel):
    # Looping over all letter in the Training Directory 
    for digital_letter in os.listdir(trainingDataPath):
        # looping over each image in the Training Data path Directory 
        for image in os.listdir(trainingDataPath+'/'+digital_letter):
            # adding training data to data list and converting the RGB to Grayscale
            trainingData.append(cv2.imread(trainingDataPath+'/'+digital_letter+ '/'+image, cv2.IMREAD_GRAYSCALE))
            # adding data laberl to data label list (in the sampe loop to result in equal mapping)
            trainingDataLabel.append(digital_letter)


    # Looping over all letter in the Training Directory 
    for digital_letter in os.listdir(testingDataPath):
        # looping over each image in the Training Data path Directory 
        for image in os.listdir(testingDataPath+'/'+digital_letter):
            # adding training data to data list converting the RGB to Grayscale
            testingData.append(cv2.imread(testingDataPath+'/'+digital_letter+ '/'+image, cv2.IMREAD_GRAYSCALE))
            # adding data laberl to data label list (in the sampe loop to result in equal mapping)
            testingDataLabel.append(digital_letter)

    # Converting data and test training image to nparray and reshaping it MNIST data dimension (28 x 28)
    trainingData = np.asarray(trainingData).reshape((10000,28,28))
    testingData = np.asarray(testingData).reshape((2000,28,28))
    # Converting lables to numpy array 
    trainingDataLabel = np.asarray(trainingDataLabel).astype(int)
    testingDataLabel = np.asarray(testingDataLabel).astype(int)

    return trainingData, trainingDataLabel, testingData, testingDataLabel

def applyDCTCompression(data, finalShape) -> None:
    
    #Reduced dct data
    dctDataReduced = np.empty(finalShape)
    
    '''
    Preforming DCT on the data upon all rows and columns 
    dct(data,axis=1) apply dct over the rows then apply dct on columns over the returned dct 
    ''' 
    dctData = dct(dct(data,axis=1), axis=2)
    
    # Looping over data and reduce their dimensions
    for d in range(data.shape[0]):
        # Empty list that holds the kept DCT coefficients
        keptDCTCof = []

        # Looping over rows and columns
        for i in range(finalShape[1]):
            for j in range(finalShape[2]):
                # Keeping only the upper left wanted rows and columns(20x20 in our case)subimage
                keptDCTCof.append(dctData[d,i,j])
        # adding the temp array to the reduced DCT array and resahpe them to the finale shape
        dctDataReduced[d] = np.asarray(keptDCTCof).reshape(finalShape[1],finalShape[2])

    # Returning the dct reduced data 
    return dctDataReduced 

def applyPCA(trainingData, testingData, pcaVar):
    # Creating PCA object with the received variance
    pca = PCA(pcaVar)
    # Fit PCA on training data
    pca.fit(trainingData.reshape((trainingData.shape[0], trainingData.shape[1]*trainingData.shape[2])))
    # Apply PCA to data (Fit the model with data and apply the dimensionality reduction on data.)
    pcaTrainingData = pca.transform(trainingData.reshape((trainingData.shape[0], trainingData.shape[1]*trainingData.shape[2])))
    pcaTestingData = pca.transform(testingData.reshape((testingData.shape[0], testingData.shape[1]*testingData.shape[2])))
    
    #Returning the pca data
    return pcaTrainingData, pcaTestingData

def applyLDA(trainingData, trainingDataLabel, testingData, nComponents=None):
    # Creating lda object
    lda = LinearDiscriminantAnalysis(n_components=nComponents)
    # Fitting the LDA model
    lda.fit(trainingData, trainingDataLabel)
    # transofrming learned and testing data using the learned transofrmation
    ldaTrainingData = lda.transform(trainingData)
    ldaTestingData = lda.transform(testingData)
    
    return ldaTrainingData, ldaTestingData


def applyKMeans(trainingData, trainingDataLabel, testingData, testingDataLabel, numberOfClusters):
    
    # start time needed for caluclating execution time
    startTime = time.time()
    # Preform K-means clustering on training data
    # n : The number of clusters to form as well as the number of centroids to generate.
    # random state: Determines random number generation for centroid initialization
    km = KMeans(n_clusters=numberOfClusters, random_state=0).fit(trainingData)
    
    #calculating processing time
    processingTime = time.time() - startTime
    
    # Getting the cluster labeling
    virtualClasses = km.labels_
    labels = np.zeros_like(virtualClasses)
    # mapping from k-means clustering lables to true lables
    # Each row of mapping will store the most frequent class label within the corresponding cluster.
    mapping = np.zeros((numberOfClusters))
    for i in range(numberOfClusters):
        mask = (virtualClasses == i)
        # mode function : Return an array of the modal (most common) value in the passed array.
        # boolean array indexing.
        '''
        for example assume i =14 in number of cluster case =16
        we getting all indices where i =14 and assign them to true in the mask list
        then, we see what theses indices hold for correct labeling by calculating the most common correct label for them
        then, we assign these correct lablel for all indices
        moreover we assign the mapping from i to the correct lable on mapping list
        '''
        labels[mask] = mode(trainingDataLabel[mask])[0]
        mapping[i]=mode(trainingDataLabel[mask])[0]
    
    # Predict the cluster using testing data
    # Prediction will hold clustering lable that will need to be mapped to the correct lableing
    prediction = km.predict(testingData)

    # Loop over all predicition and map them to the correct label
    for i in range(len(prediction)):
        prediction[i] = mapping[prediction[i]]
    
    # Calculate the testing and training accruacies
    testingAccuray = accuracy_score(testingDataLabel, prediction)
    trainingAccuracy = accuracy_score(trainingDataLabel, labels)
    
    return testingAccuray, trainingAccuracy, processingTime 
    


def applySVM(trainingData, trainingDataLabel, testingData, testingDataLabel, kernel = 'linear'):
    # Creating instance of SVC 
    svm = SVC(kernel=kernel)
    # getting start time which is need for calculating execution time
    startTime = time.time()
    # Fitting the svm model over the data and true labels
    svm.fit(trainingData, trainingDataLabel)
    # Calculating svm prcoessing time
    processingTime = time.time() - startTime 
    # predict the lablel of training data
    labels = svm.predict(trainingData)
    # predict the labels of the test data
    prediction = svm.predict(testingData)
    # calculating accuracies of training data and testing data
    testingAccuray = accuracy_score(testingDataLabel, prediction)
    trainingAccuracy = accuracy_score(trainingDataLabel, labels)

    return testingAccuray, trainingAccuracy, processingTime 


def confusionMatrix(actualLabel, predictedLabel):
    # Creating a Pandas DataFrame with two columns
    df = pd.DataFrame({'Labels': actualLabel, 'predictions': predictedLabel})
    # Computing a cross-tabulation of two (or more) factors. It calculates how many times each combination of labels and predictions occurs.
    ct = pd.crosstab(df['Labels'], df['predictions'])
    sns.heatmap(ct, annot=True, cmap='Reds', fmt='g')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.title('Confusion Matrix')
    plt.show()

#calculate the accuracies
def accCalc(y, y_hat ,c):
    y_cluster = np.zeros(y.shape)
    y_unique = np.unique(y)
    y_unique_ord = np.arange(y_unique.shape[0])
    
    for ind in range(y_unique.shape[0]):
        y[y==y_unique[ind]] = y_unique_ord[ind]

    y_unique = np.unique(y)
    bins = np.concatenate((y_unique, [np.max(y_unique)+1]), axis=0)

    for cluster in np.unique(y_hat):
        hist, _ = np.histogram(y[y_hat==cluster], bins=bins)
        correct = np.argmax(hist)
        y_cluster[y_hat==cluster] = correct
    if(c):
        return accuracy_score(y, y_cluster)
    else:
        return y_cluster
