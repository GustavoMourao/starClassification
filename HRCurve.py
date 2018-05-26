
# Script that plot set of data of stars database.
# Also record data used to train the classification model based on thresholds
# Created by Gustavo Mourao

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from numpy import array
import itertools
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


# ----------------
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

# -----plot boundaries decision
def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# ---- function plot figure
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# ------------------------------- Train and obtain model
def getModel():

	# The H-R diagram only shows the absolute magnitude and the color index.
	# Every other column is discarded, and the ones with null values are dropped.
	df = pd.read_csv('hygdata_v3.csv')[['absmag', 'ci']]
	df.dropna(inplace=True) # drops 1882 rows

	#print '%i total rows' % len(df)
	df.head(3)
	
	# numTrainber of data used to train the model
	numTrain = 110
	numTest = 120
	# ------------------------------- Get values related to SuperGiants
	valuesAbsCol = df.values 	# pandas to numTrainpy data
	
	# --- Call function meshgrid
	xx, yy = make_meshgrid(valuesAbsCol[:,1], valuesAbsCol[:,0])
	
	
	# 1. threshold of Supergiants: Magnitude < -6
	thrSuper = -6
	valuesSuper = {"Abs":[], "CI":[], "index":[]}
	
	for i in range(len(valuesAbsCol[:,0])):
		#print i
		#print valuesAbsCol[i,0]
		if valuesAbsCol[i,0] < thrSuper: 		
			#print 'supergiants founded'
			valuesSuper["Abs"].append(valuesAbsCol[i,0])
			valuesSuper["CI"].append(valuesAbsCol[i,1])
			valuesSuper["index"].append(0)
	# Save SuperGiant data here
	#plt.scatter(valuesSuper["CI"], valuesSuper["Abs"], marker = 'o')
	#plt.show()
	
	# save as numTrainpy format
	np.savetxt('superStarsData.txt', array(valuesSuper.values()), delimiter = '\t', fmt='%1.3f')
	#np.savetxt('superStarsData.txt', valuesSuper["CI"] , delimiter = '\t', fmt='%1.3f')
	# ------------------------------- 
	# ------------------------------- Get values related to Giants
	# 2. threshold of Giants
	thrCIGiants = 1.1
	thrAbsGiantsUp = 3
	thrAbsGiantsDown = -2
	valuesGiants = {"Abs":[], "CI":[], "index":[]}
	
	for i in range(len(valuesAbsCol[:,0])):
		if (valuesAbsCol[i,0] < thrAbsGiantsUp) and (valuesAbsCol[i,0] > thrAbsGiantsDown) and (valuesAbsCol[i,1] > thrCIGiants):
			valuesGiants["Abs"].append(valuesAbsCol[i,0])
			valuesGiants["CI"].append(valuesAbsCol[i,1])
			valuesGiants["index"].append(1)
	# Save Giant data here
	#plt.scatter(valuesGiants["CI"], valuesGiants["Abs"], marker = 'o')
	#plt.show()

	# save as numTrainpy format
	np.savetxt('giantsStarsData.txt', array(valuesGiants.values()), delimiter = '\t', fmt='%1.3f')	
	# -------------------------------
	# ------------------------------- Get values related to dwarfs
	# 3. threshold of dwarfs
	thrCIDwarfs = 0.6
	thrAbsDwarfs = 9
	valuesDwarfs = {"Abs":[], "CI":[], "index":[]}
	
	for i in range(len(valuesAbsCol[:,0])):
		if (valuesAbsCol[i,0] > thrAbsDwarfs) and (valuesAbsCol[i,1] < thrCIDwarfs):
			valuesDwarfs["Abs"].append(valuesAbsCol[i,0])
			valuesDwarfs["CI"].append(valuesAbsCol[i,1])
			valuesDwarfs["index"].append(2)
	#plt.scatter(valuesDwarfs["CI"], valuesDwarfs["Abs"], marker='o')
	#plt.show()
	
	# save as numTrainpy format
	np.savetxt('dwarfsStarsData.txt', array(valuesDwarfs.values()), delimiter = '\t', fmt='%1.3f')
	# -------------------------------
	# ------------------------------- Get values related to main sequence
	valuesMainSeq = {"Abs":[], "CI":[], "index":[]}
	labelMainSeq = {"index":[]}
	thrCIMain1 = 0.5
	thrCIMain2 = 0.6
	thrAbsMain1 = -7
	thrAbsMain2 = 5
	thrAbsMain3 = 5
	
	for i in range(len(valuesAbsCol[:,0])):
		if (valuesAbsCol[i,0] > thrAbsMain2 and valuesAbsCol[i,1] > thrCIMain1) or (valuesAbsCol[i,0] < thrAbsMain3 and valuesAbsCol[i,0] > thrAbsMain1 and valuesAbsCol[i,1] < thrCIMain2):
			valuesMainSeq["Abs"].append(valuesAbsCol[i,0])
			valuesMainSeq["CI"].append(valuesAbsCol[i,1])
			valuesMainSeq["index"].append(3)
	
	# save as numTrainpy format
	np.savetxt('mainStarsData.txt', array(valuesMainSeq.values()),  delimiter = '\t', fmt='%1.3f')	
	# ------------------------------- 


	# --- Test: read dict and organize on numTrainpy format - FIX HERE! - PARAMETER: fmt
	superData = np.loadtxt("superStarsData.txt")
	giantsData = np.loadtxt("giantsStarsData.txt")
	dwarfsData = np.loadtxt("dwarfsStarsData.txt")
	mainData = np.loadtxt("mainStarsData.txt")
	
	# class types
	class_names = ['Super Giants', 'Giants', 'Dwarfs', 'Main']
	
	# Input data
	inputTrainFeat1 = np.block([superData[1,0:numTrain], giantsData[1,0:numTrain], dwarfsData[1,0:numTrain], mainData[1,0:numTrain]])
	inputTrainFeat2 = np.block([superData[2,0:numTrain], giantsData[2,0:numTrain], dwarfsData[2,0:numTrain], mainData[2,0:numTrain]])
	inputTrain = np.vstack((inputTrainFeat1, inputTrainFeat2))
	
	#inputTrainFeat1 = np.block([valuesSuper["Abs"][0:numTrain], valuesGiants["Abs"][0:numTrain], valuesDwarfs["Abs"][0:numTrain], valuesMainSeq["Abs"][0:numTrain]])
	#inputTrainFeat2 = np.block([valuesSuper["CI"][0:numTrain], valuesGiants["CI"][0:numTrain], valuesDwarfs["CI"][0:numTrain], valuesMainSeq["CI"][0:numTrain]])
	#inputTrain = np.hstack((inputTrainFeat1, inputTrainFeat2))
	#inputTrain = np.stack((inputTrainFeat1, inputTrainFeat2), axis=1)
	
	print(inputTrain.shape)
	
	# Output data
	outputTrainFeat1 = np.block([superData[0,0:numTrain], giantsData[0,0:numTrain], dwarfsData[0,0:numTrain], mainData[0,0:numTrain]])
	outputTrainFeat2 = np.block([superData[0,0:numTrain], giantsData[0,0:numTrain], dwarfsData[0,0:numTrain], mainData[0,0:numTrain]])
	#outputTrain = np.hstack((outputTrainFeat1, outputTrainFeat2))
	outputTrain = np.hstack((outputTrainFeat1))
	
	#outputTrainFeat1 = np.block([valuesSuper["index"][0:numTrain], valuesGiants["index"][0:numTrain], valuesDwarfs["index"][0:numTrain], valuesMainSeq["index"][0:numTrain]])
	#outputTrainFeat2 = np.block([valuesSuper["index"][0:numTrain], valuesGiants["index"][0:numTrain], valuesDwarfs["index"][0:numTrain], valuesMainSeq["index"][0:numTrain]])
	#outputTrain = np.hstack((outputTrainFeat1, outputTrainFeat2))
	##outputTrain = np.hstack((outputTrainFeat1))
	
	# Train data
	inputTrain = inputTrain.T
	outputTrain = outputTrain.T

	classifierType = raw_input("Which classifier: (SVM/ANN/KNN):")
	classifierType
	if classifierType == 'SVM':
		# Create a classifier: a support vector classifier
		classifier = svm.SVC(kernel='rbf', gamma=0.8, C=0.1)
		#classifier = svm.SVC(kernel='poly', degree=3, C=0.01)
	elif classifierType == 'KNN':
		# KNeighborsClassifier
		classifier = KNeighborsClassifier(n_neighbors=4, metric = 'euclidean')
	elif classifierType == 'ANN':
		# Neural network
		classifier = MLPClassifier(hidden_layer_sizes=(4,4), max_iter=100, alpha=1e-4,
							solver='sgd', verbose=10, tol=1e-4, random_state=1,
							learning_rate_init=.1)
	else:
		print("Possible options: SVM/ANN/KNN")
		exit()	
	
	print(inputTrain.shape)
	print(outputTrain.shape)
	#print(inputTrain)
	
	classifier.fit(inputTrain, outputTrain)
		
	# Storage model
	if classifierType == 'SVM':
		joblib.dump(classifier, 'SVM.pkl')
	elif classifierType == 'KNN':
		joblib.dump(classifier, 'KNN.pkl')
	elif classifierType == 'ANN':
		joblib.dump(classifier, 'ANN.pkl')	
	else:
		print("Possible options: SVM/ANN/KNN")
		exit()			
	
	
	# Now predict the values
	inputTestFeat1 = np.block([superData[1,numTrain:numTest], giantsData[1,numTrain:numTest], dwarfsData[1,numTrain:numTest], mainData[1,numTrain:numTest]])
	inputTestFeat2 = np.block([superData[2,numTrain:numTest], giantsData[2,numTrain:numTest], dwarfsData[2,numTrain:numTest], mainData[2,numTrain:numTest]])
	inputTest = np.vstack((inputTestFeat1, inputTestFeat2))
	
	#inputTestFeat1 = np.block([valuesSuper["Abs"][numTrain:numTest], valuesGiants["Abs"][numTrain:numTest], valuesDwarfs["Abs"][numTrain:numTest], valuesMainSeq["Abs"][numTrain:numTest]])
	#inputTestFeat2 = np.block([valuesSuper["CI"][numTrain:numTest], valuesGiants["CI"][numTrain:numTest], valuesDwarfs["CI"][numTrain:numTest], valuesMainSeq["CI"][numTrain:numTest]])
	#inputTest = np.hstack((inputTestFeat1, inputTestFeat2))
	
	
	outputTestFeat1 = np.block([superData[0,numTrain:numTest], giantsData[0,numTrain:numTest], dwarfsData[0,numTrain:numTest], mainData[0,numTrain:numTest]])
	outputTestFeat2 = np.block([superData[0,numTrain:numTest], giantsData[0,numTrain:numTest], dwarfsData[0,numTrain:numTest], mainData[0,numTrain:numTest]])
	outputTest = np.hstack((outputTestFeat1))

	#outputTestFeat1 = np.block([valuesSuper["index"][numTrain:numTest], valuesGiants["index"][numTrain:numTest], valuesDwarfs["index"][numTrain:numTest], valuesMainSeq["index"][numTrain:numTest]])
	#outputTestFeat2 = np.block([valuesSuper["index"][numTrain:numTest], valuesGiants["index"][numTrain:numTest], valuesDwarfs["index"][numTrain:numTest], valuesMainSeq["index"][numTrain:numTest]])
	#outputTest = np.hstack((outputTestFeat1, outputTestFeat2))	
	
	expected = outputTest.T
	predicted = classifier.predict(inputTest.T)
	
	print(expected.shape)
	print(predicted.shape)

	print("Classification report for classifier %s:\n%s\n"
		% (classifier, metrics.classification_report(expected, predicted)))
	
	cnf_matrix = metrics.confusion_matrix(expected, predicted)
	
	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
						title='Confusion matrix, without normalization')	

	# To use the trained model:
	# Load model, then classify symbol
	# example: 
	# classifier = joblib.load('KNN.pkl')
	# iin = np.block([10, .8])
	# iin = np.reshape((iin), (-1, 2))
	# print iin
	# prediction = classifier.predict(iin)
	# print prediction
	
	# ---- Call plot data function
	plotHRDiagram(classifier)

# --------------------------------------
def bv2rgb(bv):
	# adaptation from: http://balbuceosastropy.blogspot.com.br/2014/03/construction-of-hertzsprung-russell.html
	
    #t = (5000 / (bv + 1.84783)) + (5000 / (bv + .673913))
    t = 4600 * ((1 / ((0.92 * bv) + 1.7)) +(1 / ((0.92 * bv) + 0.62)))
    x, y = 0, 0
    
    if 1667 <= t <= 4000:
        x = .17991 - (2.66124e8 / t**3) - (234358 / t**2) + (877.696 / t)
    elif 4000 < t:
        x = .24039 - (3.02585e9 / t**3) + (2.10704e6 / t**2) + (222.635 / t)
        
    if 1667 <= t <= 2222:
        y = (-1.1063814 * x**3) - (1.34811020 * x**2) + 2.18555832 * x - .20219683
    elif 2222 < t <= 4000:
        y = (-.9549476 * x**3) - (1.37418593 * x**2) + 2.09137015 * x - .16748867
    elif 4000 < t:
        y = (3.0817580 * x**3) - (5.87338670 * x**2) + 3.75112997 * x - .37001483
        
    X = 0 if y == 0 else x / y
    Z = 0 if y == 0 else (1 - x - y) / y
    
    r, g, b = np.dot([X, 1., Z],
        [[3.2406, -.9689, .0557], [-1.5372, 1.8758, -.204], [-.4986, .0415, 1.057]])
    
    R = np.clip(12.92 * r if (r <= 0.0031308) else 1.4 * (r**2 - .285714), 0, 1)
    G = np.clip(12.92 * g if (g <= 0.0031308) else 1.4 * (g**2 - .285714), 0, 1)
    B = np.clip(12.92 * b if (b <= 0.0031308) else 1.4 * (b**2 - .285714), 0, 1)
    
    return [R, G, B, np.random.ranf()]


def plotHRDiagram(classifier):
	# The H-R diagram only shows the absolute magnitude and the color index.
	# Every other column is discarded, and the ones with null values are dropped.
	df = pd.read_csv('hygdata_v3.csv')[['absmag', 'ci']]
	df.dropna(inplace=True) # drops 1882 rows

	#print '%i total rows' % len(df)
	df.head(3)
	
	# # ------------------------------- Get values related to SuperGiants
	valuesAbsCol = df.values 	# pandas to numTrainpy data
	
	# # --- Call function meshgrid
	xx, yy = make_meshgrid(valuesAbsCol[:,1], valuesAbsCol[:,0])
	
	color = df['ci'].apply(bv2rgb)

	fig = plt.figure(
		figsize=(6, 8),
		facecolor='black',
		dpi=72)
	ax = fig.add_axes([.1, .1, .85, .8])
	
	# BASED ON THE REGIONS, HAS TO BE CHANGED THE THRESHOLDS USED..
	plot_contours(ax, classifier, xx, yy, cmap=plt.cm.coolwarm, s=20, edgecoloers='k')	
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xticks(())
	ax.set_yticks(())
	ax.set_axis_bgcolor('black')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['left'].set_color('white')
	ax.spines['bottom'].set_color('white')

	ax.set_title('Hertzsprung-Russell Diagram', color='white', fontsize=18)
	ax.title.set_position([.5, 1.03])
	ax.set_xlabel('Color Index (B-V)', color='white')
	ax.set_ylabel('Absolute Magnitude', color='white')

	ax.scatter(
		df['ci'],
		df['absmag'],
		marker='.',
		s=[1] * len(df),
		facecolors=color,
		linewidth=0)

	ax.set_xlim(-.5, 2.5)
	ax.set_xticks(np.linspace(0, 2, 3, endpoint=True))
	ax.set_ylim(18, -16)
	ax.set_yticks(np.linspace(20, -10, 3, endpoint=True))
	ax.tick_params(top='off', right='off', direction='out', colors='white')

	ax.annotate(
		'main sequence', xy=(.6, 6.5), xycoords='data',
		fontsize='small', color='white',
		xytext=(-40, -30), textcoords='offset points',
		arrowprops=dict(
			arrowstyle="->",
			connectionstyle="arc3,rad=-.2",
			color='white'))
	ax.annotate(
		'giants', xy=(1.8, -1), xycoords='data',
		fontsize='small', color='white',
		xytext=(30, 7), textcoords='offset points',
		arrowprops=dict(
			arrowstyle="->",
			connectionstyle="arc3,rad=.2",
			color='white'))
	ax.annotate(
		'supergiants', xy=(.5, -14), xycoords='data',
		fontsize='small', color='white')
	ax.annotate(
		'white dwarfs', xy=(0, 16), xycoords='data',
		fontsize='small', color='white');
	
	
	plt.show()
	
	# ---- Call classification function
	#getModel()

def main():
	
	# ---- Call classification function
	getModel()
	
if __name__ == '__main__':
    main()
