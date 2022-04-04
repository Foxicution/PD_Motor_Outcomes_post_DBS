#DBS effect om motor output
#Paulius Lapienis Dec 2021


# import docx
from __future__ import division
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, auc, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, OneClassSVM
from statistics import stdev
from random import choice
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from numpy import interp


tf.get_logger().setLevel('ERROR')


#--------------------------------------
#Data
#--------------------------------------

df = pd.read_csv("MRMR_2.csv", index_col=0)
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.15)
base_fpr = np.linspace(0, 1, 101)

#--------------------------------------
#Errors
#--------------------------------------

def predictionR(classifier, X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    return y_test, y_pred, pipe

def evaluationR(y, y_hat, title = 'Confusion Matrix'):
    cm = confusion_matrix(y, y_hat, labels=[1.0, 2.0])
    sensitivity = cm[0,0]/(cm[0,0] + cm[0,1])
    specificity = cm[1,1]/(cm[1,1] + cm[1,0])
    accuracy = accuracy_score(y, y_hat)
    fpr, tpr, thresholds = roc_curve(y, y_hat, pos_label=2)
    AUC = auc(fpr, tpr)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    return accuracy, sensitivity, specificity, AUC, tpr

def PRES(k, res_acc, res_sens, res_spec, res_AUC, tprs):
    print("%1d %4.2f  ±%4.2f    %4.2f ±%4.2f   %4.2f ±%4.2f   %4.2f ±%4.2f" % (k, 100*sum(res_acc)/len(res_acc), 100*stdev(res_acc), 100*sum(res_sens)/len(res_sens), 100*stdev(res_sens),
          100*sum(res_spec)/len(res_spec), 100*stdev(res_spec), 100*sum(res_AUC)/len(res_AUC), 100*stdev(res_AUC)))
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.text(x = 0.5, y = 0.2, s="AUC = %4.4f" % (sum(res_AUC)/len(res_AUC)))
    plt.show()

scaler = StandardScaler()

#--------------------------------------
#Deep Neural Network
#--------------------------------------

def DNN(X_train, X_test, y_train, y_test, scaler, class_ratio):
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    lb = LabelBinarizer().fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    
    #balancing the classes
    class_Bad_DBS=float(np.sum(y_train==0))
    class_Good_DBS=float(np.sum(y_train==1))
    class_ratio=class_Good_DBS/class_Bad_DBS

    class_weights = {0:class_ratio, 1:1}

    model = Sequential()
    model.add(Dense(11, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(7, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, class_weight=class_weights, batch_size=5, verbose=0)
    predictions = model.predict(X_test)
    predictions = lb.inverse_transform(predictions)
    y_test = lb.inverse_transform(y_test)
    return y_test, predictions


#--------------------------------------
#SVM
#--------------------------------------

def predSVM(classifier, X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(X_train)
    y_pred = pipe.predict(X_test)
    y_pred[y_pred==1] = 2.0
    y_pred[y_pred==-1] = 1.0
    return y_test, y_pred, pipe


#--------------------------------------
#Autoencoder
#--------------------------------------

def Autoencoder(scaler, df):
    df1 = df[df['Efektas DBS (1-blogas, 2-geras, 3-labai geras)'] == 1]
    df2 = df[df['Efektas DBS (1-blogas, 2-geras, 3-labai geras)'] == 2]
    #bad DBS effect
    X1 = df1.iloc[:, 0:-1]
    y1 = df1.iloc[:, -1]
    #good and very good DBS effect
    X2 = df2.iloc[:, 0:-1]
    y2 = df2.iloc[:, -1]

    #good and very good DBS effect: splitting for trauining and testing
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.15)
    #scaling
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
   
    #bad DBS
    X1 = scaler.transform(X1)

    #Autoencoder
    encoding_dim = 4 
    model = Sequential()
    model.add(Dense(18, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(encoding_dim, activation='relu'))
    model.add(Dropout(0.1))
    model.add((Dense(18, activation='relu')))
    model.add(Dropout(0.1))
    model.add((Dense(X_train.shape[1], activation='linear')))
    model.compile(loss='msle', optimizer='adam', metrics=['mse'])
    history = model.fit(X_train, X_train, epochs=100, batch_size=5,
                        validation_data=(X_train, X_train), verbose=0, shuffle=True)
    #Possible other methods for anomly detection max mae for example
    
    #good and very good DBS: prediction and errors
    #training
    X_train_pred = model.predict(X_train)
    errors = np.sum(np.square(X_train_pred - X_train)/X_train.shape[1], axis=1)
    threshold = np.mean(errors) + stdev(errors)
    #testing
    X_test_pred = model.predict(X_test)
    errors = np.sum(np.square(X_test_pred - X_test)/X_train.shape[1], axis=1)
    y_pred = [1.0 if err > threshold else 2.0 for err in errors]
    
    #bad DBS: prediction of outliers
    X1_pred = model.predict(X1)
    errors = np.sum(np.square(X1_pred - X1)/X_train.shape[1], axis=1)
    y1_pred = [1.0 if err > threshold else 2.0 for err in errors] #bloga klase = 1, gera ir labai gera = 2
    
    #all predictions
    t_vals = np.concatenate([y_test.values, y1.values])
    p_vals = np.concatenate([y_pred, y1_pred])

    return t_vals, p_vals

#----------------------------------------------------------------------------------------
#Classificatin Methods
#----------------------------------------------------------------------------------------


def REP(X, y, N = 10, scaler = scaler, df=df):

    res_acc = [[], [], [], [], [], [], [], [], [], []]
    res_sens = [[], [], [], [], [], [], [], [], [], []]
    res_spec = [[], [], [], [], [], [], [], [], [], []]
    res_AUC = [[], [], [], [], [], [], [], [], [], []]
    tprs = [[], [], [], [], [], [], [], [], [], []]
    
    for i in range(N):
        print("Run %d" %(i))

        #split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)

        #scaling  
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
        #balancing the classes        
        class_Bad_DBS=float(np.sum(y_train==1))
        class_Good_DBS=float(np.sum(y_train==2))
        class_ratio=class_Good_DBS/class_Bad_DBS

        #Classes: 1 - bad DBS, 2 - good and very good DBS  
        class_weights = {1:class_ratio, 2:1}


        #----------------------------------------------
        #Classification Methods
        #----------------------------------------------
        #LogReg
        logreg = LogisticRegression(class_weight=class_weights)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[0].append(acc)
        res_sens[0].append(sens)
        res_spec[0].append(spec)
        res_AUC[0].append(AUC)
        tprs[0].append(tpr)

        #DecTree
        y_test, y_pred, model = predictionR(DecisionTreeClassifier(class_weight=class_weights), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[1].append(acc)
        res_sens[1].append(sens)
        res_spec[1].append(spec)
        res_AUC[1].append(AUC)
        tprs[1].append(tpr)
        
        #Random forrest
        y_test, y_pred, model = predictionR(RandomForestClassifier(), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[9].append(acc)
        res_sens[9].append(sens)
        res_spec[9].append(spec)
        res_AUC[9].append(AUC)
        tprs[9].append(tpr)

        #Linear Discriminant 
        y_test, y_pred, model = predictionR(LinearDiscriminantAnalysis(), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[2].append(acc)
        res_sens[2].append(sens)
        res_spec[2].append(spec)
        res_AUC[2].append(AUC)
        tprs[2].append(tpr)

        #Naive Bayes
        y_test, y_pred, model = predictionR(GaussianNB(), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[3].append(acc)
        res_sens[3].append(sens)
        res_spec[3].append(spec)
        res_AUC[3].append(AUC)
        tprs[3].append(tpr)
        
        #SVM
        y_test, y_pred, model = predictionR(SVC(class_weight=class_weights), X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[4].append(acc)
        res_sens[4].append(sens)
        res_spec[4].append(spec)
        res_AUC[4].append(AUC)
        tprs[4].append(tpr)

        ##RandomC##
        y_pred = []
        for el in y_test:
            y_pred.append(choice([1.0, 2.0]))
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[5].append(acc)
        res_sens[5].append(sens)
        res_spec[5].append(spec)
        res_AUC[5].append(AUC)
        tprs[5].append(tpr)

        #DNN
        y_test, y_pred = DNN(X_train, X_test, y_train, y_test, scaler, class_ratio)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[6].append(acc)
        res_sens[6].append(sens)
        res_spec[6].append(spec)
        res_AUC[6].append(AUC)
        tprs[6].append(tpr)
        
        ##Anomaly detection##

        #One class SVM
        y_test, y_pred, model = predSVM(OneClassSVM(nu=4/29, gamma=0.005), #split 0.1, 4 bad + 25 good
                                        X_train, X_test, y_train, y_test)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[7].append(acc)
        res_sens[7].append(sens)
        res_spec[7].append(spec)
        res_AUC[7].append(AUC)
        tprs[7].append(tpr)
     
        #Autoencoder
        y_test, y_pred = Autoencoder(scaler, df)
        acc, sens, spec, AUC, tpr = evaluationR(y_test, y_pred)
        res_acc[8].append(acc)
        res_sens[8].append(sens)
        res_spec[8].append(spec)
        res_AUC[8].append(AUC)
        tprs[8].append(tpr)
        
        
    print("Accuracy %  Sensitivity % Specificity % ")

    #LogReg
    print("\nLogistic Regression")
    PRES(1,res_acc[0], res_sens[0], res_spec[0], res_AUC[0], tprs[0])
    #DecTree
    print("\nDecision Tree Classifier")
    PRES(2, res_acc[1], res_sens[1], res_spec[1], res_AUC[1], tprs[1])
    #Linear Discriminat 
    print("\nLinear Discriminant Analysis")
    PRES(3, res_acc[2], res_sens[2], res_spec[2], res_AUC[2], tprs[2])
    #Naive Bayes
    print("\nGaussian Bayes")
    PRES(4, res_acc[3], res_sens[3], res_spec[3], res_AUC[3], tprs[3])
    #SVM
    print("\nSVM")
    PRES(5, res_acc[4], res_sens[4], res_spec[4], res_AUC[4], tprs[4])
    ##RC##
    print("\nRandom Choice")
    PRES(6, res_acc[5], res_sens[5], res_spec[5], res_AUC[5], tprs[5])
    #NN
    print("\nDeep Neural Network")
    PRES(7, res_acc[6], res_sens[6], res_spec[6], res_AUC[6], tprs[6])
    ##Anomaly##
    #One Class SVM
    print("\nOne class SVM")
    PRES(8, res_acc[7], res_sens[7], res_spec[7], res_AUC[7], tprs[7])
    #Autoencoder
    print("\nAutoencoder")
    PRES(9, res_acc[8], res_sens[8], res_spec[8], res_AUC[8], tprs[8])
    print("\nRandom Forrest")
    PRES(9, res_acc[9], res_sens[9], res_spec[9], res_AUC[9], tprs[9])
    
    

N_runs=3

REP(X, y, N_runs)


