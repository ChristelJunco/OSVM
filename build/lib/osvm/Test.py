#!/usr/bin/env python
#-*- coding:utf-8 -*-
import csv
import time
import sip
sip.setapi('QString', 2)
sip.setapi('QVariant', 2)

from PyQt4 import QtGui, QtCore

import numpy as np
from numpy import linalg
import pandas as pd
#import pickle

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from TD_SVM import TRAD_SVM
from OPTIMIZE_SVM import IS_BFFA,OPT_SVM

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class MyWindow(QtGui.QWidget):


    def __init__(self):
        super(MyWindow, self).__init__()
        self.colors = {1:'r',-1:'b'}
        self.setGeometry(70,40,1220,700)
        self.setFixedSize(self.size())
        self.setWindowTitle("DETECTION OF ONLINE IMPERSONATION THROUGH OPTIMIZED SVM")
        self.setWindowIcon(QtGui.QIcon('window.png'))

        self.tabWidget = QtGui.QTabWidget(self)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 1200, 690))
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))

        self.model1 = QtGui.QStandardItemModel(self)
        self.model3 = QtGui.QStandardItemModel(self)
        self.model5 = QtGui.QStandardItemModel(self)
        self.model6 = QtGui.QStandardItemModel(self)
        self.model7 = QtGui.QStandardItemModel(self)
        self.model8 = QtGui.QStandardItemModel(self)
        
                

        #TAB FOR DATA EXTRACTION-----------
        self.tab_1 = QtGui.QWidget()
        self.tab_1.setObjectName(_fromUtf8("tab_1"))

        self.pushButtonLoad = QtGui.QPushButton(self.tab_1)
        self.pushButtonLoad.setText("IMPORT CSV FILE")
        self.pushButtonLoad.clicked.connect(self.loadCsv)

        self.tableView1 = QtGui.QTableView(self.tab_1)
        self.tableView1.setModel(self.model1)
        self.tableView1.horizontalHeader().setStretchLastSection(True)
        self.tableView1.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)

        self.layoutVertical1 = QtGui.QVBoxLayout(self.tab_1)
        self.layoutVertical1.addWidget(self.pushButtonLoad)
        self.layoutVertical1.addWidget(self.tableView1)

        self.tabWidget.addTab(self.tab_1,"DATA EXTRACTION")
        #END TAB FOR DATA EXTRACTION--------


        #TAB FOR TRADITIONAL SVM
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))

        self.pushButtonTrain = QtGui.QPushButton(self.tab_2)
        self.pushButtonTrain.setText("TRAIN THE DATA")
        # self.pushButtonTrain.setGeometry(QtCore.QRect(370, 9, 131, 23))
        self.pushButtonTrain.clicked.connect(self.train_TRADSVM)

        self.tableWidget2 = QtGui.QTableWidget()
        self.tableWidget2.setRowCount(1)
        self.tableWidget2.setColumnCount(2)
        header2 = self.tableWidget2.horizontalHeader()
        header2.setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        header2.setResizeMode(1, QtGui.QHeaderView.Stretch)
        self.tableWidget2.verticalHeader().setVisible(False)
        self.tableWidget2.horizontalHeader().setVisible(False)
        self.tableWidget2.setItem(0,0, QtGui.QTableWidgetItem("SPEED OF TRAINING:"))

        self.layoutVertical2 = QtGui.QVBoxLayout(self.tab_2)

        self.figure1 = plt.figure(1)
        self.canvas = FigureCanvas(self.figure1)
        
        self.layoutVertical2.addWidget(self.pushButtonTrain)
        self.layoutVertical2.addWidget(self.canvas)
        self.layoutVertical2.addWidget(self.tableWidget2)
              
        self.tabWidget.addTab(self.tab_2, "TRADITIONAL SVM")
        #END TAB FOR TRADITIONAL SVM
        

        #TAB FOR INSTANCE SELECTION-------
        self.tab_3 = QtGui.QWidget()
        self.tab_3.setObjectName(_fromUtf8("tab_3"))

        self.pushButtonInstance = QtGui.QPushButton(self.tab_3)
        self.pushButtonInstance.setText("SELECTION OF INSTANCE")     
        self.pushButtonInstance.clicked.connect(self.instanceSelect)

        self.tableView3 = QtGui.QTableView(self.tab_3)
        self.tableView3.setModel(self.model3)
        self.tableView3.horizontalHeader().setStretchLastSection(True)
        self.tableView3.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        
        self.layoutVertical3 = QtGui.QVBoxLayout(self.tab_3)
        self.layoutVertical3.addWidget(self.pushButtonInstance)
        self.layoutVertical3.addWidget(self.tableView3)

        self.tabWidget.addTab(self.tab_3, "INSTANCE SELECTION")
        #END TAB FOR INSTANCE SELECTION------
        

        #TAB FOR OPTIMIZE SVM-------
        self.tab_4 = QtGui.QWidget()
        self.tab_4.setObjectName(_fromUtf8("tab_4"))

        self.pushButtonTrain_OS = QtGui.QPushButton(self.tab_4)
        self.pushButtonTrain_OS.setText("TRAINED DATA")
        self.pushButtonTrain_OS.setEnabled(False)
        #self.pushButtonTrain_OS.clicked.connect(self.train_TRADSVM)

        self.tableWidget4 = QtGui.QTableWidget()
        self.tableWidget4.setRowCount(1)
        self.tableWidget4.setColumnCount(2)
        header4 = self.tableWidget4.horizontalHeader()
        header4.setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        header4.setResizeMode(1, QtGui.QHeaderView.Stretch)
        self.tableWidget4.verticalHeader().setVisible(False)
        self.tableWidget4.horizontalHeader().setVisible(False)
        self.tableWidget4.setItem(0,0, QtGui.QTableWidgetItem("SPEED OF TRAINING:"))

        self.figure2 = plt.figure(2)
        self.canvas_OPTSVM = FigureCanvas(self.figure2)

        self.layoutVertical4 = QtGui.QVBoxLayout(self.tab_4)
        self.layoutVertical4.addWidget(self.pushButtonTrain_OS)
        self.layoutVertical4.addWidget(self.canvas_OPTSVM)
        self.layoutVertical4.addWidget(self.tableWidget4)
              
        self.tabWidget.addTab(self.tab_4, "OPTIMIZE SVM")
        #END TAB FOR OPTIMIZE SVM------


        #TAB FOR USER PROFILING-------
        self.tab_5 = QtGui.QWidget()
        self.tab_5.setObjectName(_fromUtf8("tab_5"))

        self.pushButtonGenProf = QtGui.QPushButton(self.tab_5)
        self.pushButtonGenProf.setText("LOAD SVM MODEL")
        self.pushButtonGenProf.clicked.connect(self.loadModel_TSVM)

        self.tableView5  = QtGui.QTableView(self.tab_5)
        self.tableView5.setModel(self.model5)
        self.tableView5.horizontalHeader().setStretchLastSection(True)
        self.tableView5.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        
        self.layoutVertical5 = QtGui.QVBoxLayout(self.tab_5)
        self.layoutVertical5.addWidget(self.pushButtonGenProf)

        self.canvas_Profile = FigureCanvas(self.figure1)
        self.layoutVertical5.addWidget(self.canvas_Profile)
        self.layoutVertical5.addWidget(self.tableView5)

        self.tabWidget.addTab(self.tab_5, "USER PROFILING (TSVM)")
        #END TAB FOR USER PROFILING-------

        #TAB FOR USER PROFILING-------
        self.tab_6 = QtGui.QWidget()
        self.tab_6.setObjectName(_fromUtf8("tab_6"))

        self.pushButtonGenProf_OSVM = QtGui.QPushButton(self.tab_6)
        self.pushButtonGenProf_OSVM.setText("LOAD SVM MODEL")
        self.pushButtonGenProf_OSVM.clicked.connect(self.loadModel_OSVM)

        self.tableView6  = QtGui.QTableView(self.tab_6)
        self.tableView6.setModel(self.model6)
        self.tableView6.horizontalHeader().setStretchLastSection(True)
        self.tableView6.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        
        self.layoutVertical6 = QtGui.QVBoxLayout(self.tab_6)
        self.layoutVertical6.addWidget(self.pushButtonGenProf_OSVM)
        
        self.canvas_Profile_OSVM = FigureCanvas(self.figure2)
        self.layoutVertical6.addWidget(self.canvas_Profile_OSVM)
        self.layoutVertical6.addWidget(self.tableView6)

        self.tabWidget.addTab(self.tab_6, "USER PROFILING (OSVM)")
        #END TAB FOR USER PROFILING-------


        #TAB FOR IMPERSONATION DETECTION-------
        self.tab_7 = QtGui.QWidget()
        self.tab_7.setObjectName(_fromUtf8("tab_7"))

        self.pushButtonImportNew = QtGui.QPushButton(self.tab_7)
        self.pushButtonImportNew.setText("IMPORT NEW DATASET AND TEST")
        self.pushButtonImportNew.clicked.connect(self.loadTestCsv)

        self.tableWidget7 = QtGui.QTableWidget()
        self.tableWidget7.setRowCount(2)
        self.tableWidget7.setColumnCount(2)
        header7 = self.tableWidget7.horizontalHeader()
        header7.setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        header7.setResizeMode(1, QtGui.QHeaderView.Stretch)
        self.tableWidget7.verticalHeader().setVisible(False)
        self.tableWidget7.horizontalHeader().setVisible(False)
        self.tableWidget7.setItem(0,0, QtGui.QTableWidgetItem("ACCURACY SCORE OF TESTING:"))
        self.tableWidget7.setItem(1,0, QtGui.QTableWidgetItem("PREDICTED LABEL:"))        

        self.layoutVertical7 = QtGui.QVBoxLayout(self.tab_7)
        #self.layoutVertical7.addWidget(self.tableView6)
        self.layoutVertical7.addWidget(self.pushButtonImportNew)
        
        self.canvas_pred = FigureCanvas(self.figure1)
        self.layoutVertical7.addWidget(self.canvas_pred)
        self.layoutVertical7.addWidget(self.tableWidget7)       

        self.tabWidget.addTab(self.tab_7, "IMPERSONATION DETECTION (TSVM)")
        #END TAB FOR IMPERSONATION DETECTION-------

        #TAB FOR IMPERSONATION DETECTION OSVM-------
        self.tab_8 = QtGui.QWidget()
        self.tab_8.setObjectName(_fromUtf8("tab_8"))

        self.pushButtonImportNew_OSVM = QtGui.QPushButton(self.tab_8)
        self.pushButtonImportNew_OSVM.setText("IMPORT NEW DATASET AND TEST")
        self.pushButtonImportNew_OSVM.clicked.connect(self.loadTestCsv_OSVM)

        self.tableWidget8 = QtGui.QTableWidget()
        self.tableWidget8.setRowCount(2)
        self.tableWidget8.setColumnCount(2)
        header8 = self.tableWidget8.horizontalHeader()
        header8.setResizeMode(0, QtGui.QHeaderView.ResizeToContents)
        header8.setResizeMode(1, QtGui.QHeaderView.Stretch)
        self.tableWidget8.verticalHeader().setVisible(False)
        self.tableWidget8.horizontalHeader().setVisible(False)
        self.tableWidget8.setItem(0,0, QtGui.QTableWidgetItem("ACCURACY SCORE OF TESTING:"))
        self.tableWidget8.setItem(1,0, QtGui.QTableWidgetItem("PREDICTED LABEL:"))        

        self.layoutVertical8 = QtGui.QVBoxLayout(self.tab_8)
        #self.layoutVertical6.addWidget(self.tableView6)
        self.layoutVertical8.addWidget(self.pushButtonImportNew_OSVM)
        
        self.canvas_pred_OSVM = FigureCanvas(self.figure2)
        self.layoutVertical8.addWidget(self.canvas_pred_OSVM)
        self.layoutVertical8.addWidget(self.tableWidget8)       

        self.tabWidget.addTab(self.tab_8, "IMPERSONATION DETECTION (OSVM)")
        #END TAB FOR IMPERSONATION DETECTION-------

    def loadCsv(self):

        self.model1.clear()

        self.fileName=QtGui.QFileDialog.getOpenFileName() 
        with open(self.fileName, "r") as fileInput:
            for row in csv.reader(fileInput):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model1.appendRow(items)

    def loadTestCsv(self):

        self.model7.clear()       

        self.fileName_new=QtGui.QFileDialog.getOpenFileName() 
        with open(self.fileName_new, "r") as fileInput_new:
            for row in csv.reader(fileInput_new):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model7.appendRow(items)


        self.testData_TSVM()
   
    def loadTestCsv_OSVM(self):

        self.model8.clear()        

        self.fileName_new_OSVM=QtGui.QFileDialog.getOpenFileName() 
        with open(self.fileName_new_OSVM, "r") as fileInput_new_OSVM:
            for row in csv.reader(fileInput_new_OSVM):    
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model8.appendRow(items)

        self.testData_OSVM()
   
    def train_TRADSVM(self):

        #Execution time of Training
        start_time = time.time()

        # import some data to play with
        with open(self.fileName) as fp:
            reader=csv.reader(fp,delimiter=",")
            data=[line for line in reader]
            del data[0]
            datas=np.asfarray(data,float)

            X=datas[:, [0,1]]
                 
            Y=datas[:, [2]]
            Y=np.array(Y).ravel()     

        # we create an instance of SVM and fit out data.
        self.clf = TRAD_SVM(C=1000.1)
        self.clf.fit(X, Y)

        #Print execution time of Training
        self.tableWidget2.setItem(0,1, QtGui.QTableWidgetItem("%s seconds" % (time.time() - start_time)))
       

        #VISUALIZATION
        #3fignum=1
        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(1, figsize=(3, 2))
        plt.clf()

        #plotting the vectors
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
       
        #GET THE NEAREST VECTORS
        plt.scatter(self.clf.sv[:, 0], self.clf.sv[:, 1], s=80, facecolors='none', zorder=10, edgecolors='k')   
                
        #Plot the Labels X and Y
        plt.xlabel("Profile Duration")
        plt.ylabel("Number of Actions")

        #MARGIN OF HYPERPLANE
        plt.axis('tight')
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.mgrid[x_min:x_max:400j, y_min:y_max:400j]

        X = np.array([[xx, yy] for xx, yy in zip(np.ravel(xx), np.ravel(yy))])
        Z = self.clf.project(X).reshape(xx.shape)

        # Put the result into a color plot
        plt.figure(1, figsize=(3, 2))
        plt.pcolormesh(xx, yy, Z>0, cmap=plt.cm.Paired)
        plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        #fignum = fignum + 1
        # #END VISUALIZATION

        self.canvas.draw() #--show training plot

    def instanceSelect(self):

        self.model3.clear()

        with open(self.fileName) as fp:
            reader=csv.reader(fp,delimiter=",")
            data=[line for line in reader]
            del data[0]
            self.fireflies=np.asfarray(data,float) 
              
        self.ffa=IS_BFFA(self.fireflies)

        #GET SELECTED INSTANCE, PUT IN TABLE
        for row in self.ffa.best_training_subset:    
            items = [
                QtGui.QStandardItem(str(self.ffa.best_training_subset))
                for self.ffa.best_training_subset in row
            ]
            self.model3.appendRow(items)

        #TRAIN IN OPTIMIZE SVM
        self.ffa=IS_BFFA(self.fireflies)
        self.datasets=self.ffa.best_training_subset

        #Execution time of Training
        start_time = time.time()

        # import some data to play with
        X=self.datasets[:, [0,1]]
                 
        Y=self.datasets[:, [2]]
        Y=np.array(Y).ravel()     

        # we create an instance of SVM and fit out data.
        self.clf_OSVM = OPT_SVM(C=1000.1)
        self.clf_OSVM.fit(X, Y)

        #Print execution time of Training
        self.tableWidget4.setItem(0,1, QtGui.QTableWidgetItem("%s seconds" % (time.time() - start_time)))
       

        #VISUALIZATION
        #3fignum=1
        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(2, figsize=(3, 2))
        plt.clf()

        #plotting the vectors
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
       
        #GET THE NEAREST VECTORS
        plt.scatter(self.clf_OSVM.sv[:, 0], self.clf_OSVM.sv[:, 1], s=80, facecolors='none', zorder=10, edgecolors='k')   
                
        #Plot the Labels X and Y
        plt.xlabel("Profile Duration")
        plt.ylabel("Number of Actions")

        #MARGIN OF HYPERPLANE
        plt.axis('tight')
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.mgrid[x_min:x_max:400j, y_min:y_max:400j]

        X = np.array([[xx, yy] for xx, yy in zip(np.ravel(xx), np.ravel(yy))])
        Z = self.clf_OSVM.project(X).reshape(xx.shape)

        # Put the result into a color plot
        plt.figure(2, figsize=(3, 2))
        plt.pcolormesh(xx, yy, Z>0, cmap=plt.cm.Paired)
        plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        #fignum = fignum + 1
        # #END VISUALIZATION

        self.canvas_OPTSVM.draw() #--show training plot


    #def train_OPTSVM(self):

       


    def loadModel_TSVM(self):
       
        #Display in Table Nearest Vectors
        support_vecs=np.concatenate((self.clf.sv,self.clf.sv_y[:,None]),axis=1)
        for row in support_vecs:    
            items = [
                QtGui.QStandardItem(str(support_vecs))
                for support_vecs in row
            ]
            self.model5.appendRow(items)

        self.canvas_Profile.draw() #---show Model plot

    def loadModel_OSVM(self):
       
        #Display in Table Nearest Vectors
        support_vecs=np.concatenate((self.clf_OSVM.sv,self.clf_OSVM.sv_y[:,None]),axis=1)
        for row in support_vecs:    
            items = [
                QtGui.QStandardItem(str(support_vecs))
                for support_vecs in row
            ]
            self.model6.appendRow(items)

        self.canvas_Profile_OSVM.draw() #---show Model plot

    def testData_TSVM(self):

        with open(self.fileName_new) as fp:
            reader=csv.reader(fp,delimiter=     ",")
            data=[line for line in reader]
            del data[0]
            datas=np.asfarray(data,float)

            X_test=datas[:, [0,1]]
            
        count_genuine=0
        count_intruder=0

        for p in X_test:   

            Pred=self.clf.predict([p])
           
            Pred=Pred.astype(int)
            
            plt.figure(1, figsize=(3, 2))
            plt.scatter(p[0], p[1], s=200, marker='*', c=self.colors[Pred[0]])

            if Pred[0]==1:
                count_genuine=count_genuine+1
            if Pred[0]==-1:
                count_intruder=count_intruder+1

        if count_genuine>count_intruder:
            detect_label="GENUINE USER"
        if count_intruder>count_genuine:
            detect_label="IMPERSONATOR" 
        if count_genuine==count_intruder:
            detect_label="THE PREDICTED RESULT CANNOT BE LABELED"   

        self.tableWidget7.setItem(1,1, QtGui.QTableWidgetItem(detect_label))
        
        #Get the Accuracy Score in Predict

        #USE SKLEARN.METRICS TO GET THE ACCURACY SCORE

        Pred=self.clf.predict(X_test)
        Prd=self.clf.score(X_test,Pred)

        Prd=Prd*100
        str_Prd="%d%%" % (Prd)

        #Show accuracy score of Testing
        self.tableWidget7.setItem(0,1, QtGui.QTableWidgetItem(str_Prd))
        self.canvas_pred.draw() #---show Testing plot

        #End Predict
    def testData_OSVM(self):

        with open(self.fileName_new_OSVM) as fp:
            reader=csv.reader(fp,delimiter=     ",")
            data=[line for line in reader]
            del data[0]
            datas=np.asfarray(data,float)

            X_test=datas[:, [0,1]]
            
        count_genuine=0
        count_intruder=0

        for p in X_test:   

            Pred=self.clf_OSVM.predict([p])
           
            Pred=Pred.astype(int)
            
            plt.figure(2, figsize=(3, 2))
            plt.scatter(p[0], p[1], s=200, marker='*', c=self.colors[Pred[0]])

            if Pred[0]==1:
                count_genuine=count_genuine+1
            if Pred[0]==-1:
                count_intruder=count_intruder+1

        if count_genuine>count_intruder:
            detect_label="GENUINE USER"
        if count_intruder>count_genuine:
            detect_label="IMPERSONATOR" 
        if count_genuine==count_intruder:
            detect_label="THE PREDICTED RESULT CANNOT BE LABELED"   

        self.tableWidget8.setItem(1,1, QtGui.QTableWidgetItem(detect_label))
        
        #Get the Accuracy Score in Predict

        #USE SKLEARN.METRICS TO GET THE ACCURACY SCORE

        Pred=self.clf_OSVM.predict(X_test)
        Prd=self.clf_OSVM.score(X_test,Pred)

        Prd=Prd*100
        str_Prd="%d%%" % (Prd)

        #Show accuracy score of Testing
        self.tableWidget8.setItem(0,1, QtGui.QTableWidgetItem(str_Prd))
        self.canvas_pred_OSVM.draw() #---show Testing plot

        #End Predict

if __name__ == "__main__":
    import sys

    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('MyWindow')

    main=MyWindow()
    main.show()

    sys.exit(app.exec_())