import pickle
import os
import struct
import numpy as np
from numpy.random import randint
import math

def loaddata(data_dir, data):    
    """
    Load RAW DATA from DATA file
    INPUT:
        data_dir: directory of data file
        data: name of the dataset
    OUTPUT:
        train_X, train_Y, test_X, test_Y
        idx: half of the total classes. 
            (for transferring every dataset into binary classification)
    """
    if data == 'cifar10':
        idx = 5
        train_X, train_Y, test_X, test_Y = cifar10(data_dir)
        
    if data == 'mnist':
        idx = 5
        train_X, train_Y, test_X, test_Y = mnist(data_dir)
        
    if data == 'arcene':
        idx = 1
        train_X, train_Y, test_X, test_Y = arcene(data_dir)
    
    if data == 'gisette':
        idx = 1
        train_X, train_Y, test_X, test_Y = gisette(data_dir)
        
    if data == 'hapt':
        idx = 6
        train_X, train_Y, test_X, test_Y = hapt(data_dir)
        
    if data == 'drive_diagnostics':
        idx = 6
        train_X, train_Y, test_X, test_Y = drive_diagnostics(data_dir)
        
    if data == 'BlogFeedback':
        idx = 1
        train_X, train_Y, test_X, test_Y = BlogFeedback(data_dir)
        
    if data == 'housing':
        idx = 20
        train_X, train_Y, test_X, test_Y = housing(data_dir)
        
    if data == 'forest_fire':
        idx = 1
        train_X, train_Y, test_X, test_Y = forest_fire(data_dir)
        
    if data == 'power_plant':
        idx = 450
        train_X, train_Y, test_X, test_Y = power_plant(data_dir)
        
    if data == 'covtype':
        idx = 3
        train_X, train_Y, test_X, test_Y = covtype(data_dir)

    return train_X, train_Y, test_X, test_Y, idx

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin-1')
    return data


def unpickle_csv(file):
    with open(file, 'rb') as fo:
        data = np.loadtxt(fo, delimiter=",", skiprows=1)           
    return data


def cifar10(data_dir):
    train_data = None
    train_labels = []
    
    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/cifar10/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data'].astype(np.float64)
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']
    
    test_data_dic = unpickle(data_dir + "/cifar10/test_batch")
    test_data = test_data_dic['data'].astype(np.float64)
    test_labels = test_data_dic['labels']    
    
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    
    return train_data, train_labels, test_data, test_labels


def mnist(data_dir):    
    train_labels=os.path.abspath(data_dir + '/MNIST/train-labels-idx1-ubyte')    
    train_data=os.path.abspath(data_dir + '/MNIST/train-images-idx3-ubyte')
    test_labels=os.path.abspath(data_dir + '/MNIST/t10k-labels-idx1-ubyte')
    test_data=os.path.abspath(data_dir + '/MNIST/t10k-images-idx3-ubyte')
    
    with open(train_labels, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        train_labels = np.fromfile(lbpath,dtype=np.uint8)
        
    with open(test_labels, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        test_labels = np.fromfile(lbpath,dtype=np.uint8)
    
    with open(train_data, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        train_data = np.fromfile(imgpath,dtype=np.uint8).reshape(len(train_labels), 784)
        
    with open(test_data, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
        test_data = np.fromfile(imgpath,dtype=np.uint8).reshape(len(test_labels), 784)
        
    train_data = train_data.astype(np.float64)
    test_data = test_data.astype(np.float64)    
        
    return train_data, train_labels, test_data, test_labels


def arcene(data_dir):
    train_data = np.loadtxt(data_dir + "/arcene/arcene_train.data")
    train_data = np.asarray(train_data, order='f', dtype=np.float64)    
    train_labels = np.loadtxt(data_dir + "/arcene/arcene_train.labels")
    
    test_data = np.loadtxt(data_dir + "/arcene/arcene_valid.data")
    test_labels = np.loadtxt(data_dir + "/arcene/arcene_valid.labels")
    test_data = np.asarray(test_data, dtype=np.float32)
    
    return train_data, train_labels, test_data, test_labels


def gisette(data_dir):
    train_data = np.loadtxt(data_dir + "/gisette/gisette_train.data")
    train_data = np.asarray(train_data, order='f', dtype=np.float64)    
    train_labels = np.loadtxt(data_dir + "/gisette/gisette_train.labels")
    
    test_data = np.loadtxt(data_dir + "/gisette/gisette_valid.data")
    test_labels = np.loadtxt(data_dir + "/gisette/gisette_valid.labels")
    test_data = np.asarray(test_data, dtype=np.float32)
    
    return train_data, train_labels, test_data, test_labels


def hapt(data_dir):
    train_data = np.loadtxt(data_dir + "/hapt/Train/X_train.txt")
    train_labels = np.loadtxt(data_dir + "/hapt/Train/Y_train.txt")
    
    test_data = np.loadtxt(data_dir + "/hapt/Test/x_test.txt")
    test_labels = np.loadtxt(data_dir + "/hapt/Test/y_test.txt")
    
    return train_data, train_labels-1, test_data, test_labels-1


def drive_diagnostics(data_dir):
    train_data = np.loadtxt(data_dir + "/drive_diagnostics/train_mat.txt")
    train_labels = np.loadtxt(data_dir + "/drive_diagnostics/train_vec.txt")
    
    test_data = np.loadtxt(data_dir + "/drive_diagnostics/test_mat.txt")
    test_labels = np.loadtxt(data_dir + "/drive_diagnostics/test_vec.txt")
    
    return train_data, train_labels, test_data, test_labels


def covtype(data_dir):
    raw_data = np.loadtxt(data_dir + "/CoveType/covtype.data", delimiter=',')
#    raw_data = np.loadtxt(data_dir + "/CoveType/covtype.txt", delimiter=',')
    train_raw = raw_data[:, 0:-1]
    labels_raw = raw_data[:, -1] - 1
    
    test_index = randint(len(labels_raw), size = math.ceil(len(labels_raw)/4))
    train_index =np.setdiff1d(np.arange(len(labels_raw)),test_index)
    train_data = train_raw[train_index, :]
    train_labels = labels_raw[train_index]
#    train_labels = train_labels[:,0:-1]
    test_data = train_raw[test_index, :]
    test_labels = labels_raw[test_index]
    
    return train_data, train_labels, test_data, test_labels


def housing(data_dir):
    raw_data = unpickle_csv(data_dir + "/housing/housing.csv")
    train_raw = raw_data[:, 0:-1]
    labels_raw = raw_data[:, -1]
    
    test_index = randint(len(labels_raw), size = math.ceil(len(labels_raw)/10))
    train_index =np.setdiff1d(np.arange(len(labels_raw)),test_index)
    train_data = train_raw[train_index, :]
    train_labels = labels_raw[train_index]
    test_data = train_raw[test_index, :]
    test_labels = labels_raw[test_index]
    return train_data, train_labels, test_data, test_labels


def BlogFeedback(data_dir):
    raw_data = unpickle_csv(data_dir + "/BlogFeedback/blogData_train.csv")
    train_raw = raw_data[:, 0:-1]
    labels_raw = raw_data[:, -1]
    
    test_index = randint(len(labels_raw), size = math.ceil(len(labels_raw)/10))
    train_index =np.setdiff1d(np.arange(len(labels_raw)),test_index)
    train_data = train_raw[train_index, :]
    train_labels = labels_raw[train_index]
    test_data = train_raw[test_index, :]
    test_labels = labels_raw[test_index]
    return train_data, train_labels, test_data, test_labels


def forest_fire(data_dir):
    raw_data = unpickle_csv(data_dir + "/forest_fire/forestfires.csv")
    train_raw = raw_data[:, 0:-1]
    labels_raw = raw_data[:, -1]
    
    test_index = randint(len(labels_raw), size = math.ceil(len(labels_raw)/10))
    train_index =np.setdiff1d(np.arange(len(labels_raw)),test_index)
    train_data = train_raw[train_index, :]
    train_labels = labels_raw[train_index]
    test_data = train_raw[test_index, :]
    test_labels = labels_raw[test_index]
    return train_data, train_labels, test_data, test_labels


def power_plant(data_dir):
    raw_data = unpickle_csv(data_dir + "/CCPP/power_plant.csv")
    train_raw = raw_data[:, 0:-1]
    labels_raw = raw_data[:, -1]
    
    test_index = randint(len(labels_raw), size = math.ceil(len(labels_raw)/10))
    train_index =np.setdiff1d(np.arange(len(labels_raw)),test_index)
    train_data = train_raw[train_index, :]
    train_labels = labels_raw[train_index]
    test_data = train_raw[test_index, :]
    test_labels = labels_raw[test_index]
    return train_data, train_labels, test_data, test_labels


def main():
    data_dir = 'Data'
    X_train, y_train, X_test, y_test = covtype(data_dir)
    print(X_train.shape,X_train.dtype) #(60000,784) uint8
    print(y_train.shape,y_train.dtype) #(60000,) uint8
    print(X_test.shape,X_test.dtype) #(10000,784) uint8
    print(np.mean(y_test),y_test.dtype) #(10000,) uint8
    
if __name__ == '__main__':
    main()
    