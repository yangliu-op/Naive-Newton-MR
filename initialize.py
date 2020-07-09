import numpy as np
from numpy.linalg import inv, svd
from numpy.random import multivariate_normal, randn, rand
from logistic import logit
from optim_algo import (Newton_MR, Newton_CG, L_BFGS, Gauss_Newton, 
                        MomentumSGD, Adagrad, Adadelta, RMSprop, Adam, SGD)
from loaddata import loaddata
from showFigure import showFigure
from sklearn import preprocessing
from regularizer import regConvex, regNonconvex
import os
import matplotlib.pyplot as plt
from fraction import fraction
from pProfile import pProfile
from softmax import softmax
from least_square import least_square
from scipy import sparse
from GMM import GMM

    
def initialize(data, methods, prob, regType, x0Type, HProp_all, batchsize, 
               algPara, learningRate, total_run, fullHessian=True, 
               plotAll=False, lamda=1): 
    """
    data: name of chosen dataset
    methods: name of chosen algorithms
    prob: name of chosen objective problems
    regType: type of regularization 
    x0Type: type of starting point
    HProp_all: all Hessian proportion for sub-sampling methods
    batchsize: batchsize of first order algorithms
    algPara: a class that contains:
        mainLoopMaxItrs: maximum iterations for main loop
        funcEvalMax: maximum oracle calls (function evaluations) for algorithms
        innerSolverMaxItrs: maximum iterations for inner (e.g., CG) solvers
        lineSearchMaxItrs: maximum iterations for line search methods
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        innerSolverTol: inexactness tolerance of inner solvers
        beta: parameter of Armijo line-search
        beta2: parameter of Wolfe curvature condition (line-search)
        show: print result for every iteration
    learningRate: learning rate of the first order algorithms
    fullHessian: whether run Newton-type algorithm with full Hessian
    lamda: parameter of regularizer
    plotAll: plot 3 iterations plots as well
    """
    print('Initialization...')
    regType = regType[0]
    prob = prob[0]
    x0Type = x0Type[0]
    print('Problem:', prob, end='  ')
    print('regulization = %8s' % regType, end='  ')
    print('innerSolverMaxItrs = %8s' % algPara.innerSolverMaxItrs, end='  ')
    print('lineSearchMaxItrs = %8s' % algPara.lineSearchMaxItrs, end='  ')
    print('gradTol = %8s' % algPara.gradTol, end='  ')
    print('innerSolverTol= %8s' % algPara.innerSolverTol, end='  ')
    print('Starting point = %8s ' % x0Type)  
    if regType == 'None':
        reg = None
    if regType == 'Convex':
        reg = lambda x: regConvex(x, lamda)
    if regType == 'Nonconvex':
        reg = lambda x: regNonconvex(x, lamda)
        
    filename = '%s_reg_%s_%s_solItr_%s_x0_%s_subH_%s' % (
            prob, regType, data, algPara.innerSolverMaxItrs, x0Type, len(HProp_all))  
    mypath = filename
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    
    if prob == 'fraction':
        execute_fraction(x0Type, algPara, HProp_all, mypath, fullHessian, 
                         plotAll)
    else:
        if total_run != 1:            
            execute_pProfile(data, methods, x0Type, algPara, learningRate, 
                             HProp_all, batchsize, reg, mypath, fullHessian, 
                             total_run)
        else:
            methods_all, record_all = execute(data, methods, x0Type, algPara, 
                                              learningRate, HProp_all, 
                                              batchsize, reg, mypath, 
                                              fullHessian, prob, plotAll)
            

def execute_fraction(x0Type, algPara, HProp_all, mypath, fullHessian, plotAll):
    """
    Excute fraction problem.
    """
    obj = lambda x, control=None, HProp=None: fraction(
            x, HProp, control) 
    x0 = generate_x0(x0Type, 2)
    methods_all, record_all = run_algorithms(
            obj, x0, 'Newton_MR', algPara, None, HProp_all, 
            None, None, None, mypath, fullHessian=True)
    methods_all = ['Newton_MR',
            'Newton_MR_%.2E' % (HProp_all[0]),
            'Newton_MR_%.2E' % (HProp_all[1]),
            'Newton_MR_%.2E' % (HProp_all[2]),]
    showFigure(methods_all, record_all, 'fraction', mypath, plotAll)
    

def execute(data, methods, x0Type, algPara, learningRate, HProp_all, 
            batchsize, reg, mypath, fullHessian, prob, plotAll):  
    """
    Excute all methods/problems with 1 total run and give plots.
    """            
    if prob == 'gmm':
        train_X, theta, l, C1, C2 = loaddata_synthetic()
    else:
        data_dir = 'Data'
        print('Dataset:', data[0])
        train_X, train_Y, test_X, test_Y, idx = loaddata(data_dir, data[0])
        theta = None
        
#    train_X = scale_train_X(train_X, standarlize=False, normalize=False) 
    
    if prob == 'gmm':
        obj = lambda x, control=None, HProp=None: GMM(
                train_X, C1, C2, x, HProp, control, reg)
        obj_mini_g = lambda x, bs: GMM(train_X, C1, C2, x, arg='g', 
                                            reg=reg, batchsize=bs)
    
    if prob == 'softmax':      
        X, Y, l = sofmax_init(train_X, train_Y)    
        obj = lambda x, control=None, HProp=None: softmax(
                X, Y, x, HProp, control, reg)  
        obj_mini_g = lambda x, bs: softmax(
                X, Y, x, arg='g', reg=reg, batchsize=bs)
        
    if prob == 'nls':
        X, Y, l = nls_init(train_X, train_Y, idx=5)
        obj = lambda x, control=None, HProp=None: least_square(
                X, Y, x, HProp, control, reg)
        obj_mini_g = lambda x, bs: least_square(X, Y, x, arg='g', reg=reg, 
                                                batchsize=bs)
                      
    x0 = generate_x0(x0Type, l, zoom=100) 
        
    methods_all, record_all = run_algorithms(
            obj, x0, methods, algPara, learningRate, HProp_all, 
            batchsize, obj_mini_g, theta, mypath, fullHessian)
    
    showFigure(methods_all, record_all, prob, mypath, plotAll)
    
    return methods_all, record_all

  
def execute_pProfile(data, methods, x0Type, algPara, learningRate, HProp_all, 
                     batchsize, reg, mypath, fullHessian, total_run): 
    """
    Excute all methods/problems with multiple runs. Compare the best results
    and give performance profile plots between methods.
    
    Record a pProfile matrix s.t., every row are the best (f/g/error) result 
        of all listed methods. e.g., 500 total_run of 5 different 
        optimisation methods will be a 500 x 5 matrix.
    """            
    for j in range(total_run):
        print(' ')
        print('The %8g th Total Run' % (j+1))  
        train_X, theta, l, C1, C2 = loaddata_synthetic()   
        
        
        obj = lambda x, control=None, HProp=None: GMM(
                train_X, C1, C2, x, HProp, control, reg)
        obj_mini_g = lambda x, bs: GMM(train_X, C1, C2, x, arg='g', 
                                            reg=reg, batchsize=bs)
        
        x0 = generate_x0(x0Type, l)  
             
        record_all = []
        pProfile_fi = []
        pProfile_gi = []
        pProfile_erri = []

        methods_all, record_all = run_algorithms(
                obj, x0, methods, algPara, learningRate, HProp_all, 
                batchsize, obj_mini_g, theta, mypath, fullHessian)
        
        number_of_methods = len(methods_all)     
        pProfile_fi = np.zeros((1,number_of_methods))
        pProfile_gi = np.zeros((1,number_of_methods))
        pProfile_erri = np.zeros((1,number_of_methods))         
        
        for i in range(number_of_methods):
            record_matrices_i = record_all[i]
            pProfile_fi[0,i] = record_matrices_i[-1,0]
            pProfile_gi[0,i] = record_matrices_i[-1,1]
            pProfile_erri[0,i] = record_matrices_i[-1,-1]
            
        if j == 0:     
            pProfile_f = pProfile_fi
            pProfile_g = pProfile_gi
            pProfile_err = pProfile_erri
        else:
            pProfile_f = np.append(pProfile_f, pProfile_fi, axis=0)
            pProfile_g = np.append(pProfile_g, pProfile_gi, axis=0)
            pProfile_err = np.append(pProfile_err, pProfile_erri, axis=0)    
    
    figsz = (6,4)
    mydpi = 200      
    
    mypath_pProfile = 'pProfile'
    if not os.path.isdir(mypath_pProfile):
       os.makedirs(mypath_pProfile)
    
    fig1 = plt.figure(figsize=figsz)    
    pProfile(methods_all, pProfile_f)
    fig1.savefig(os.path.join(mypath_pProfile, 'objVal'), dpi=mydpi)
    
    fig2 = plt.figure(figsize=figsz)    
    pProfile(methods_all, pProfile_g)
    fig2.savefig(os.path.join(mypath_pProfile, 'gradNorm'), dpi=mydpi)
    
    fig3 = plt.figure(figsize=figsz)    
    pProfile(methods_all, pProfile_err)
    fig3.savefig(os.path.join(mypath_pProfile, 'err'), dpi=mydpi)
    
    with open(os.path.join(mypath_pProfile, 'methods.txt'), 'w') as myfile:
        for method in methods_all:
            myfile.write('%s\n' % method)
    np.savetxt(os.path.join(mypath_pProfile, 'objVal.txt'), pProfile_f, delimiter=',')
    np.savetxt(os.path.join(mypath_pProfile, 'gradNorm.txt'), pProfile_g, delimiter=',')
    np.savetxt(os.path.join(mypath_pProfile, 'err.txt'), pProfile_err, delimiter=',')
    
        
def run_algorithms(obj, x0, methods, algPara, learningRate, HProp_all, 
                   batchsize, obj_mini_g, theta, mypath, fullHessian=True):
    """
    Distribute all problems to its cooresponding optimisation methods.
    """
    record_all = []    
    if 'Newton_MR' in methods:  
        print(' ')
        if fullHessian:
            print('Newton_MR with Gradient Line-search')
            record_all.append('Newton_MR')
            HProp = None
            x, record = Newton_MR(
                    obj, x0, HProp, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                    algPara.innerSolverTol, algPara.innerSolverMaxItrs, 
                    algPara.lineSearchMaxItrs, algPara.gradTol, algPara.beta, 
                    theta, algPara.show)
            np.savetxt(os.path.join(mypath, 'Newton_MR.txt'), record, delimiter=',')
            record_all.append(record)
            
        for i in range(len(HProp_all)):
            obj_subsampled = lambda x, control=None: obj(x, control, HProp_all[i])
            myMethod = 'ssNewton_MR_%s%%' % (int(HProp_all[i]*100))
            record_all.append(myMethod)
            HProp = HProp_all[i]
            x, record = Newton_MR(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, theta, algPara.show)
            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
            record_all.append(record)
                    
    if 'L_BFGS' in methods:
        record_all.append('L_BFGS')
        print(' ')
        print('L_BFGS with Strong Wolfe Line-search')
        x, record = L_BFGS(
                obj, x0, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                algPara.lineSearchMaxItrs, algPara.gradTol, algPara.beta, 
                algPara.beta2, theta, algPara.show)
        np.savetxt(os.path.join(mypath, 'L_BFGS.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'Newton_CG' in methods:
        print(' ')
        if fullHessian:
            print('Newton_CG with Armijo Line-search')
            record_all.append('Newton_CG')
            HProp = None
            x, record = Newton_CG(
                    obj, x0, HProp, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                    algPara.innerSolverTol, algPara.innerSolverMaxItrs, 
                    algPara.lineSearchMaxItrs, algPara.gradTol, algPara.beta, 
                    theta, algPara.show)
            np.savetxt(os.path.join(mypath, 'Newton_CG.txt'), record, delimiter=',')
            record_all.append(record)
            
        for i in range(len(HProp_all)):
            obj_subsampled = lambda x, control=None: obj(x, control, HProp_all[i])
            myMethod = 'ssNewton_CG_%s%%' % (int(HProp_all[i]*100))   
            record_all.append(myMethod)
            HProp = HProp_all[i] 
            x, record = Newton_CG(
                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
                    algPara.funcEvalMax, algPara.innerSolverTol, 
                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
                    algPara.gradTol, algPara.beta, theta, algPara.show)
            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
            record_all.append(record)
            
    if 'Gauss_Newton' in methods:
        print(' ')
        if fullHessian:
            print('Gauss_Newton with Armijo Line-search') 
            HProp = None
            record_all.append('Gauss_Newton')
            x, record = Gauss_Newton(
                    obj, x0, HProp, algPara.mainLoopMaxItrs, algPara.funcEvalMax, 
                    algPara.innerSolverTol, algPara.innerSolverMaxItrs, 
                    algPara.lineSearchMaxItrs, algPara.gradTol, algPara.beta, 
                    theta, algPara.show)
            np.savetxt(os.path.join(mypath, 'Gauss_Newton.txt'), record, delimiter=',')
            record_all.append(record)
            
        #Gauss_newton methods perform badly on GMM, non execute
#        for i in range(len(HProp_all)):
#            obj_subsampled = lambda x, control=None: obj(x, control, HProp_all[i])
#            myMethod = 'ssGauss_Newton_%s%%' % (int(HProp_all[i]*100))   
#            record_all.append(myMethod)
#            HProp = HProp_all[i]
#            x, record = Gauss_Newton(
#                    obj_subsampled, x0, HProp, algPara.mainLoopMaxItrs, 
#                    algPara.funcEvalMax, algPara.innerSolverTol, 
#                    algPara.innerSolverMaxItrs, algPara.lineSearchMaxItrs, 
#                    algPara.gradTol, algPara.beta, theta, algPara.show)
#            np.savetxt(os.path.join(mypath, myMethod+'.txt'), record, delimiter=',')
#            record_all.append(record)
        
    if 'Momentum' in methods:
        print(' ')
        print('Momentum')
        record_all.append('Momentum')
        x, record = MomentumSGD(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.Momentum, algPara.gradTol, 
                theta, algPara.show)
        np.savetxt(os.path.join(mypath, 'Momentum.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'Adagrad' in methods:
        print(' ')
        print('Adagrad')
        record_all.append('Adagrad')
        x, record = Adagrad(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.Adagrad, algPara.gradTol, 
                theta, algPara.show)
        np.savetxt(os.path.join(mypath, 'Adagrad.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'Adadelta' in methods:
        print(' ')
        print('Adadelta')
        record_all.append('Adadelta')
        x, record = Adadelta(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.Adadelta, algPara.gradTol, 
                theta, algPara.show)
        np.savetxt(os.path.join(mypath, 'Adadelta.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'RMSprop' in methods:
        print(' ')
        print('RMSprop')
        record_all.append('RMSprop')
        x, record = RMSprop(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.RMSprop, algPara.gradTol, 
                theta, algPara.show)
        np.savetxt(os.path.join(mypath, 'RMSprop.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'Adam' in methods:
        print(' ')
        print('Adam')
        record_all.append('Adam')
        x, record = Adam(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.Adam, algPara.gradTol, 
                theta, algPara.show)
        np.savetxt(os.path.join(mypath, 'Adam.txt'), record, delimiter=',')
        record_all.append(record)
        
    if 'SGD' in methods:
        print(' ')
        print('SGD')
        record_all.append('SGD')
        x, record = SGD(
                obj, x0, obj_mini_g, batchsize, algPara.mainLoopMaxItrs, 
                algPara.funcEvalMax, learningRate.SGD, algPara.gradTol, 
                theta, algPara.show)
        np.savetxt(os.path.join(mypath, 'SGD.txt'), record, delimiter=',')
        record_all.append(record)
    
    methods_all = record_all[::2]
    record_all = record_all[1::2]
    
    return methods_all, record_all

        
def sofmax_init(train_X, train_Y):
    """
    Initialize data matrix for softmax problems.
    For multi classes classification.
    INPUT:
        train_X: raw training data
        train_Y: raw label data
    OUTPUT:
        train_X: DATA matrix
        Y: label matrix
        l: dimensions
    """
    n, d= train_X.shape
    Classes = sorted(set(train_Y))
    Total_C  = len(Classes)
    if Total_C == 2:
        train_Y = (train_Y == 1)*1
    l = d*(Total_C-1)
    I = np.ones(n)
    
    X_label = np.array([i for i in range(n)])
    Y = sparse.coo_matrix((I,(X_label, train_Y)), shape=(
            n, Total_C)).tocsr().toarray()
    Y = Y[:,:-1]
    return train_X, Y, l    

        
def nls_init(train_X, train_Y, idx=5):
    """
    Initialize data matrix for non-linear least square problems.
    For binary classification.
    INPUT:
        train_X: raw training data
        train_Y: raw label data
        idx: a number s.t., relabelling index >= idx classes into 1, the rest 0. 
    OUTPUT:
        train_X: DATA matrix
        Y: label matrix
        l: dimensions
    """
    n, d= train_X.shape
    Y = (train_Y >= idx)*1 #bool to int
    Y = Y.reshape(n,1)
    l = d
    return train_X, Y, l


def loaddata_synthetic(cond=4, n=1000, d=100):
    """
    Generate synthetic Gaussian Mixture Data
    INPUT:
        cond: Condition number control 10^(2*cond)
        n: number of samples
        d: dimension
    OUTPUT:
        train_X: synthetic data
        theta: a (2*d+1) column vector contains: 
            weights of 2 Gaussian distributions and their real mean
        l: 2*d+1
        C1: Root of covariance matrix of 1st Gaussian distribution
        C2: Root of covariance matrix of 2nd Gaussian distribution
    """
    #Condition number control 10^(2*cond)  
    t1 = randn(d,1)-1 #mu
    t2 = rand(d,1)+3 #mu
    W1 = randn(d,d)
    W2 = rand(d,d)
    U1, _, V1 = svd(W1, full_matrices=True)
    U2, _, V2 = svd(W2, full_matrices=True)
    s = np.diag(np.logspace(cond, 0, d))
    C1 = U1.dot(s.dot(V1.T))
    C2 = U2.dot(s.dot(V2.T))
    var1 = inv(C1.T.dot(C1))#sigma
    var2 = inv(C2.T.dot(C2))#sigma
    theta0 = randn(1,1)
    rho = logit(theta0)[0]
    train_X = rho*multivariate_normal(t1.T[0], 
            var1, n)+(1-rho)*multivariate_normal(t2.T[0], var2, n)  
    theta = np.r_[theta0, t1, t2]
    l = 2*d + 1
    return train_X, theta, l, C1, C2

def scale_train_X(train_X, standarlize=False, normalize=False): 
    """
    Standarlization/Normalization of trainning DATA.
    """
    if standarlize:
        train_X = preprocessing.scale(train_X)            
    if normalize:
        train_X = preprocessing.normalize(train_X, norm='l2')
    return train_X

    
def generate_x0(x0Type, l, zoom=1):    
    """
    Generate different type starting point.
    """
    if x0Type == 'randn':
        x0 = randn(l,1)/zoom
    if x0Type == 'rand':
        x0 = rand(l,1)/zoom
    if x0Type == 'ones':
        x0 = np.ones((l,1))
    if x0Type == 'zeros':
        x0 = np.zeros((l,1))
    return x0