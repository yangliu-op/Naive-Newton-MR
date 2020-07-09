import numpy as np
import numpy.random as rand
from derivativetest import derivativetest
from logistic import logistic
from regularizer import regConvex, regNonconvex
from scipy.sparse import spdiags, csr_matrix

def least_square(X, y, w, HProp=None, arg=None, reg=None, act='logistic', 
                 batchsize=None):
    """
    Least square problem sum(phi(Xw) - y)^2, where phi is logistic function.
    INPUT:
        X: data matrix
        y: lable vector
        w: variables
        HProp: subsampled(perturbed) Hessian proportion
        arg: output control
        reg: regularizer control
        act: activation function
        batchsize: the proportion of mini-batch size
    OUTPUT:
        f, gradient, Hessian-vector product/Gauss_Newton_matrix-vector product
    """
    if reg == None:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    else:
        reg_f, reg_g, reg_Hv = reg(w)
        
    n, d = X.shape
    
    if batchsize is not None:
        n_mini = np.int(np.floor(n*batchsize))
        index_batch = np.random.choice(n, n_mini, replace = False)
        X = X[index_batch,:]
        y = y[index_batch]
        n = n_mini
        
    X = csr_matrix(X)    
    if act == 'logistic':
        fx, grad, Hess = logistic(X, w)
        
    #output control with least computation
    f = np.sum((fx-y)**2)/2/n + reg_f
    if arg == 'f':        
        return f    
    
    g = X.T.dot(grad*(fx-y))/n + reg_g
        
    if arg == 'g':        
        return g
        
    if arg == 'fg':        
        return f, g
    
    if arg == None:
        if HProp == None:
            #W is NxN diagonal matrix of weights with ith element=s2
            Hess = (grad**2 + Hess*(fx-y))/n
            W = spdiags(Hess.T[0], 0, n, n)
            XW = X.T.dot(W)
            Hv = lambda v: hessvec(XW, X, v) + reg_Hv(v)
            return f, g, Hv
        else:
            n_H = np.int(np.floor(n*HProp))
            idx_H = np.random.choice(n, n_H, replace = False)
            if act == 'logistic':
                fx_H, grad_H, Hess_H = logistic(X[idx_H,:], w)       
            Hess = (grad**2 + Hess*(fx-y))/n
            W = spdiags(Hess.T[0], 0, n, n)
            XW = X.T.dot(W)
#                fullHv = lambda v: hessvec(XW, X, v) + reg_Hv(v)
            Hess = (grad_H**2 + Hess_H*(fx_H-y[idx_H,:]))/n
            W = spdiags(Hess.T[0], 0, n_H, n_H)
            XW = X[idx_H,:].T.dot(W)
            Hv = lambda v: hessvec(XW, X[idx_H,:], v) + reg_Hv(v)
            return f, g, Hv
    
    if arg == 'gn': #hv product        
        if HProp == None:
            Hess_gn = grad**2/n
            W = spdiags(Hess_gn.T[0], 0, n, n)
            XW = X.T.dot(W)
            Hv = lambda v: hessvec(XW, X, v) + reg_Hv(v)
            return f, g, Hv    
        else:
            n_H = np.int(np.floor(n*HProp))
            idx_H = np.random.choice(n, n_H, replace = False)
            if act == 'logistic':
                fx_H, grad_H, Hess_H = logistic(X[idx_H,:], w)          
            Hess_gn = grad_H**2/n
            W = spdiags(Hess_gn.T[0], 0, n_H, n_H)
            XW = X[idx_H,:].T.dot(W)
            Hv = lambda v: hessvec(XW, X[idx_H,:], v) + reg_Hv(v)
            return f, g, Hv
    
def hessvec(XW, X, v):
    H2 = X.dot(v)
    Hv = XW.dot(H2)
    return Hv

#@profile
def main():        
    n = 100
    d = 50
    total_C = 2
    X = rand.randn(n,d)
    I = np.eye(total_C, total_C - 1)
    ind = rand.randint(total_C, size = n)
    Y = I[ind, :]
    w = rand.randn(d*(total_C-1),1)
    lamda = 1
#    reg = None
#    reg = lambda x: regConvex(x, lamda)
    reg = lambda x: regNonconvex(x, lamda)
    fun1 = lambda x: least_square(X,Y,x,act='logistic', reg = reg)
    derivativetest(fun1,w)    
#    
if __name__ == '__main__':
    main()