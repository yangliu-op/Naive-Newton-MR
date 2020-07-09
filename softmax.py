import numpy as np
from scipy.sparse import spdiags, identity
import numpy.random as rand
from numpy import matlib as mb
from derivativetest import derivativetest
from scipy.linalg import block_diag
from regularizer import regConvex, regNonconvex


def softmax(X, Y, w, HProp = None, arg=None, reg=None, batchsize=None):  
    """
    All vectors are column vectors.
    INPUTS:
        X: a nxd data matrix.
        Y: a nxC label matrix, C = total class number - 1
        w: starting point
        HProp: porposion of Hessian perturbation
        arg: output control
        reg: a function handle of regulizer function that returns f,g,Hv
        batchsize: the proportion of mini-batch size
    OUTPUTS:
        f: objective function value
        g: gradient of objective function
        Hv: a Hessian-vector product function handle of any column vector v
        H: Hessian matrix of objective function
    """
    if reg == None:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    else:
        reg_f, reg_g, reg_Hv = reg(w)
    global d, C
    n, d = X.shape
    
    if batchsize is not None:
        n_mini = np.int(np.floor(n*batchsize))
        index_batch = np.random.choice(n, n_mini, replace = False)
#        print(index_batch[:5])
        X = X[index_batch,:]
        Y = Y[index_batch]
        n = n_mini
        
    C = int(len(w)/d)
    w = w.reshape(d*C,1) #[d*C x 1]
    W = w.reshape(C,d).T #[d x C]
    XW = np.dot(X,W) #[n x C]
    large_vals = np.amax(XW,axis = 1).reshape(n, 1) #[n,1 ]
    large_vals = np.maximum(0,large_vals) #M(x), [n, 1]
    #XW - M(x)/<Xi,Wc> - M(x), [n x C]
    XW_trick = XW - np.tile(large_vals, (1, C))
    #sum over b to calc alphax, [n x total_C]
    XW_1_trick = np.append(-large_vals, XW_trick,axis = 1)
    #alphax, [n, ]
    sum_exp_trick = np.sum(np.exp(XW_1_trick), axis = 1).reshape(n, 1)
    log_sum_exp_trick = large_vals + np.log(sum_exp_trick)  #[n, 1]
    
    f = np.sum(log_sum_exp_trick)/n - np.sum(np.sum(XW*Y,axis=1))/n + reg_f
    if arg == 'f':        
        return f
    inv_sum_exp = 1./sum_exp_trick
    inv_sum_exp = np.tile(inv_sum_exp,(1,np.size(W,axis = 1)))
    S = inv_sum_exp*np.exp(XW_trick) #h(x,w), [n x C] 
    g = np.dot(X.T, S-Y)/n #[d x C]
    g = g.T.flatten().reshape(d*C,1) + reg_g#[d*C, ]  

    if arg == 'g':
        return g    
    
    if arg == 'fg':
        return f, g

    if HProp == None:
        Hv = lambda v: hessvec(X, S, n, v) + reg_Hv(v)   
        return f, g, Hv
    else:
        n_H = np.int(np.floor(n*HProp))
        idx_H = np.random.choice(n, n_H, replace = False)
        inv_sum_exp_H = 1./(sum_exp_trick[idx_H,:])
        inv_sum_exp_H = np.tile(inv_sum_exp_H,(1,np.size(W,axis = 1)))
        S_H = inv_sum_exp_H*np.exp(XW_trick[idx_H,:]) #h(x,w), [S x C] 
        Hv = lambda v: hessvec(X[idx_H,:], S_H, n_H, v) + reg_Hv(v)
        return f, g, Hv
        
    if arg == 'explicit':
        f = np.sum(log_sum_exp_trick) - np.sum(np.sum(XW*Y,axis=1)) + reg_f
        g = np.dot(X.T, S-Y) #[d x C]
        g = g.T.flatten().reshape(d*C,1) + reg_g #[d*C, ]
        Hv = lambda v: hessvec(X, S, v, reg)
        #S is divided into C parts {1:b}U{c}, [n, ] * C
        S_cell = np.split(S.T,C) 
        SX_cell = np.array([]).reshape(n,0) #empty [n x 0] array
        SX_self_cell = np.array([]).reshape(0,0)
        for column in S_cell:
            c = spdiags(column,0,n,n) #value of the b/c class
            SX_1_cell = np.dot(c.A,X) #WX = W x X,half of W, [n x d]
            #fill results from columns, [n x d*C]
            SX_cell = np.c_[SX_cell, SX_1_cell] 
            SX_cross = np.dot(SX_cell.T,SX_cell) #take square, [d*C x d*C]     
            #X.T x WX        half of W, [d x d]
            SX_1self_cell = np.dot(X.T,SX_1_cell) 
            #put [d x d] in diag, W_cc, [d*C x d*C]  
            SX_self_cell = block_diag(SX_self_cell,SX_1self_cell) 
            H = SX_self_cell - SX_cross #compute W_cc, [d*C x d*C]
        H = H + 2*reg*identity(d*C)
        return f, g, Hv, H

def hessvec(X, S, n, v):
    v = v.reshape(len(v),1)
    V = v.reshape(C, d).T #[d x C]
    A = np.dot(X,V) #[n x C]
    AS = np.sum(A*S, axis=1).reshape(n, 1)
    rep = mb.repmat(AS, 1, C)#A.dot(B)*e*e.T
    XVd1W = A*S - S*rep #[n x C]
    Hv = np.dot(X.T, XVd1W)/n #[d x C]
    Hv = Hv.T.flatten().reshape(d*C,1)#[d*C, ] #[d*C, ]
    return Hv

#@profile
def main():    
    rand.seed(1)
    n = 100
    d = 50
    total_C = 2
    X = rand.randn(n,d)
    I = np.eye(total_C, total_C - 1)
    ind = rand.randint(total_C, size = n)
    Y = I[ind, :]
    lamda = 0
#    reg = None
    reg = lambda x: regConvex(x, lamda)
#    reg = lambda x: regNonconvex(x, lamda)
    w = rand.randn(d*(total_C-1),1)
    fun = lambda x: softmax(X,Y,x, reg=reg)
    derivativetest(fun,w)    

if __name__ == '__main__':
    main()