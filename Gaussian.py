import numpy as np
from numpy.linalg import norm

def Gaussian(X, C, u, arg = None):
    """
    Gaussian distribution.
    INPUT:
        X: data matrix
        C: root of covariance matric
        u: mean vector
        arg: output control
    OUTPUT:
        f, g, Hv
    """
    n, d = X.shape
    u = u.reshape(d, 1)
    Z = X - np.tile(u.T,(n,1))
    S = C.T.dot(C)
    f = np.exp(-norm(Z.dot(C.T),axis=1)**2/2)    
    
    if arg == 'f':
        return f
    
    g = f[:,None]*Z.dot(S)
    
    if arg == 'g':
        return g
    
    if arg == 'fg':
        return f, g
    
    if arg == None:
        Hv = lambda v: hessvec(Z, S, f, v)
        return f, g, Hv
    
def hessvec(Z, S, f, v):
    n, d = Z.shape
    V = np.tile(v.reshape(1,d),(n,1)) #nxd
    Hv = f[:,None]*(Z.dot(S)*Z.dot(S.dot(v))-V.dot(S))
    return Hv