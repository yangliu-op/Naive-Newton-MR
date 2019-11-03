import numpy as np

def logistic(X, w, arg = None):
    """
    Logistic function
    INPUT:
        X: Matrix
        w: variable vector
        arg: output control
    OUTPUT:
        f, gradient, Hessian
    """
    #s = 1/(1+ np.exp(-t)) =  e^t/(1+e^t) = e^(t-M)/a
    #a = e^(-M)+e^(t-M)
    n, d = X.shape
    w = w.reshape(d,1) 
    t = X.dot(w)
    M = np.maximum(0,t)
    power = np.append(-M, t-M, axis = 1)
    a = np.sum(np.exp(power), axis = 1).reshape(len(t), 1)
    s = np.exp(t-M)/a
    
    if arg == 'fx':
        return s
    
    if arg == 'grad':
        g = s*(1-s)
        return g
    
    if arg == 'Hess':
        H = s*(1-s)*(1-2*s)
        return H
    
    if arg == None:
        g = s*(1-s)
        H = s*(1-s)*(1-2*s)
        return s, g, H


def logit(w, arg = None):
    """
    Logistic function with out linear predictor
    """
    d = len(w)
    w = w.reshape(d,1) 
    t = w
    M = np.maximum(0,t)
    power = np.append(-M, t-M, axis = 1)
    a = np.sum(np.exp(power), axis = 1).reshape(d, 1)
    s = np.exp(t-M)/a
    s = s[0]
    
    if arg == 'fx':
        return s
    
    if arg == 'grad':
        g = s*(1-s)
        return g
    
    if arg == 'Hess':
        H = s*(1-s)*(1-2*s)
        return H
    
    if arg == None:
        g = s*(1-s)
        H = s*(1-s)*(1-2*s)
        return s, g, H

def logit_rho(w, rho, arg = None):
    """
    Tool function for GMM model, s.t., 
    logit(t, rho) = f1/(rho*f1+(1-rho)*f2), 
    logit(-t, 1-rho) = f2/(rho*f1+(1-rho)*f2)
    """
    d = len(w)
    w = w.reshape(d,1) 
    t = w
    M = np.maximum(0,t)
    power = np.append(-M, t-M, axis = 1)
    ep = np.exp(power)
    ep[ep==1] = ep[ep==1]*rho/(1-rho)
    a = np.sum(ep, axis = 1).reshape(d, 1)
    s = np.exp(t-M)/a/(1-rho)
    return s