import numpy as np
from numpy.random import multivariate_normal, randn, rand
from numpy.linalg import inv, svd, norm
from derivativetest import derivativetest
from regularizer import regConvex, regNonconvex
from logistic import logit_rho, logit

def GMM(X, C1, C2, theta, HProp= None, arg=None, reg=None, batchsize=None):
    '''
    Gaussian mixture model is a mixture of two Gaussian distribution here.
    F(x0, x1, x2) = sum log[*e^(-(ai - x1)^2/Sigma1/2) + 
                         (1 - 1/(1+e^(-x0)))e^(-(ai - x2)^2/Sigma2/2)]
    where phi(x0) = 1/(1+e^(-x0))
    INPUT:
        X: input nxd data matrix
        C1, C2: C1.T.dot(C1) = 1/Sigma1, C2.T.dot(C2) = 1/Sigma2
        theta: the [x0, x1, x2] variable.
        arg: output control
        reg: regulizer control
        HProp: porposion of Hessian perturbation
        batchsize: the proportion of mini-batch size
    OUTPUT:
        f, gradient, Hessian-vector product/Gauss_Newton_matrix-vector product
    '''
    if reg == None:
        reg_f = 0
        reg_g = 0
        reg_Hv = lambda v: 0
    else:
        reg_f, reg_g, reg_Hv = reg(theta)
    n, d = X.shape
    
    if batchsize is not None:
        n_mini = np.int(np.floor(n*batchsize))
        index_batch = np.random.choice(n, n_mini, replace = False)
        X = X[index_batch,:]
        n = n_mini
    rho, rhog, rhoH = logit(theta[0])
    
    W = np.vsplit(theta[1:].reshape(len(theta)-1,1), 2)
    

    Z1 = X - np.tile(W[0].T,(n,1))
    S1 = C1.T.dot(C1) # 1/Sigma1
    U1 = (norm(Z1.dot(C1.T),axis=1)**2/2).reshape(n,1)
    
    Z2 = X - np.tile(W[1].T,(n,1))
    S2 = C2.T.dot(C2) # 1/Sigma2
    U2 = (norm(Z2.dot(C2.T),axis=1)**2/2).reshape(n,1)
    t = U2-U1
    
    fx = np.zeros((n,1))
    fx[t<=0] = -U2[t<=0] + np.log(1-rho+rho*np.exp(t[t<=0]))
    fx[t>0] = -U1[t>0] + np.log(rho+(1-rho)*np.exp(-t[t>0]))      
    fx = -sum(fx)/n + reg_f
    
    if arg == 'f':
        return fx
    #logit(t, rho) = f1/(rho*f1+(1-rho)*f2), 
    #logit(-t, 1-rho) = f2/(rho*f1+(1-rho)*f2)
    ZS1 = Z1.dot(S1)
    ZS2 = Z2.dot(S2)
    #G0 nx1 (f1-f2)/(rho*f1+(1-rho)*f2)
    G0 = rhog*(logit_rho(t, rho) - logit_rho(-t, 1-rho))
    G1 = logit_rho(t, rho)*ZS1*rho #nxd
    G2 = logit_rho(-t, 1-rho)*ZS2*(1-rho)#nxd
    G = np.append(G0, np.append(G1, G2,axis=1), axis=1) #n*(2d+1)
    grad = -(sum(G)/n).reshape(2*d+1,1) + reg_g #n*(2d+1)
    
    if arg == 'g': 
        return grad
    
    if arg == 'fg':
        return fx, grad 
    
    if arg == None:
        if HProp == None:
            Hess = lambda v: hessvec(X, G, t, G0, G1, G2, ZS1, ZS2, S1, S2, v, 
                                     rho, rhog, rhoH) + reg_Hv(v)
            return fx, grad, Hess
        else:
            n_H = np.int(np.floor(n*HProp))
            idx_H = np.random.choice(n, n_H, replace = False)
            Hess = lambda v: hessvec(X[idx_H,:], G[idx_H,:], t[idx_H,:], G0[
                    idx_H,:], G1[idx_H,:], G2[idx_H,:], 
                                         ZS1[idx_H,:], ZS2[idx_H,:], S1, S2, v, 
                                         rho, rhog, rhoH) + reg_Hv(v)
            return fx, grad, Hess
            
    if arg == 'gn':
        if HProp == None:
            Hess = lambda v: hessvec(X, G, t, G0, G1, G2, ZS1, ZS2, S1, S2, v, 
                                     rho, rhog, rhoH, 'gn') + reg_Hv(v)
            return fx, grad, Hess
        else:
            n_H = np.int(np.floor(n*HProp))
            idx_H = np.random.choice(n, n_H, replace = False)
            Hess = lambda v: hessvec(X[idx_H,:], G[idx_H,:], t[idx_H,:], G0[
                    idx_H,:], G1[idx_H,:], G2[idx_H,:], 
                                     ZS1[idx_H,:], ZS2[idx_H,:], S1, S2, v, 
                                     rho, rhog, rhoH, 'gn') + reg_Hv(v)   
            return fx, grad, Hess
        
def hessvec(X, G, t, G0, G1, G2, ZS1, ZS2, S1, S2, v, rho, rhog, rhoH, arg=None):
    n, d = X.shape
    V0 = v[0]
    V = np.vsplit(v[1:].reshape(len(v)-1,1), 2)   
    V1 = V[0]
    V2 = V[1]
    Gv = G.dot(v) #nx1
    if arg == None:
        Hess0 = sum(G0*V0*rhoH/rhog + G1.dot(V1)*rhog/rho - 
                    G2.dot(V2)*rhog/(1-rho))/n - G0.T.dot(Gv)/n
        Hess1 = (sum(Hessv1(ZS1, S1, t, rho, V1) + 
                     rhog*G1*V0/rho)/n).reshape(d,1) - G1.T.dot(Gv)/n
        Hess2 = (sum(Hessv2(ZS2, S2, t, rho, V2) - 
                     rhog*G2*V0/(1-rho))/n).reshape(d,1) - G2.T.dot(Gv)/n
        Hess = -np.append(Hess0, np.append(Hess1, Hess2, axis=0), axis=0)
    if arg == 'gn':
        Hess0 = sum(G1.dot(V1)/rho - G2.dot(V2)/(1-rho))/n - G0.T.dot(Gv)/n
        Hess1 = (sum(rhog*G1*V0/rho)/n).reshape(d,1) - G1.T.dot(Gv)/n
        Hess2 = (sum(-rhog*G2*V0/(1-rho))/n).reshape(d,1) - G2.T.dot(Gv)/n
        Hess = -np.append(Hess0, np.append(Hess1, Hess2, axis=0), axis=0)
    return Hess

def Hessv1(ZS, S, t, rho, v):
    n, d = ZS.shape
    V = np.tile(v.reshape(1,d),(n,1)) #nxd
    Hv = rho*logit_rho(t, rho)*(ZS*ZS.dot(v)-V.dot(S))
    return Hv

def Hessv2(ZS, S, t, rho, v):
    n, d = ZS.shape
    V = np.tile(v.reshape(1,d),(n,1)) #nxd
    Hv = (1-rho)*logit_rho(-t, 1-rho)*(ZS*ZS.dot(v)-V.dot(S))
    return Hv

def gmmtest(x, u):
    rho_u = logit(u[0], 'fx')
    u_tmp = np.vsplit(u[1:].reshape(len(u)-1,1), 2)
    uhat = np.append(rho_u*u_tmp[0], (1-rho_u)*u_tmp[1], axis=0)
    
    rho_x = logit(x[0], 'fx')
    x_tmp = np.vsplit(x[1:].reshape(len(x)-1,1), 2)
    xhat = np.append(rho_x*x_tmp[0], (1-rho_x)*x_tmp[1], axis=0)
    
#    err = (abs(rho_x - rho_u)/rho_u + norm(xhat - uhat)/norm(uhat) )/2
    err = norm(xhat - uhat)/norm(uhat)    
    return err
    
#def gmmtest(x, u):        
#    # gmmtest of Newon_MR paper
#    err = (abs(x[0] - u[0])/abs(u[0]) + (norm(x[1:] - u[1:])/norm(u[1:])))/2
#    return err
    

def main():    
    n=100
    d = 200
    t1 = randn(d,1)+1 #mu
    t2 = rand(d,1)+3 #mu
    W1 = randn(d,d)
    W2 = randn(d,d)
    cond = 1 #Condition number control
    U1, _, V1 = svd(W1, full_matrices=True)
    U2, _, V2 = svd(W2, full_matrices=True)
#    s = np.diag(np.logspace(np.log10(cond), 0, d))
    s = np.diag(np.logspace(cond, 0, d))
    C1 = U1.dot(s.dot(V1.T))
    C2 = U2.dot(s.dot(V2.T))
    #train_X = 0.5*multivariate_normal(t1.T[0], np.identity(
    #        d), n)+0.5*multivariate_normal(t2.T[0], np.identity(d), n)
    var1 = inv(C1.T.dot(C1))#sigma
    var2 = inv(C2.T.dot(C2))#sigma
#    print(norm(var))
#    print(var)
    
    theta0 = randn(1,1)
#    rho = 1/(1+np.exp(-theta0))
    rho = logit(theta0, 'fx')
    train_X = rho*multivariate_normal(t1.T[0], 
            var1, n)+(1-rho)*multivariate_normal(t2.T[0], var2, n)
    lamda = 1
    reg = None
#    reg = lambda x: regConvex(x, lamda)
#    reg = lambda x: regNonconvex(x, lamda)
    obj = lambda u: GMM(train_X, C1, C2, u, reg=reg)
#    obj = lambda u: GMM(train_X, C1, C2, u, 'gn', reg=reg)
#    x0 = np.ones((d*2+1,1))*1.5
#    x0 = np.zeros((d*2+1,1))
    x0 = randn(d*2+1,1)
    derivativetest(obj, x0)
    
if __name__ == '__main__':
    main()