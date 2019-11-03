import numpy as np
from numpy.linalg import norm
    
def myCG(A, b, tol, maxiter):
    """
    Conjugate gradient mathods. Solve Ax=b for PD matrices.
    INPUT:
        A: Positive definite matrix
        b: column vector
        tol: inexactness tolerance
        maxiter: maximum iterations
    OUTPUT:
        x: best solution x
        rel_res: relative residual
        T: iterations
    """
    b = b.reshape(len(b), 1)
    x = np.zeros((len(b),1))
    r = b
    T = 0
    rel_res_best = np.inf
    rel_res = 1
        
    delta = r.T.dot(r)
    p = np.copy(r)
    x = np.zeros((len(b), 1))
    
    while T < maxiter and rel_res > tol:
        T += 1
        Ap = Ax(A, p)
        pAp = p.T.dot(Ap)
#        if pAp < 0:
#            print('pAp =', pAp)
#            raise ValueError('pAp < 0 in myCG')
        alpha = delta/pAp
        x = x + alpha*p
        r = r - alpha*Ap
        rel_res = norm(r)/norm(b)            
        if rel_res_best > rel_res:
            rel_res_best = rel_res
        prev_delta = delta
        delta = r.T.dot(r)
        p = r + (delta/prev_delta)*p
    return x, rel_res, T

def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = A.dot(x)
    return Ax