import numpy as np
from numpy.linalg import svd
from numpy.random import randn
from derivativetest import derivativetest

def fraction(x, HProp = None, arg = None, a=100, b=1):
    """
    Fraction problems: 
        F(x, y): a*x^2/(b - y)
    INPUT:
        x: variable
        HProp: porposion of Hessian perturbation
        arg: output control
    OUTPUT:
        f, gradient, Hessian/perturbed Hessian
    """
    x1 = x[0]
    x2 = x[1]
    H1 = a/(x2 - b) # a/(x2-b)
    f = -H1*x1**2
    H2 = H1/(x2 - b) # a/(x2-b)^2
    H3 = H2/(x2 - b) # a/(x2-b)^3
    
    g1 = -2*x1*H1
    g2 = H2*x1**2
    g = np.append(g1,g2,axis=0).reshape(2,1)
        
    if arg == 'f':
        return f
    
    if arg == 'g':
        return g
    if arg == 'fg':
        return f, g
    
    if arg == 'Hv':
        Hv = lambda v: Hessvec(H1, H2, H3, x1, x2, v)
        if HProp != None:
            Hv2 = lambda v: Hessvec(H1, H2, H3, x1, x2, v) + HProp*v
            return f, g, Hv2, Hv 
        else:
            return f, g, Hv 
        
    if arg is None:
        H23 = 2*x1*H2
        H_top = np.append(-2*H1, H23)
        H_bot = np.append(H23, -2*x1**2*H3)
        H = np.c_[H_top, H_bot]
        if HProp == None:
            return f, g, H
        else:
            s = svd(H)[1]
            # Stability Analysis of Newton-MR Under Hessian Perturbations
            # Condition 1, \epsilon < \gamma/4 (2\nu -1)
            eps = 1E-12 # ignore small epsilon
            sigma = s[s>eps]
            if len(sigma) == 0:
                raise ValueError('0 rank Hessian!')
            epsilon = min(sigma)
            E = np.eye(2)*epsilon*HProp # \vnorm{\EE} = \epsllon*HProp
            return f, g, H+E
     
    
def Hessvec(H1, H2, H3, x1, x2, v):
    v1 = v[0]
    v2 = v[1]
    H23 = 2*x1*H2
    Hv1 = -2*H1*v1 + H23*v2
    Hv2 = H23*v1 - 2*x1**2*H3*v2
    Hv = np.append(Hv1,Hv2,axis=0).reshape(2,1)
    return Hv


def main():    
    obj = lambda w: fraction(w,a=2,b=1)
    x0 = randn(2,1)
    derivativetest(obj, x0) 
    
    
if __name__ == '__main__':
    main()