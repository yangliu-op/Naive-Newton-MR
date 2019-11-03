import numpy as np

def regConvex(x, lamda, arg=None):
    """
    Returns f, g, Hv of lamda*||x||^2
    """
    f = 0.5*lamda*np.dot(x.T, x)   
    if arg == 'f':
        return f
    
    g = lamda*x    
    if arg == 'g':
        return g
    if arg == 'fg':
        return f, g    
    
    Hv = lambda v: lamda*v
    
    if arg is None:
        return f, g, Hv    
    
def regNonconvex(x, lamda, a=1, arg=None):
    """
    returns  f, g, Hv of lamda*a*x^2/(1+a*x^2)
    """
    ax2 = a*x**2
    f = lamda*sum(ax2/(1+ax2))  
    if arg == 'f':
        return f
    
    g = lamda*2*a*x/(1+ax2)**2    
    if arg == 'g':
        return g
    if arg == 'fg':
        return f, g    
    
    Hv = lambda v: lamda*2*a*(1-3*ax2)/(1+ax2)**3*v
    
    if arg is None:
        return f, g, Hv    