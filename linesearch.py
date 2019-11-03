import numpy as np
from numpy.linalg import norm

def linesearch(obj, g, x, p, maxiter=200, alpha=1, c1=1e-4, c2=0.9):
    """    
    INPUTS:
        f: a function handle of objective value
        g: starting gradient gk
        x: starting x
        p: direction p
        maxiter: maximum iteration of Armijo line-search
        alpha: initial step-size
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    T = 0
    fa = obj(x + alpha*p, 'f')
    fk = obj(x, 'f')
    while fa > fk+alpha*c1*np.matmul(g.T,p) and T < maxiter:
        alpha *= 0.5
        fa = obj(x + alpha*p, 'f')
        T += 1    
#    if alpha < 1E-6: 
#        alpha = 0 # Kill small alpha
    x += alpha*p    
    return x, alpha, T

def linesearchgrad(obj, H, x, p, maxiter=200, alpha=1, c1=1e-4):
    """    
    INPUTS:
        g: a function handle of gradient
        H: Hessian Hk vector product
        x: starting x
        p: direction p
        maxiter: maximum iteration of Armijo line-search
        alpha: initial step-size
        c1: parameter of Armijo line-search
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    T = 0    
    ga = obj(x + alpha*p, 'g')
    gk = obj(x, 'g')
    while norm(ga)**2 > norm(gk)**2 + 2*alpha*c1*np.matmul(
            p.T, Ax(H, gk)) and T < maxiter:
        alpha *= 0.5
        ga = obj(x + alpha*p, 'g')
        T += 1       
#    if alpha < 1E-6:
#        alpha = 0 # Kill small alpha
    x += alpha*p        
    return x, alpha, T

def linesearchzoom(obj, x, p, maxiter=200, c1=1e-4, c2=0.9, alpha0=1):
    """    
    All vector are column vectors.
    INPUTS:
        fg: a function handle of both objective function and its gradient
        x: starting x
        p: direction p
        maxiter: maximum iteration of line search with strong Wolfe
        c1: parameter of Armijo line-search
        c2: parameter of strong Wolfe condition
        alpha0: initial step-size
    OUTPUTS:
        x: results
        alpha: proper step-size
        T: iterations
    """
    #phi(x) = f(x), phi'(x) = g(x).T.dot(p)
    itrs = 0
    itrs2 = 0
    a1 = 0
#    a2 = infun(0, amax)
    a2 = alpha0
    f0, g0 = obj(x, 'fg')
    fb = f0 # f preview
    while itrs < maxiter:
        fa, ga = obj(x + a2*p, 'fg')
        g0p = g0.T.dot(p)
        if fa > f0 + a2*c1*g0p or (fa >= fb and itrs > 0):
            alpha, itrs2 = zoomf(a1, a2, obj, f0, fa, g0p, x, p, c1, c2, itrs, maxiter)
            break
        gap = ga.T.dot(p)
        if abs(gap) <= -c2*g0p:
            alpha = a2
            itrs2 = 0
            break
        if gap >= 0:
            alpha, itrs2 = zoomf(a2, a1, obj, f0, fa, g0p, x, p, c1, c2, itrs, maxiter)
#            alpha, itrs2 = zoomf(a1, a2, fg, f0, fa, g0p, x, p, c1, c2, itrs, maxiter)
            break
        a2 = a2*2
        fb = fa
        itrs += 1
    itrs += itrs2
#     if itrs >= maxiter:        
#         alpha = 0
    return alpha, itrs

def zoomf(a1, a2, obj, f0, fa, g0p, x, p, c1, c2, itrs, maxiter):
    itrs2 = 0
    while (itrs2 + itrs) < maxiter :
        #quadratic
        itrs2 += 1
        # lower bound
        fa1, ga1 = obj(x + a1*p, 'fg')
        ga1p = ga1.T.dot(p)
        # upper bound
        fa2, ga2 = obj(x + a2*p, 'fg')
        ga2p = ga2.T.dot(p)
        aj = cubicInterp(a1, a2, fa1, fa2, ga1p, ga2p)
        a_mid = (a1 + a2)/2
        if not inside(aj, a1, a_mid):
            aj = a_mid
        faj, gaj = obj(x + aj*p, 'fg')
        if faj > f0 + aj*c1*g0p or faj >= fa1:
            a2 = aj
        else:
            gajp = gaj.T.dot(p)
            if np.abs(gajp) <= -c2*g0p:
                break
            if gajp*(a2 - a1) >= 0:
                a2 = a1
            a1 = aj
    return aj, itrs2

def cubicInterp(x1, x2, f1, f2, g1, g2):
    """
    find minimizer of the Hermite-cubic polynomial interpolating a
    function of one variable, at the two points x1 and x2, using the
    function (f1 and f2) and derivative (g1 and g2).
    """
    # Nocedal and Wright Eqn (3.59)
    d1 = g1 + g2 - 3*(f1 - f2)/(x1 - x2)
    d2 = np.sign(x2 - x1)*np.sqrt(d1**2 - g1*g2)
    xmin = x2 - (x2 - x1)*(g2 + d2 - d1)/(g2 - g1 + 2*d2);
    return xmin
    
def inside(x, a, b):
    """
    test x \in (a, b) or not
    """
    l = 0
    if not np.isreal(x):
        return l
    if a <= b:
        if x >= a and x <= b:
            l = 1
    else:
        if x >= b and x <= a:
            l = 1
    return l

def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax =A.dot(x)
    return Ax