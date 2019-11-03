import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from numpy.linalg import norm

def derivativetest(fun, x0):
    """
    Test the gradient and Hessian of a function. A large proportion 
    parallel in the middle of both plots means accuraccy.
    INPUTS:
        fun: a function handle that gives f, g, Hv
        x0: starting point
    OUTPUTS:
        derivative test plots
    """
    x0 = x0.reshape(len(x0),1)
    fun0 = fun(x0)
    dx = rand.randn(len(x0),1)
    M = 20;
    dxs = np.zeros((M,1))
    firsterror = np.zeros((M,1))
    order1 = np.zeros((M-1,1))
    seconderror = np.zeros((M,1))
    order2 = np.zeros((M-1,1))
    
    for i in range(M):
        x = x0 + dx
        fun1 = fun(x)
        H0 = Ax(fun0[2],dx)
        firsterror[i] = abs(fun1[0] - (fun0[0] + np.dot(
                dx.T, fun0[1])))/abs(fun0[0])
        seconderror[i] = abs(fun1[0] - (fun0[0] + np.dot(
                dx.T, fun0[1]) + 0.5* np.dot(dx.T, H0)))/abs(fun0[0])
        print('First Order Error is %8.2e;   Second Order Error is %8.2e'% (
                firsterror[i], seconderror[i]))
        if i > 0:
            order1[i-1] = np.log2(firsterror[i-1]/firsterror[i])
            order2[i-1] = np.log2(seconderror[i-1]/seconderror[i])
        dxs[i] = norm(dx)
        dx = dx/2
    
    step = [2**(-i-1) for i in range(M)]
    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.loglog(step, abs(firsterror),'b', label = '1st Order Err')
    plt.loglog(step, dxs**2,'r', label = 'order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(222)
    plt.semilogx(step[1:], order1,'b', label = '1st Order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(223)
    plt.loglog(step, abs(seconderror),'b', label = '2nd Order Err')
    plt.loglog(step, dxs**3,'r', label = 'Order')
    plt.gca().invert_xaxis()
    plt.legend()
    
    plt.subplot(224)
    plt.semilogx(step[1:], order2,'b', label = '2nd Order')
    plt.gca().invert_xaxis()
    plt.legend()
            
    return plt.show()


def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax =A.dot(x)
    return Ax