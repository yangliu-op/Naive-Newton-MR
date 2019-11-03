# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 23:43:12 2018

Note that: 
    This code is modified from the minresQLP algorithms:
        http://www.stanford.edu/group/SOL/software.html
    Edited to suit Newton-MR methods.
        
Contact: 
    yang.liu2(AT)uq.edu.au

REFERENCES:
    Sou-Cheng T. Choi, Christopgher C. Paige, and Michael A. Saunders,
    MINRES-QLP: A Krylov subspace method for indefinite or singular symmetric
    systems, SIAM Journal of Scientific Computing, submitted on March 7, 2010.

    Sou-Cheng T. Choi's PhD Dissertation, Stanford University, 2006:
        http://www.stanford.edu/group/SOL/dissertations.html
    
    Yang Liu, Fred Roosta
    Stability Analysis of Newton-MR Under Hessian Perturbations, 
        https://arxiv.org/abs/1909.06224
    
    Fred Roosta, Yang Liu, Peng Xu, Michael W. Mahoney
    Newton-MR: Newton's Method Without Smoothness or Convexity
        https://arxiv.org/abs/1810.00303

--------------------------------------------------------------------------
MinresQLP: Aim to obtain the min-length solution of symmetric 
   (possibly singular) Ax=b or min||Ax-b||.

   X = MinresQLP(A,B) solves the system of linear equations A*X=B
   or the least-squares problem min norm(B-A*X) if A is singular.
   The N-by-N matrix A must be symmetric or Hermitian, but need not be
   positive definite or nonsingular.  It may be double or single.
   The rhs vector B must have length N.  It may be real or complex,
   double or single, 

   X = MinresQLP(AFUN,B) accepts a function handle AFUN instead of
   the matrix A.  Y = AFUN(X) returns the matrix-vector product Y=A*X.
   In all of the following syntaxes, A can be replaced by AFUN.

   X = MinresQLP(A,B,RTOL) specifies a stopping tolerance.
   If RTOL=[] or is absent, a default value is used.
   (Similarly for all later input parameters.)
   Default RTOL=1e-6.

   X = MinresQLP(A,B,RTOL,MAXIT)
   specifies the maximum number of iterations.  Default MAXIT=N.

   X = MinresQLP(A,B,RTOL,MAXIT,M)
   uses a matrix M as preconditioner.  M must be positive definite
   and symmetric or Hermitian.  It may be a function handle MFUN
   such that Y=MFUN(X) returns Y=M\X.
   If M=[], a preconditioner is not applied.

   X = MinresQLP(A,B,RTOL,MAXIT,M,SHIFT)
   solves (A - SHIFT*I)X = B, or the corresponding least-squares problem
   if (A - SHIFT*I) is singular, where SHIFT is a real or complex scalar.
   Default SHIFT=0.

   X = MinresQLP(A,B,RTOL,MAXIT,M,SHIFT,MAXXNORM,ACONDLIM,TRANCOND)
   specifies three parameters associated with singular or
   ill-conditioned systems (A - SHIFT*I)*X = B.

   MAXXNORM is an upper bound on NORM(X).
   Default MAXXNORM=1e7.

   ACONDLIM is an upper bound on ACOND, an estimate of COND(A).
   Default ACONDLIM=1e15.

   TRANCOND is a real scalar >= 1.
   If TRANCOND>1,        a switch is made from MINRES iterations to
                         MINRES-QLP iterationsd when ACOND >= TRANCOND.
   If TRANCOND=1,        all iterations will be MINRES-QLP iterations.
   If TRANCOND=ACONDLIM, all iterations will be conventional MINRES
                         iterations (which are slightly cheaper).
   Default TRANCOND=1e7.   

   FLAG:        
   -1 (beta2=0)  B and X are eigenvectors of (A - SHIFT*I).
    0 (beta1=0)  B = 0.  The exact solution is X = 0.               
    1 X solves the compatible/incompatible (possibly singular) system 
    (A - SHIFT*I)X = B to the desired inexactness:
         <B, AX> < - ( 1 - rtol) ||B||^2
    2 X converged to an eigenvector of (A - SHIFT*I).
    3 XNORM exceeded MAXXNORM.
    4 ACOND exceeded ACONDLIM.
    5 MAXIT iterations were performed before one of the previous
      conditions was satisfied.
    9 The system appears to be exactly singular.  XNORM does not
      yet exceed MAXXNORM, but would if further iterations were
      performed.

    ITER:  the number of iterations performed.
    QLPITER:  the number of MINRES-QLP iterations.
    
    RELRES: Relative residuals for (A - SHIFT*I)X = B            
    ANORM:  an estimate of the 2-norm of A-SHIFT*I.
    ACOND:  an estimate of COND(A-SHIFT*I,2).
    XNORM:  a recurred estimate of NORM(X). 
"""

import numpy as np
from numpy.linalg import norm
from myCG import myCG

def MinresQLP(A, b, rtol=None, maxit=None, M=None, shift=None,maxxnorm=None,
              Acondlim=None, TranCond=None):
    if rtol is None:
        rtol = 1e-4
    if maxit is None:
        maxit = 100
    if shift is None:
        shift = 0
    if maxxnorm is None:
        maxxnorm = 1e7
    if Acondlim is None:
        Acondlim = 1e15
    if TranCond is None:
        TranCond = 1e7
    n = len(b)
    x0 = np.zeros((n,1))
    x = np.copy(x0)    
    b = b.reshape(n,1)
    Ab = Ax(A, b)
    r2 = b
    r3 = r2
    beta1 = norm(r2)
    
    #function handle with M x r_hat = r
    if M is None:
        #test with M = lambda u: u
        noprecon = True
        pass
    else:
        noprecon = False
        r3 = Precond(M, r2)
        beta1 = np.matmul(r3.T,r2) #theta
        if beta1 <0:
            print('Error: "M" is indefinite!')
        else:
            beta1 = np.sqrt(beta1)
    
    ## Initialize
    flag0 = -2
    flag = -2
    iters = 0
    QLPiter = 0
    beta = 0
    tau = 0
    taul = 0
    phi = beta1
    betan = beta1
    gmin = 0
    cs = -1
    sn = 0
    cr1 = -1
    sr1 = 0
    cr2 = -1
    sr2 = 0
    dltan = 0
    eplnn = 0
    gama = 0
    gamal = 0
    gamal2 = 0
    eta = 0
    etal = 0
    etal2 = 0
    vepln = 0
    veplnl = 0
    veplnl2 = 0
    ul3 = 0
    ul2 = 0
    ul = 0
    u = 0
    rnorm = betan
    xnorm = 0
    xl2norm = 0
    Anorm = 0
    Acond = 1
#    relres = rnorm / (beta1 + 1e-50)
    relres = 1
#    x = np.zeros((n,1))
    w = np.zeros((n,1))
    wl = np.zeros((n,1))
    
    #b = 0 --> x = 0 skip the main loop       
    while flag == flag0 and iters < maxit:
        #lanczos
        iters += 1
        betal = beta
        beta = betan
        v = r3/beta
        r3 = Ax(A, v)
        if shift == 0:
            pass
        else:
            r3 = r3 - shift*v
        
        if iters > 1:
            r3 = r3 - r1*beta/betal
        
        alfa = np.real(np.matmul(r3.T,v))
        r3 = r3 - r2*alfa/beta
        r1 = r2
        r2 = r3
        
        if noprecon:
            betan = norm(r3)
            if iters == 1:
                if betan == 0:
                    if alfa == 0:
                        flag = 0
                        print('WARNNING: flag = 0')
                        break
                    else:
                        flag = -1
                        print('WARNNING: flag = -1')
                        # Probbaly lost all the info, x=0 is true solution
                        x = b/alfa
                        break
        else:
            r3 = Precond(M, r2)
            betan = np.matmul(r2.T, r3)
            if betan > 0:
                betan = np.sqrt(betan)
            else:
                print('Error: "M" is indefinite or singular!')
                
        pnorm = np.sqrt(betal ** 2 + alfa ** 2 + betan ** 2)
        
        #previous left rotation Q_{k-1}
        dbar = dltan
        dlta = cs*dbar + sn*alfa
        epln = eplnn
        gbar = sn*dbar - cs*alfa
        eplnn = sn*betan
        dltan = -cs*betan
        dlta_QLP = dlta
        #current left plane rotation Q_k
        gamal3 = gamal2
        gamal2 = gamal
        gamal = gama
        cs, sn, gama = SymGivens(gbar, betan)
        gama_tmp = gama
        taul2 = taul
        taul = tau
        tau = cs*phi
        phi = sn*phi
        #previous right plane rotation P_{k-2,k}
        if iters > 2:
            veplnl2 = veplnl
            etal2 = etal
            etal = eta
            dlta_tmp = sr2*vepln - cr2*dlta
            veplnl = cr2*vepln + sr2*dlta
            dlta = dlta_tmp
            eta = sr2*gama
            gama = -cr2 *gama
        #current right plane rotation P{k-1,k}
        if iters > 1:
            cr1, sr1, gamal = SymGivens(gamal, dlta)
            vepln = sr1*gama
            gama = -cr1*gama
        
        ul4 = ul3
        ul3 = ul2
        if iters > 2:
            ul2 = (taul2 - etal2*ul4 - veplnl2*ul3)/gamal2
        if iters > 1:
            ul = (taul - etal*ul3 - veplnl *ul2)/gamal
        xnorm_tmp = np.sqrt(xl2norm**2 + ul2**2 + ul**2)
        if abs(gama) > np.finfo(np.double).tiny and xnorm_tmp < maxxnorm:
            u = (tau - eta*ul2 - vepln*ul)/gama
            if np.sqrt(xnorm_tmp**2 + u**2) > maxxnorm:
                u = 0
                flag = 3
                print('WARNNING: flag = 3')
        else:
            u = 0
            flag = 6
        xl2norm = np.sqrt(xl2norm**2 + ul2**2)
        xnorm = np.sqrt(xl2norm**2 + ul**2 + u**2)
        #update w&x
        #Minres
        if (Acond < TranCond) and flag != flag0 and QLPiter == 0:
            wl2 = wl
            wl = w
            w = (v - epln*wl2 - dlta_QLP*wl)/gama_tmp
            if xnorm < maxxnorm:
                x += tau*w
            else:
                flag = 3
                print('WARNNING: flag = 3')
        #Minres-QLP
        else:
            QLPiter += 1
            if QLPiter == 1:
                xl2 = np.zeros((n,1))
                if (iters > 1):  # construct w_{k-3}, w_{k-2}, w_{k-1}
                    if iters > 3:
                        wl2 = gamal3*wl2 + veplnl2*wl + etal*w
                    if iters > 2:
                        wl = gamal_QLP*wl + vepln_QLP*w
                    w = gama_QLP*w
                    xl2 = x - wl*ul_QLP - w*u_QLP
                    
            if iters == 1:
                wl2 = wl
                wl = v*sr1
                w = -v*cr1                
            elif iters == 2:
                wl2 = wl
                wl = w*cr1 + v*sr1
                w = w*sr1 - v*cr1
            else:
                wl2 = wl
                wl = w
                w = wl2*sr2 - v*cr2
                wl2 = wl2*cr2 +v*sr2
                v = wl*cr1 + w*sr1
                w = wl*sr1 - w*cr1
                wl = v
            xl2 = xl2 + wl2*ul2
            x = xl2 + wl*ul + w*u 
            
        #next right plane rotation P{k-1,k+1}
        gamal_tmp = gamal
        cr2, sr2, gamal = SymGivens(gamal, eplnn)
        #transfering from Minres to Minres-QLP
        gamal_QLP = gamal_tmp
        vepln_QLP = vepln
        gama_QLP = gama
        ul_QLP = ul
        u_QLP = u
        ## Estimate various norms
        abs_gama = abs(gama)
        Anorm = max([Anorm, pnorm, gamal, abs_gama])
        if iters == 1:
            gmin = gama
            gminl = gmin
        elif iters > 1:
            gminl2 = gminl
            gminl = gmin
            gmin = min([gminl2, gamal, abs_gama])
        Acondl = Acond
        Acond = Anorm / gmin
        rnorml = rnorm
        relresl = relres
        if flag != 9:
            rnorm = phi
        
        # inexactness of Newton-MR
        relres = 1 - Ab.T.dot(x)/beta1**2 # <g, Hp> < - ( 1 - rtol) ||g||^2
#        relres = rnorm / beta1 # relative residual ||Hp+g|| < rtol ||g||
        
        ## See if any of the stopping criteria are satisfied.
        epsx = Anorm * xnorm * np.finfo(float).eps
        if (flag == flag0) or (flag == 6):            
            if iters >= maxit:
                flag = 5 #exit before maxit
            if Acond >= Acondlim:
                flag = 4 #Huge Acond
                print('WARNNING: Acondlim exceeded!')
            if xnorm >= maxxnorm:
                flag = 3 #xnorm exceeded
                print('WARNNING: maxxnorm exceeded!')
            if epsx >= beta1:
                flag = 2 #x = eigenvector
            if relres <= rtol:
                flag = 1 #Trustful Solution
        if flag == 3 or flag == 4:
            print('WARNNING: possibly singular!')
            #possibly singular
            iters = iters - 1
            Acond = Acondl
            rnorm = rnorml
            relres = relresl  
            
    return x,relres,iters

def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax =A.dot(x)
    return Ax

def Precond(M, r):
    if callable(M):
        h = myCG(M, r, 10e-6, 1000)[0]
#        h = cg(M, r)
    else:
        h = np.matmul(M, r)
    return h

def SymGivens(a, b):    
    if b == 0:
        if a == 0:
            c = 1
        else:
            c = np.sign(a)
        s = 0
        r = abs(a)
    elif a == 0:
        c = 0
        s = np.sign(b)
        r = abs(b)
    elif abs(b) > abs(a):
        t = a / b
        s = np.sign(b) / np.sqrt(1 + t ** 2)
        c = s * t
        r = b / s
    else:
        t = b / a
        c = np.sign(a) / np.sqrt(1 + t ** 2)
        s = c * t
        r = a / c
    return c, s, r
