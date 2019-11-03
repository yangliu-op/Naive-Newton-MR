"""
Optimisation algorithms, including 
Newton-MR and its subsampling variants,  
Newton-CG and its subsampling variants,
L-BFGS
Gauss-Newton
(mini-batch) SGD with/without Momentum
(mini-batch) Adagrad
(mini-batch) Adadelta
(mini-batch) RMSprop
(mini-batch) Adam

Termination condition: norm(gradient) < gradtol. 
Otherwise either reach maximum iterations or maximum oracle calls
""" 

import numpy as np
from numpy.linalg import norm, pinv
from lbfgs import lbfgs
from myCG import myCG
from time import time
from linesearch import linesearch, linesearchgrad, linesearchzoom
from MinresQLP import MinresQLP
from GMM import gmmtest
     
def myPrint(fxi, gxi, fei, iters, tmi, 
            alphai=0, iterLS=0, iterSolver=0, rel_res=0):
    """
    A print function for every iteration.
    """
    if iters%10 == 0:
        prthead1 = '  iters  iterSolver   iterLS   Time       f          ||g||'
        prthead2 = '       alphai      Prop      Relres'        
        prt = prthead1 + prthead2
        print(prt)
    prt1 = '%8g %8g' % (iters, iterSolver)
    prt2 = '%8s %8.2f' % (iterLS, tmi)
    prt3 = ' %8.2e     %8.2e ' % (fxi, gxi)
    prt4 = '   %8.2e %8g ' % (alphai, fei)
    prt5 = '  %8.2e' % (rel_res)
    print(prt1, prt2, prt3, prt4, prt5)  
    
def Newton_MR(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, innerSolverTol,
              innerSolverMaxItrs=200, lineSearchMaxItrs=50, gradTol=1e-10,
              beta=1e-4, theta=None, show=True):
    """
    NewtonMR algorithms
    INPUT:
        obj: function handle of objective function, gradient and 
            Hessian-vector product
        x0: starting point
        HProp: subsampled(perturbed) Hessian proportion
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        innerSolverTol: inexactness tolerance of Inner-solver MINRES-QLP
        lineSearchMaxItrs: maximum iteration of line search
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        beta: parameter of Armijo line-search
        theta: True weight+mean of GMM problems
        show: print result for every iteration
    
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, step-size, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk, Hk = obj(x)        
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi    
    alphai = 1
    alpha = alphai    
    tmi = 0
    tm = tmi    
    rel_res = 1
    iterSolver = 0
    iterLS = 0
    
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True:
        if show:
            myPrint(fxi, gxi, fei, iters, tmi, alphai, iterLS, iterSolver, 
                    rel_res)
            
        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break

        t0 = time()
        if innerSolverTol == 'exact': # for fraction problem
            p = -pinv(Hk).dot(gk)
            rel_res = norm(Hk.dot(p) + gk)/norm(gk)
        else:
            p, rel_res, iterSolver = MinresQLP(Hk, -gk, rtol=innerSolverTol,
                                               maxit=innerSolverMaxItrs)
        x, alphai, iterLS = linesearchgrad(obj, Hk, x, p, 
                                           lineSearchMaxItrs, 1, c1=beta)
        
        if HProp == None:
            fei += 2*iterSolver + 2*(1 + iterLS)
        else:
            fei += 2*iterSolver*HProp + 2*(1 + iterLS)    
            
        iters += 1  
        fk, gk, Hk = obj(x)
        fxi = fk
        gxi = norm(gk)              
        tmi += time()-t0
                
        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)
            
        fe = np.append(fe, fei)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)  
        alpha = np.append(alpha, alphai)    
        tm = np.append(tm, tmi)             
        
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, alpha, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm, alpha]
    return x, record
        
           
def L_BFGS(obj, x0, mainLoopMaxItrs, funcEvalMax, lineSearchMaxItrs=50, 
           gradTol=1e-10, beta=1e-4, beta2=0.4, theta=None, show=True):
    """
    L_BFGS algorithms
    INPUT:
        obj: a function handle of objective function, gradient 
            (and Hessian-vector product)
        x0: starting point
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        lineSearchMaxItrs: maximum iteration of line search
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        beta: parameter of Armijo line-search
        beta2: parameter of Wolfe curvature condition (line-search)
        theta: True weight+mean of GMM problems
        show: print result for every iteration
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk = obj(x, 'fg')    
    l = len(gk)   
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi    
    alphai = 1
    alpha = alphai    
    tmi = 0
    tm = tmi    
    L = 20
    iterLS = 0
    
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True:
        if show:
            myPrint(fxi, gxi, fei, iters, tmi, alphai, iterLS)
     
        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break 
        
        t0 = time()  
        if iters == 0:
            p = -gk
            Yy = np.array([]).reshape(l,0)
            S = np.array([]).reshape(l,0)
        else:
            s = alphai_prev * p_prev
            y = gk - g_prev
            
            if s.shape[1] >= L:
                S = np.append(S[:,1:], s, axis=1)
                Yy = np.append(Yy[:,1:], y, axis=1)
            else:
                S = np.append(S, s, axis=1)
                Yy = np.append(Yy, y, axis=1)   
        p = -lbfgs(gk, S, Yy)
        
        #Strong wolfe's condition with zoom
        alphai, iterLS = linesearchzoom(obj, x, p, lineSearchMaxItrs, c1=beta, c2=beta2)
        x = x + alphai*p
        
        g_prev = gk
        p_prev = p
        alphai_prev = alphai            
        tmi += time()-t0            

        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)
        
        fk, gk = obj(x, 'fg')
        fxi = fk
        gxi = norm(gk)
        iters += 1
        fei += 2 + 2*iterLS
        fe = np.append(fe, fei)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)
        alpha = np.append(alpha, alphai)
        tm = np.append(tm, tmi)
            
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, alpha, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm, alpha]
    return x, record

        
def Newton_CG(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, innerSolverTol,
              innerSolverMaxItrs=200, lineSearchMaxItrs=50, gradTol=1e-10,
              beta=1e-4, theta=None, show=True):
    """
    Newton_CG algorithms
    INPUT:
        obj: function handle of objective function, gradient and 
            Hessian-vector product
        x0: starting point
        HProp: subsampled(perturbed) Hessian proportion
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        innerSolverTol: inexactness tolerance of Inner-solver MINRES-QLP
        lineSearchMaxItrs: maximum iteration of line search
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        beta: parameter of Armijo line-search
        theta: True weight+mean of GMM problems
        show: print result for every iteration
    
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, step-size, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk, Hk = obj(x)        
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi    
    alphai = 1
    alpha = alphai    
    tmi = 0
    tm = tmi    
    rel_res=1
    iterSolver = 0
    iterLS = 0
    
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True:
        if show:
            myPrint(fxi, gxi, fei, iters, tmi, alphai, iterLS, iterSolver, 
                    rel_res)
        
        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break 

        t0 = time() 
        p, rel_res, iterSolver = myCG(Hk, -gk, innerSolverTol, 
                                      innerSolverMaxItrs)
        x, alphai, iterLS = linesearch(obj, gk, x, p, 
                                       lineSearchMaxItrs, 1, c1=beta)
        
        if HProp == None:
            fei += 2*iterSolver + 2 + iterLS
        else:
            fei += 2*iterSolver*HProp + 2 + iterLS                  

        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)

        iters += 1
        fk, gk, Hk = obj(x)
        gxi = norm(gk) 
        
        tmi += time()-t0    
        fxi = fk
        fe = np.append(fe, fei)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)
        alpha = np.append(alpha, alphai)
        tm = np.append(tm, tmi)
        
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, alpha, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm, alpha]
    return x, record


def Gauss_Newton(obj, x0, HProp, mainLoopMaxItrs, funcEvalMax, innerSolverTol,
              innerSolverMaxItrs=200, lineSearchMaxItrs=50, gradTol=1e-10,
              beta=1e-4, theta=None, show=True):
    """
    Gauss_Newton algorithms
    INPUT:
        obj: function handle of objective function, gradient and 
            Gauss_Newton_matrix-vector product
        x0: starting point
        HProp: subsampled(perturbed) Hessian proportion
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        innerSolverTol: inexactness tolerance of Inner-solver MINRES-QLP
        lineSearchMaxItrs: maximum iteration of line search
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        beta: parameter of Armijo line-search
        theta: True weight+mean of GMM problems
        show: print result for every iteration
    
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, step-size, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk, Hk = obj(x, 'gn')        
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi    
    alphai = 1
    alpha = alphai    
    tmi = 0
    tm = tmi    
    rel_res = 1
    iterSolver = 0
    iterLS = 0
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True:
        if show:
            myPrint(fxi, gxi, fei, iters, tmi, alphai, iterLS, iterSolver, 
                    rel_res)
              
        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break 

        t0 = time() 
        p, rel_res, iterSolver = myCG(Hk, -gk, innerSolverTol, innerSolverMaxItrs)
        x, alphai, iterLS = linesearch(obj, gk, x, p, lineSearchMaxItrs, 1, c1=beta)
        
        if HProp == None:
            fei += 2*iterSolver + 2 + iterLS
        else:
            fei += 2*iterSolver*HProp + 2 + iterLS        
            
        iters += 1
        fk, gk, Hk = obj(x, 'gn')
        fxi = fk
        gxi = norm(gk)
        tmi += time()-t0

        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)

        fe = np.append(fe, fei)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)
        alpha = np.append(alpha, alphai)
        tm = np.append(tm, tmi)
                     
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, alpha, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm, alpha]
    return x, record


def MomentumSGD(obj, x0, obj_mini_g, batchsize, mainLoopMaxItrs, funcEvalMax, 
                learningRate, gradTol=1e-10, theta=None, show=True):
    """
    Momentum SGD algorithms
    INPUT:
        obj: a function handle of objective function, gradient 
            (and Hessian-vector product)
        x0: starting point
        obj_mini_g: a function handle of the gradient of minibatch
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        learningRate: learning rate
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        theta: True weight+mean of GMM problems
        show: print result for every iteration
    
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk = obj(x, 'fg')        
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi    
    tmi = 0
    tm = tmi  
    gamma = 0.9
        
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True: 
        if show:
            myPrint(fxi, gxi, fei, iters, tmi)
            
        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break 

        t0 = time()   
        if obj_mini_g != None:
            gk_mini = obj_mini_g(x)
        else:
            gk_mini = gk
            
        if iters == 0:
            p_prev = np.copy(x0)
            
        p = gamma*p_prev + learningRate*gk_mini
        x = x - p
        p_prev = p
        fei += 2*batchsize
           
        tmi += time() - t0   

        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)

        iters += 1
        fk, gk = obj(x, 'fg')
        fxi = fk
        gxi = norm(gk)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)   
        tm = np.append(tm, tmi)   
        fe = np.append(fe, fei)
              
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm]
        
    return x, record

            
def Adagrad(obj, x0, obj_mini_g, batchsize, mainLoopMaxItrs, funcEvalMax, 
            learningRate, gradTol=1e-10, theta=None, show=True, smooth=1e-8):
    """
    Adagrad algorithms
    INPUT:
        obj: a function handle of objective function, gradient 
            (and Hessian-vector product)
        x0: starting point
        obj_mini_g: a function handle of the gradient of minibatch
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        learningRate: learning rate
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        theta: True weight+mean of GMM problems
        show: print result for every iteration
        smooth: smooth parameter/avoid zero
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk = obj(x, 'fg')        
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi    
    tmi = 0
    tm = tmi  
        
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True: 
        if show:
            myPrint(fxi, gxi, fei, iters, tmi)
          
        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break 

        t0 = time()   
        if obj_mini_g != None:
            gk_mini = obj_mini_g(x)
        else:
            gk_mini = gk   
            
        if iters == 0:
            G = np.copy(x0)            
        G = G + gk_mini*gk_mini
        step = learningRate*gk_mini/np.sqrt(G + smooth)
        x = x - step
        fei += 2*batchsize
        
        tmi += time() - t0   

        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)

        iters += 1
        fk, gk = obj(x, 'fg')
        fxi = fk
        gxi = norm(gk)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)   
        tm = np.append(tm, tmi)   
        fe = np.append(fe, fei)
                
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm]
        
    return x, record


def Adadelta(obj, x0, obj_mini_g, batchsize, mainLoopMaxItrs, funcEvalMax, 
             learningRate, gradTol=1e-10, theta=None, show=True, smooth=1e-8):
    """
    Adadelta algorithms
    INPUT:
        obj: a function handle of objective function, gradient 
            (and Hessian-vector product)
        x0: starting point
        obj_mini_g: a function handle of the gradient of minibatch
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        learningRate: learning rate
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        theta: True weight+mean of GMM problems
        show: print result for every iteration
        smooth: smooth parameter/avoid zero
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk = obj(x, 'fg')        
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi    
    tmi = 0
    tm = tmi  
    gamma1 = 0.9
    gamma2 = 0.9
        
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True: 
        if show:
            myPrint(fxi, gxi, fei, iters, tmi)
      
        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break 

        t0 = time()   
        if obj_mini_g != None:
            gk_mini = obj_mini_g(x)
        else:
            gk_mini = gk   
                
        if iters == 0:
            Egrad2 = np.copy(x0)
            p2 = np.copy(x0)
        Egrad2 = gamma1*Egrad2 + (1 - gamma1)*gk_mini*gk_mini
        RMSUpdate = np.sqrt(p2 + smooth)
        RMSGrad = np.sqrt(Egrad2 + smooth)
        p = -(RMSUpdate/RMSGrad)*gk_mini
        x = x + learningRate*p
        p2 = gamma2*p2 + (1 - gamma2)*p*p
        fei += 2*batchsize
                    
        tmi += time() - t0   

        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)

        iters += 1
        fk, gk = obj(x, 'fg')
        fxi = fk
        gxi = norm(gk)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)   
        tm = np.append(tm, tmi)   
        fe = np.append(fe, fei)
                
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm]
    return x, record


def RMSprop(obj, x0, obj_mini_g, batchsize, mainLoopMaxItrs, funcEvalMax, 
            learningRate, gradTol=1e-10, theta=None, show=True, smooth=1e-8):
    """
    RMSprop algorithms
    INPUT:
        obj: a function handle of objective function, gradient 
            (and Hessian-vector product)
        x0: starting point
        obj_mini_g: a function handle of the gradient of minibatch
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        learningRate: learning rate
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        theta: True weight+mean of GMM problems
        show: print result for every iteration
        smooth: smooth parameter/avoid zero
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk = obj(x, 'fg')        
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi      
    tmi = 0
    tm = tmi  
    gamma = 0.9
        
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True: 
        if show:
            myPrint(fxi, gxi, fei, iters, tmi)
        
        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break 

        t0 = time()   
        if obj_mini_g != None:
            gk_mini = obj_mini_g(x)
        else:
            gk_mini = gk   
                
        if iters == 0:
            Egrad2 = np.copy(x0)
        Egrad2 = gamma*Egrad2 + (1 - gamma)*gk_mini*gk_mini
        step = learningRate*gk_mini/np.sqrt(Egrad2 + smooth)
        x = x - step
        fei += 2*batchsize            
        tmi += time() - t0   

        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)

        iters += 1
        fk, gk = obj(x, 'fg')
        fxi = fk
        gxi = norm(gk)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)   
        tm = np.append(tm, tmi)   
        fe = np.append(fe, fei)
                
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm]
    return x, record


def Adam(obj, x0, obj_mini_g, batchsize, mainLoopMaxItrs, funcEvalMax, 
         learningRate, gradTol=1e-10, theta=None, show=True, smooth=1e-8):
    """
    Adam algorithms
    INPUT:
        obj: a function handle of objective function, gradient 
            (and Hessian-vector product)
        x0: starting point
        obj_mini_g: a function handle of the gradient of minibatch
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        learningRate: learning rate
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        theta: True weight+mean of GMM problems
        show: print result for every iteration
        smooth: smooth parameter/avoid zero
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk = obj(x, 'fg')        
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi     
    tmi = 0
    tm = tmi  
    mPara = 0.9
    vPara = 0.999
        
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True: 
        if show:
            myPrint(fxi, gxi, fei, iters, tmi)
          
        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break 

        t0 = time()   
        if obj_mini_g != None:
            gk_mini = obj_mini_g(x)
        else:
            gk_mini = gk   
                       
        if iters == 0:
            m = np.copy(x0)
            v = np.copy(x0)
            m_prev = m
            v_prev = v
        m = mPara*m_prev + (1 - mPara)*gk_mini
        v = vPara*v_prev + (1 - vPara)*gk_mini*gk_mini
        m_prev = m
        v_prev = v
        m_hat = m/(1 - mPara)
        v_hat = v/(1 - mPara)
        x = x - learningRate*(m_hat/(np.sqrt(v_hat)+smooth))
        fei += 2*batchsize
        
        tmi += time() - t0   

        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)

        iters += 1
        fk, gk = obj(x, 'fg')
        fxi = fk
        gxi = norm(gk)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)   
        tm = np.append(tm, tmi)   
        fe = np.append(fe, fei)
                    
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm]
    return x, record

    
def SGD(obj, x0, obj_mini_g, batchsize, mainLoopMaxItrs, funcEvalMax, 
        learningRate, gradTol=1e-10, theta=None, show=True):
    """
    SGD algorithms
    INPUT:
        obj: a function handle of objective function, gradient 
            (and Hessian-vector product)
        x0: starting point
        obj_mini_g: a function handle of the gradient of minibatch
        mainLoopMaxItrs: maximum iteration of the main algorithms
        funcEvalMax: maximum oracle calls (function evaluations)
        learningRate: learning rate
        gradTol: stopping condition para s.t., norm(gk) <= Tol
        theta: True weight+mean of GMM problems
        show: print result for every iteration
    OUTPUTS:
        x: best x solution
        record: a record matrix with columns vectors:
            [fx, gx, oracle-calls, time, (GMM estimation error)]
    """
    iters = 0
    fei = 0
    fe = fei    
    x = np.copy(x0)
    fk, gk = obj(x, 'fg')        
    fxi = fk
    fx = fxi
    gxi = norm(gk)
    gx = gxi    
    tmi = 0
    tm = tmi  
        
    if theta is not None:
        gmm_norm = gmmtest(x, theta)
    
    while True: 
        if show:
            myPrint(fxi, gxi, fei, iters, tmi)

        if gxi < gradTol or iters >= mainLoopMaxItrs:
            break 
        if gxi < gradTol or fei >= funcEvalMax:  
            break 

        t0 = time()   
        if obj_mini_g != None:
            gk_mini = obj_mini_g(x)
        else:
            gk_mini = gk   
            
        x = x - learningRate*gk_mini
        fei += 2*batchsize
        
        tmi += time() - t0   

        if theta is not None:
            gmm_normi = gmmtest(x, theta)
            gmm_norm = np.append(gmm_norm, gmm_normi)

        iters += 1
        fk, gk = obj(x, 'fg')
        fxi = fk
        gxi = norm(gk)
        fx = np.append(fx, fxi)
        gx = np.append(gx, gxi)   
        tm = np.append(tm, tmi)   
        fe = np.append(fe, fei)
                
    if theta is not None:
        record = np.c_[fx, gx, fe, tm, gmm_norm]
    else:
        record = np.c_[fx, gx, fe, tm]
    return x, record
    
    
    
