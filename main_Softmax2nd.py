"""
Code for "Stability Analysis of Newton-MR Under Hessian Perturbations".
Authors: Yang Liu, Fred Roosta. ArXiv:1909.06224
Recover Figure 2. Comparison among Newton-type method for Softmax regression.
"""

from initialize import initialize
        
class algPara():
    def __init__(self, value):
        self.value = value
        
#initialize methods
data = [
        'mnist',
#        'cifar10',
        ]

prob = [
        'softmax', # realData
        ]

methods = [
        'Newton_MR',
        'L_BFGS',
        'Newton_CG', # not for least_square
        ]

regType = [
        'None',
#        'Convex',
#        'Nonconvex',
        ] 

#initial point
x0Type = [
#        'randn',
#        'rand',
#        'ones'
        'zeros', # note: 0 is a saddle point for fraction problems
        ]

#initialize parameters
algPara.mainLoopMaxItrs = 1E5 #Set mainloop stops with Maximum Iterations
algPara.funcEvalMax = 1000 #Set mainloop stops with Maximum Function Evaluations
algPara.innerSolverMaxItrs = 200 
algPara.lineSearchMaxItrs = 50
algPara.gradTol = 1e-10 #If norm(g)<gradTol, minFunc loop breaks
algPara.innerSolverTol = 0.01 #Inexactness of inner solver
algPara.beta = 1E-4 #Line search para
algPara.beta2 = 0.4 #Wolfe's condition for L-BFGS
#algPara.show = False 
algPara.show = True 
HProp_all = [0.1, 0.05, 0.01] # \leq 3 inputs
fullHessian = True #Run full Hessian cases for all methods
    
## Initialize
initialize(data, methods, prob, regType, x0Type, HProp_all, 0, 
           algPara, None, 1, fullHessian)









