"""
Code for "Stability Analysis of Newton-MR Under Hessian Perturbations".
Authors: Yang Liu, Fred Roosta. ArXiv:1909.06224
To recover Figure 1: Performance of Newton-MR under an unstable perturbation 
for \epsilon = 10^{-2}\gamma, 10^{-5}\gamma,and 10^{−13}\gamma, where
\gamma denote the smallest singular value of full Hessian.
We may randomly generate such Hessian with very small spectral norm 
and can be regarded as 0 rank.

Reminder: Hessian may be generated with 0 ranks.
"""

from initialize import initialize

class algPara():
    def __init__(self, value):
        self.value = value
                
prob = [
        'fraction',
        ]

methods = [
        'Newton_MR',
        ]

regType = [
        'None',
        ] 

#initial point
x0Type = [
#        'randn',
        'rand',
#        'ones'
        ]

#initialize parameters
algPara.mainLoopMaxItrs = 15 #Set mainloop stops with Maximum Iterations
algPara.funcEvalMax = 1E5 #Set mainloop stops with Maximum Function Evaluations
algPara.innerSolverMaxItrs = 200 
algPara.lineSearchMaxItrs = 50
algPara.gradTol = 1e-10 #If norm(g)<gradTol, minFunc loop breaks
algPara.innerSolverTol = 'exact' #Inexactness of inner solver
algPara.beta = 1E-4 #Line search para
algPara.beta2 = 0.4 #Wolfe's condition for L-BFGS
algPara.show = True 
HProp_all = [1E-2, 1E-5, 1E-13] # only for Fraction problems
fullHessian = True #Run full Hessian cases for all methods
plotAll = True
    
initialize(None, methods, prob, regType, x0Type, HProp_all, 0, 
           algPara, None, 1, fullHessian, plotAll)









