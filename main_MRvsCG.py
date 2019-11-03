"""
Code for "Stability Analysis of Newton-MR Under Hessian Perturbations".
Authors: Yang Liu, Fred Roosta. ArXiv:1909.06224
Recover Figure 4 and 5. Stability comparison between full and sub-sampled 
variants of Newton-MR and Newton-CG using s = 0.1n, 0.05n, 0.01n.

To seperate MR & CG plots
1, copy & paste the recording .txt files to showFig folder,
2, run showFigure.py with different options, 
"""

from initialize import initialize

class learningRate():
    def __init__(self, value):
        self.value = value
        
class algPara():
    def __init__(self, value):
        self.value = value
        
#initialize methods
data = [
        'mnist',
#        'hapt',
        ]

prob = [
        'softmax', # realData
        ]

methods = [
        'Newton_MR',
        'Newton_CG', 
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
algPara.mainLoopMaxItrs = 15 #Set mainloop stops with Maximum Iterations
algPara.funcEvalMax = 1E6 #Set mainloop stops with Maximum Function Evaluations
algPara.innerSolverMaxItrs = 200 
algPara.lineSearchMaxItrs = 50
algPara.gradTol = 1e-10 #If norm(g)<gradTol, minFunc loop breaks
algPara.innerSolverTol = 0.01 #Inexactness of inner solver
algPara.beta = 1E-4 #Line search para
algPara.beta2 = 0.4 #Wolfe's condition for L-BFGS
algPara.show = True 
HProp_all = [0.1, 0.05, 0.01] # \leq 3 inputs
fullHessian = True #Run full Hessian cases for all methods
plotAll = True

## Initialize
initialize(data, methods, prob, regType, x0Type, HProp_all, 0, 
           algPara, learningRate, 1, fullHessian, plotAll)








