"""
Code for "Stability Analysis of Newton-MR Under Hessian Perturbations".
Authors: Yang Liu, Fred Roosta. ArXiv:1909.06224
Recover Figure 3, Comparison among sub-sampled Newton-MR and several first 
order methods.
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
        'cifar10',
        ]

prob = [
        'softmax', # realData
        ]

methods = [
        'Newton_MR',
        'Momentum',
        'Adagrad',
        'Adadelta',
        'RMSprop',
        'Adam',
        'SGD',
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
algPara.funcEvalMax = 1E3 #Set mainloop stops with Maximum Function Evaluations
algPara.innerSolverMaxItrs = 200 
algPara.lineSearchMaxItrs = 50
algPara.gradTol = 1e-10 #If norm(g)<gradTol, minFunc loop breaks
algPara.innerSolverTol = 0.01 #Inexactness of inner solver
algPara.beta = 1E-4 #Line search para
algPara.beta2 = 0.4 #Wolfe's condition for L-BFGS
algPara.show = True 
#algPara.show = False # print value in every iteration
    
#####     full gradient ###### suggest algPara.funcEvalMax = 1E4
#learningRate.Momentum = 5.976873506562755e-07
#learningRate.Adagrad = 6.1484967152219825e-06
#learningRate.Adadelta = 63.214676616915423E-4
#learningRate.RMSprop = 1.721313398398448e-06
#learningRate.Adam = 2.0549880802622692e-07
#learningRate.SGD = 3.4531770481261144e-07
#batchsize = 1 # proportion of mini-batch size
    
    
#####     0.05 batch ###### suggest algPara.funcEvalMax = 1E3
learningRate.Momentum = 2.966741304347826e-07
learningRate.Adagrad = 5.356066237949585e-06
learningRate.Adadelta = 0.00272575
learningRate.RMSprop = 6.738569565217392e-07
learningRate.Adam = 2.069303607054962e-07
learningRate.SGD = 3.219708160377358e-07
batchsize = 0.05 # proportion of mini-batch size

############################For 1 run plot#####################################
# comment off HProp if you don't want subsampling
HProp_all = [0.05] # \leq 3 inputs
fullHessian = False #Run full Hessian cases for all methods
    
## Initialize
initialize(data, methods, prob, regType, x0Type, HProp_all, batchsize, 
           algPara, learningRate, 1, fullHessian)









