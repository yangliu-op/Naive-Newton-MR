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
#        'mnist',
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
algPara.funcEvalMax = 100 #Set mainloop stops with Maximum Function Evaluations
algPara.innerSolverMaxItrs = 200 
algPara.lineSearchMaxItrs = 50
algPara.gradTol = 1e-10 #If norm(g)<gradTol, minFunc loop breaks
algPara.innerSolverTol = 0.01 #Inexactness of inner solver
algPara.beta = 1E-4 #Line search para
algPara.beta2 = 0.4 #Wolfe's condition for L-BFGS
algPara.show = True 

if data[0] == 'mnist':
    learningRate.Momentum = 1E-5
    learningRate.Adagrad = 1E-4
    learningRate.Adadelta = 1E-1
    learningRate.RMSprop = 1E-5
    learningRate.Adam = 1E-5
    learningRate.SGD = 1E-5
    
if data[0] == 'cifar10':
    learningRate.Momentum = 1E-8 #1E-7
    learningRate.Adagrad = 1E-6
    learningRate.Adadelta = 1E-4
    learningRate.RMSprop = 1E-7 #1E-6
    learningRate.Adam = 1E-8 #1E-7
    learningRate.SGD = 1E-7

batchsize = 0.05 # proportion of mini-batch size
############################For 1 run plot#####################################
# comment off HProp if you don't want subsampling
HProp_all = [0.05] # \leq 3 inputs
fullHessian = False #Run full Hessian cases for all methods
    
## Initialize
initialize(data, methods, prob, regType, x0Type, HProp_all, batchsize, 
           algPara, learningRate, 1, fullHessian)









