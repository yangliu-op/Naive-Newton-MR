"""
Code for "Stability Analysis of Newton-MR Under Hessian Perturbations".
Authors: Yang Liu, Fred Roosta. ArXiv:1909.06224
One can play around with different methods here. If one want to recover figures 
of the above paper...
Run main_Fraction.py   to regenerate Figure 1.
Run main_Softmax2nd.py to regenerate Figure 2.
Run main_Softmax1st.py to regenerate Figure 3.
Run main_MRvsCG.py     to regenerate Figure 4 and 5.
Run main_GMM2nd.py     to regenerate Figure 6.
Run main_GMM1st.py     to regenerate Figure 7 and 8.

Reference:
    "Stability Analysis of Newton-MR Under Hessian Perturbations".
Authors: Yang Liu, Fred Roosta. ArXiv:1909.06224
    
    "Newton-MR: Newton’s Method Without Smoothness or Convexity"
Authors: Fred Roosta, Yang Liu, Peng Xu, Michael W. Mahoney. ArXiv: 1810.00303
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
#        'cifar10',
#        'hapt',
        ]

prob = [
        'softmax', # realData
#        'nls',
##########################################################
#        'gmm', # Always generate synthetic Data
#        'fraction', # Always generate synthetic Data
        ]

methods = [
        'Newton_MR',
##########################################################
        'L_BFGS',
        'Newton_CG', # not for least_square
#        'Gauss_Newton', # not for softmax
##########################################################
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
algPara.mainLoopMaxItrs = 100 #Set mainloop stops with Maximum Iterations
algPara.funcEvalMax = 100 #Set mainloop stops with Maximum Function Evaluations
algPara.innerSolverMaxItrs = 200 
algPara.lineSearchMaxItrs = 50
algPara.gradTol = 1e-10 #If norm(g)<gradTol, minFunc loop breaks
algPara.innerSolverTol = 0.01 #Inexactness of inner solver
#algPara.innerSolverTol = 'exact' #Inexactness of inner solver
algPara.beta = 1E-4 #Line search para
algPara.beta2 = 0.4 #Wolfe's condition for L-BFGS
algPara.show = True 
lamda = 1 #regularizer

# for softmax MNIST
learningRate.Momentum = 1E-5
learningRate.Adagrad = 1E-4
learningRate.Adadelta = 1E-1
learningRate.RMSprop = 1E-5
learningRate.Adam = 1E-5
learningRate.SGD = 1E-5

# for softmax Cifar10
#learningRate.Momentum = 1E-7
#learningRate.Adagrad = 1E-6
#learningRate.Adadelta = 1E-4
#learningRate.RMSprop = 1E-7 #1E-6
#learningRate.Adam = 1E-8 #1E-7
#learningRate.SGD = 1E-7

# for gmm
#learningRate.Momentum = 1E-8
#learningRate.Adagrad = 1E-1
#learningRate.Adadelta = 10
#learningRate.RMSprop = 1E-2
#learningRate.Adam = 1E-2
#learningRate.SGD = 1E-8
##########################For 1 run plot#####################################
batchsize = 0.05 # proportion of mini-batch size

##########################For 1 run plot#####################################
# comment off HProp if you don't want subsampling
#HProp_all = [0.1, 0.05, 0.01] # \leq 3 inputs
HProp_all = [0.05]
#HProp_all = [1E-2, 1E-5, 1E-13] # only for Fraction problems

fullHessian = True #Run full Hessian cases for all methods
plotAll = False

#########################GMM performance profile plot########################
# multiple run performance profile plot
total_run = 1

## Initialize
initialize(data, methods, prob, regType, x0Type, HProp_all, batchsize, 
           algPara, learningRate, total_run, fullHessian, plotAll, lamda)









