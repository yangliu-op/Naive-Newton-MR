import numpy as np
import os
import matplotlib.pyplot as plt

def pProfile(methods, Matrix, length=300, arg='log10', cut=None, 
             ylabel='Performance Profile', ylim=None):
    """
    A performance profile plot is the CDF of the performance of every 
    algorithms. Put your pProfile record (objVal/gradNorm/err) matrix to 
    pProfile folder and run this file to regenerate plots with proper scale.
    
    Reference:        
        E. D. Dolan and J. J. More. Benchmarking optimization software with 
        performance pro
files. Mathematical programming, 91(2): 201-213, 2002.
        
        N. Gould and J. Scott. A note on performance pro
les for benchmarking 
        software. ACM Transactions on Mathematical Software (TOMS), 43(2):15, 
        2016.
    INPUT:
        methods: list of all methods
        Matrix: a record matrix s.t., every row are the best (f/g/error) result 
            of all listed methods. e.g., 500 total_run of 5 different 
            optimisation methods will be a 500 x 5 matrix.
        length: number of nodes to compute on range x
        arg: scalling methods of x-axis
        cute: number of nodes to plot/display
        ylabel: label of y-axis
        ylim: plot/display [ylim, 1]
    OUTPUT:
        performance profile plots of the cooresponding (f/g/error) record 
        matrix.
    """
    
    M_hat = np.min(Matrix, axis=1)
    R = Matrix.T/M_hat
    R = R.T   
    x_high = np.max(R)
    if arg is None:
        x = np.linspace(1, x_high, length)
        myplt = plt.plot
        plt.xlabel('lambda')
    if arg is 'log2':
        x = np.logspace(0, np.log2(x_high), length)
        myplt = plt.semilogx
        plt.xlabel('log2(lambda)')
    if arg is 'log10':
        x = np.logspace(0, np.log10(x_high), length)
        myplt = plt.semilogx
        plt.xlabel('log(lambda)')
    n, d = Matrix.shape
    if cut != None:
        x = x[:cut]
        length = cut
    
    colors = ['k', 'm', 'g', 'y', 'r', 'b', 'c']
    linestyles = ['-', '-.', '--']
    
    if ylim != None:
        axes = plt.axes()
        axes.set_ylim(ylim, 1)
    
    for i in range(len(methods)):
        myMethod = methods[i]
        loop_i = int((i+1)/7)
        Ri = np.tile(R[:,i], (length, 1))
        xi = np.tile(x, (n,1))
        yi = np.sum((Ri.T <= xi), axis=0)/n
        myplt(x, yi, color=colors[i-7*loop_i], linestyle=linestyles[loop_i], 
              label = myMethod)
        
    plt.ylabel(ylabel)
    plt.legend()
    
    
def main():        
    with open('pProfile/methods.txt', 'r') as myfile:
        methods = myfile.read().split()
    F = np.loadtxt(open("pProfile/objVal.txt","rb"),delimiter=",",skiprows=0)
    G = np.loadtxt(open("pProfile/gradNorm.txt","rb"),delimiter=",",skiprows=0)
    Err = np.loadtxt(open("pProfile/err.txt","rb"),delimiter=",",skiprows=0)
    
    mypath = 'pProfile'
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
    figsz = (6,4)
    mydpi = 200
    length = 300
    
    fig1 = plt.figure(figsize=figsz)    
    pProfile(methods, F, length, arg='log10',cut=80) #80 150
    fig1.savefig(os.path.join(mypath, 'F'), dpi=mydpi)
    
    fig2 = plt.figure(figsize=figsz)    
    pProfile(methods, G, length, arg='log10', ylabel='GradientNorm')
    fig2.savefig(os.path.join(mypath, 'GradientNorm'), dpi=mydpi)
    
    fig3 = plt.figure(figsize=figsz)    
    pProfile(methods, Err,length, arg='log10',cut=40, ylabel='Error') #40 80
    fig3.savefig(os.path.join(mypath, 'Error'), dpi=mydpi)
    
if __name__ == '__main__':
    main()