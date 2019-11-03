import matplotlib.pyplot as plt
import os
import numpy as np

def showFigure(methods_all, record_all, prob, mypath, plotAll=False):
    """
    Plots generator.
    Input: 
        methods_all: a list contains all methods
        record_all: a list contains all record matrix of listed methods, 
        s.t., [fx, norm(gx), oracle calls, time, stepsize, GMM error]
        prob: name of problem
        mypath: directory path for saving plots
    OUTPUT:
        Oracle calls vs. F
        Oracle calls vs. Gradient norm
        Oracle calls vs. Estimation error(GMM) or Iteration vs. Time
        Iteration vs. F
        Iteration vs. Gradient norm
        Iteration vs. Step Size
    """
    fsize = 12
    myplt = plt.loglog
        
    colors = ['k', 'm', 'g', 'y', 'r', 'b', 'c']
    linestyles = ['-', '-.', '--']
    
    figsz = (6,4)
    mydpi = 200
    
    if prob != 'fraction':
        fig1 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = record_all[i]
            loop_i = int((i+1)/7)
            myplt(record[:,2]+1, record[:,0], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
        plt.xlabel('Oracle calls', fontsize=fsize)
        plt.ylabel('F', fontsize=fsize)
        plt.legend()
        fig1.savefig(os.path.join(mypath, 'F'), dpi=mydpi)
        
        
        fig2 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = record_all[i]
            loop_i = int((i+1)/7)
            myplt(record[:,2]+1, record[:,1], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
        plt.xlabel('Oracle calls', fontsize=fsize)
        plt.ylabel('Gradient norm', fontsize=fsize)
        plt.legend()
        fig2.savefig(os.path.join(mypath, 'Gradient norm'), dpi=mydpi)
        
        fig3 = plt.figure(figsize=figsz)
        if prob == 'gmm':        
            for i in range(len(methods_all)):
                record = record_all[i]
                loop_i = int((i+1)/7)
                myplt(record[:,2]+1, record[:,-1], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
            plt.xlabel('Oracle calls', fontsize=fsize)
            plt.ylabel('Estimation error', fontsize=fsize)
            plt.legend()
            fig3.savefig(os.path.join(mypath, 'Error'), dpi=mydpi)
        else:
            for i in range(len(methods_all)):
                record = record_all[i]
                loop_i = int((i+1)/7)
                myplt(record[:,3]+1, record[:,0], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
            plt.xlabel('Time', fontsize=fsize)
            plt.ylabel('F', fontsize=fsize)
            plt.legend()
            fig3.savefig(os.path.join(mypath, 'Time'), dpi=mydpi)
        
    if plotAll == True:
        fig4 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = record_all[i]
            loop_i = int((i+1)/7)
            myplt(range(1,len(record[:,0])+1), record[:,0], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('F', fontsize=fsize)
        plt.legend()
        fig4.savefig(os.path.join(mypath, 'Iteration_F'), dpi=mydpi)
        
        fig5 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = record_all[i]
            loop_i = int((i+1)/7)
            myplt(range(1,len(record[:,1])+1), record[:,1], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('Gradient norm', fontsize=fsize)
        plt.legend()
        fig5.savefig(os.path.join(mypath, 'Iteration_G'), dpi=mydpi)
        
        fig6 = plt.figure(figsize=figsz)
        for i in range(len(methods_all)):
            record = record_all[i]
            loop_i = int((i+1)/7)
            if record.shape[1] < 5:
                raise ValueError('First Order methods do not have stepsize!')
            myplt(range(1,len(record[:,4])+1), record[:,4], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('Step Size', fontsize=fsize)
        plt.legend()
        fig6.savefig(os.path.join(mypath, 'Iteration_alpha'), dpi=mydpi)
        

def showFigure_CGvsMR(methods_all, record_all, mypath):
    """
    Plots generator.
    Input: 
        methods_all: a list contains all methods
        record_all: a list contains all record matrix of listed methods, 
        s.t., [fx, norm(gx), oracle calls, time, stepsize, GMM error]
        mypath: directory path for saving plots
    OUTPUT:
        Iteration vs. F, Newton CG
        Iteration vs. Gradient norm, Newton CG
        Iteration vs. F, Newton MR
        Iteration vs. Gradient norm, Newton MR
    """
    fsize = 12
    myplt = plt.semilogx
    myplt2 = plt.semilogx
        
    colors = ['k', 'm', 'g', 'y', 'r', 'b', 'c']
    linestyles = ['-', '-.', '--']
    
    figsz = (6,4)
    mydpi = 200
    
    fig1 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        if i < 5 and i != 1: #Newton-CG            
            record = record_all[i]
            loop_i = int((i+1)/7)
            myplt(range(1,len(record[:,0])+1), record[:,0], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('F', fontsize=fsize)
        plt.legend()
        fig1.savefig(os.path.join(mypath, 'Iteration_F'), dpi=mydpi)
        
    fig2 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        if i >= 5 or i == 1: #Newton-MR           
            record = record_all[i]
            loop_i = int((i+1)/7)
            myplt(range(1,len(record[:,0])+1), record[:,0], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('F', fontsize=fsize)
        plt.legend()
        fig2.savefig(os.path.join(mypath, 'Iteration_F'), dpi=mydpi)        
    
    fig3 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        if i < 5 and i != 1: #Newton-CG          
            record = record_all[i]
            loop_i = int((i+1)/7)
            myplt2(range(1,len(record[:,1])+1), record[:,1], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('Gradient norm', fontsize=fsize)
        plt.legend()
        fig3.savefig(os.path.join(mypath, 'Iteration_G'), dpi=mydpi)
    
    fig4 = plt.figure(figsize=figsz)
    for i in range(len(methods_all)):
        if i >= 5 or i == 1: #Newton-MR          
            record = record_all[i]
            loop_i = int((i+1)/7)
            myplt2(range(1,len(record[:,1])+1), record[:,1], color=colors[i-7*loop_i], linestyle=linestyles[loop_i], label = methods_all[i])
        plt.xlabel('Iteration', fontsize=fsize)
        plt.ylabel('Gradient norm', fontsize=fsize)
        plt.legend()
        fig4.savefig(os.path.join(mypath, 'Iteration_G'), dpi=mydpi)
        
def main():
    methods_all = []
    record_all = []
    for method in os.listdir('showFig'): #only contains txt files
        methods_all.append(method.rsplit('.', 1)[0])
        record = np.loadtxt(open('showFig/'+method,"rb"),delimiter=",",skiprows=0)
        record_all.append(record)
    mypath = 'showFig_plots'
    if not os.path.isdir(mypath):
       os.makedirs(mypath)
# =============================================================================
    """
    regenerate any 1 total run plot via txt record matrix in showFig folder.
    Note that directory only contains txt files.
    For performace profile plots, see pProfile.py
    """
#    showFigure(methods_all, record_all, None, mypath)
# =============================================================================
    """
    regenerate/seperate MR vs. CG plot via txt history matrix in showFig folder.
    """
    showFigure_CGvsMR(methods_all, record_all, mypath)
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()