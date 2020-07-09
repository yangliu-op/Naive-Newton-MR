# Newton-MR
Newton-MR is a second-order Newton-type optimisation algorithms. This python codes is for the paper [Stability Analysis of Newton-MR Under Hessian Perturbations](https://arxiv.org/abs/1909.06224).

    Authors: 
        Yang Liu, 
        School of Mathematics and Physics, the University of Queensland, Australia.
        contact: yang.liu2(AT)uq.edu.au
        
        Fred Roosta,  
        School of Mathematics and Physics, the University of Queensland, Australia, 
            and International Computer Science Institute, Berkeley, USA.
        contact: fred.roosta(AT)uq.edu.au

All optimisation algorithms can be found at optim_algo.py

One can play around with different methods in main.py. 
To regenerate figures of the above paper...

    Run main_Fraction.py   to regenerate Figure 2.
    Run main_Softmax2nd.py to regenerate Figure 2.
    Run main_Softmax1st.py to regenerate Figure 4 and 5.
    Run main_MRvsCG.py     to regenerate Figure 6 and 7.
    Run main_GMM2nd.py     to regenerate Figure 8.
    Run main_GMM1st.py     to regenerate Figure 9 and 10.

Reference:

    "Convergence of Newton-MR under Inexact Hessian Information".
    Authors: Yang Liu, Fred Roosta. ArXiv:1909.06224

    "Newton-MR: Newton’s Method Without Smoothness or Convexity".
    Authors: Fred Roosta, Yang Liu, Peng Xu, Michael W. Mahoney. ArXiv: 1810.00303
