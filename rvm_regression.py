import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import UnivariateSpline as spline
import copy, itertools

my_globals = {"basis_params":None,
              "antiders":None,
              "lb":None,
              "ub":None,
              "basis":None,
              "fun_names":None}

implemented_basis_functions = {"Cos","Lorentzian"}

#################################################
######### Setting up the RVM regression #########
#################################################

def set_basis_params(basis_coefs,basis_type,lb=None,ub=None,antiders=np.array([0],dtype=int)):
    """Utility function to set parameters for the generation of the basis set.

    Parameters
    ----------
    basis_coefs : dict of np.ndarray
        key = the name of the basis function
        value = numpy array with parameters specific for the basis functions
    basis_type : Phi_sparse_trans
        specifies the basis functions, must be one of those specified in
        implemented_basis_functions
    lb : float
        specifies the lower bound of the input domain for the basis functions
    ub : float
        specifies the upper bound of the input domain for the basis functions
    antiders : np.ndarray of int
        specifies the occuring derivatives/antiderivatives. 
        Example 1: antiders = [-1,0,1] # first derivative, the function itself 
                                         and the 1st antiderivative
        Example 2: antiders = [0] # the function itself only
    """
    assert basis_type in implemented_basis_functions, "The specified basis_type ({}) isn't one of the expected: {}".format(basis_type,implemented_basis_functions)
    basis_params = {"basis_type":basis_type,
                    "basis_coefs":basis_coefs,}
    my_globals["basis_params"] = basis_params
    my_globals["antiders"] = antiders
    my_globals["lb"] = lb
    my_globals["ub"] = ub
   
# basis functions

def Lorentzian_fun(params,x):
    """Lorentzian function.
    
    Taken from: http://fityk.nieto.pl/model.html
    f = a0/(1+(x-a1)^2/a2)

    Parameters
    ----------
    params : np.ndarray
        a0,a1,a2 = params
    x : float / np.ndarray of float
        values from the input domain
    
    Returns
    -------
    Amazing functions values. Just fantastic.
    """
    N = params.shape[0]
    a_tuples = [tuple(list(params[i])) for i in range(N)]
    fun = lambda a0,a1,a2: a0/(1.+(x-a1)**2/a2)
    return np.sum([fun(*a) for a in a_tuples],axis=0)

def Cos_fun(params,x):
    """Cosine function.

    f = cos(pi*x*a0 + a1)
    """
    return np.cos(np.pi*x*params[0]+params[1])

# basis

def basis_fun_wrapper(d):
    """Wraps functions for implemented basis functions.
    """

    def _Lorentzian_fun(x):
        p = my_globals["basis_params"]["basis_coefs"][d]
        return Lorentzian_fun(p,x)
    
    def _Cos_fun(x):
        p = my_globals["basis_params"]["basis_coefs"][d]
        return Cos_fun(p,x)
    
    basis_type = my_globals["basis_params"]["basis_type"]
    if basis_type == "Lorentzian":
        return _Lorentzian_fun
    elif basis_type == "Cos":
        return _Cos_fun
    else:
        raise NotImplementedError

def get_basis(N_spline = 1000):
    """Generating the basis set.

    Parameters
    ----------
    N_spline : int
        default 1000. specifies the number of input domain values between lb and ub. this
        is important for the generation of the derivative and antiderivative basis set
        which is here based on the splines of the basis set for the actual function with
        antiders value 0.

    Returns
    -------
    basis : dict
        key = tuple (i,j) with i indicating the basis function and j the derivative
        Examples:
            (1,-1) is the second basis function for the first derivative
            (0,2) is the first basis function for the second antiderivative
            (42,0) is the 42nd basis function for the actual function
    keys : list of tuples 
        sorted list of the basis keys
    """
    basis_coefs = my_globals["basis_params"]["basis_coefs"]
    basis_type = my_globals["basis_params"]["basis_type"]
    antiders = my_globals["antiders"]
    lb, ub = my_globals["lb"], my_globals["ub"]

    set_antiders = set([v for v in antiders if v!=0])
    if len(set_antiders)>0: 
        assert lb is not None and ub is not None, "lb and ub need to be set if handling derivatives / integrals!"
        
    keys = sorted(basis_coefs.keys())
    basis = {(v,0):basis_fun_wrapper(v) for v in keys}
    if len(set_antiders)>0: 
        x = np.linspace(lb,ub,N_spline)
        splines = {v: spline(x,basis[(v,0)](x),s=.05) for v in keys}
    for santider in set_antiders:
        if santider>0:
            tmp = {(v,v2+1):splines[v].antiderivative(n=v2+1) for v in keys for v2 in range(santider)}
        else:
            tmp = {(v,-v2-1):splines[v].derivative(n=v2+1) for v in keys for v2 in range(abs(santider))}
        basis.update(tmp)
        
    my_globals["basis"] = basis
    my_globals["fun_names"] = keys
    
    return basis, keys
        
def generate_observations(lb,ub,obs_weights,N=100,beta=1,antider_type=0):
    """This is what is being observed and shall be regressed later.

    Parameters
    ----------
    lb : float
        input domain lower bound
    ub : float
        input domain upper bound
    obs_weights : dict
        key = name/integer for the respective basis function
        value = weight for the basis function
    N : int
        number of observations to generate
    beta : float
        noise level = 1/sigma^2
    antider_type : int
        0 = actual function
        1 = 1st antiderivative
        -1 = 1st derivative and so on

    Returns
    -------
    observation : np.ndarray of float
        noise observations
    X : np.ndarray of float
        input domain values for observations
    t : np.ndarray of float
        true value of observations if there is no noise
    """
    
    basis = my_globals["basis"]
    X = np.linspace(lb,ub,N)
    
    e_fun = lambda x: sum([w*basis[(v,antider_type)](x) for v,w in obs_weights.items()])
    t = e_fun(X)
        
    return t + np.random.normal(size=N,loc=0,scale=1./np.sqrt(beta)), X, t

#################################################
######### Setting up the RVM regression #########
#################################################

def get_design_matrix(all_X):
    """Calculates the N x M design matrix (N observations and M basis functions)
    
    Parameters
    ----------
    all_X : dict
        all input domain values to use for the modelling.
        keys = antider label, i.e. -1 (fist derivative), 0 (actual function), 1 (fist antiderivative),...
        values = np.ndarrays
    
    Returns
    -------
    Phi : np.ndarray of float
        The allmighty design matrix.
    mapper : dict
        key = antider label
        value = np.ndarray of int indicating which rows in Phi correspond to the given antider label
    """
    basis = my_globals["basis"]
    antiders = sorted(list(set([v[1] for v in basis.keys()])))
    phi_idx = sorted(list(set(v[0] for v in basis.keys())))
    M = len(phi_idx)
    mapper = {}
    c = 0
    for h,antider in enumerate(antiders):
        tmp_X = all_X[antider]
        tmp_basis = {k[0]: basis[k] for k in basis.keys() if k[1]==antider}
        N = len(all_X[antider])
        
        tmp_Phi = np.zeros((N,M),dtype=float)
        
        ijs = itertools.product(range(N),range(M))
        for i,j in ijs:
            tmp_Phi[i,j] = tmp_basis[j](tmp_X[i])
        if h==0:
            Phi = np.array(tmp_Phi)
        else:
            Phi = np.vstack((Phi,tmp_Phi))
        mapper[antider] = np.arange(c,c+N)
        c += N
    return Phi, mapper

#################################################
##############  RVM regression ##################
#################################################

def get_covariance_after_evidence(alpha,beta,Phi):
    """Calculates the covariance of p(w|t).
    
    Parameters
    ----------
    alpha : np.ndarray of float
    beta : float
    Phi : np.array of float (N,M)
        design matrix
    
    Returns
    -------
    inverse covariance matrix : np.ndarray (N,N) of float
    covariance matrix : np.ndarray (N,N) of float
    """
    N,M = Phi.shape
    A = np.eye(M)
    for i in range(M):
        A[i,i] = alpha[i]
    
    inv_covariance = A + beta * np.dot(Phi.T,Phi)
    covariance = np.linalg.inv(inv_covariance)
    return inv_covariance, covariance

def get_mean_after_evidence(beta,Sn,Phi,t):
    """Calculates the mean of p(w|t).
    
    Parameters
    ----------
    beta : float
    Sn : np.ndarray of float
        covariance matrix 
    Phi : np.array of float (N,M)
        design matrix
    t : np.ndarray of float
        noisy target values to be regressed
    
    Returns
    -------
    mean : np.ndarray
        most probable mean values of the weight distributions P(w|t)
    """
    N,M = Phi.shape
    mean = np.zeros(M,dtype=float)
    
    mean = beta * np.dot(Sn ,np.dot(Phi.T,t))
    mean = np.reshape(mean,(-1,))
    return mean

def get_updated_hyperparameters(t,mean,Phi,Sigma,N,M,alpha_old):
    
    gamma = np.array([1-alpha_old[v]*Sigma[v,v] for v in range(M)],dtype=float)
    alpha = np.array([gamma[v]/mean[v]**2 for v in range(M)])
    inv_beta = np.linalg.norm(t-np.dot(Phi,mean))**2 / (N-np.sum(gamma))
    beta = 1./inv_beta
    return beta, alpha

def get_updated_hyperparameters_sparse_sequential(i,t,mean_sparse,Phi_sparse,Phi_sparse_trans,Phi,Phi_trans,Sigma_sparse,
                                       N,M,alpha_old,beta_old):
    """Making use of section 7.2.2 in Bishop's book.
    
    Considering a single alpha

    """

    tmp0 = np.dot(Phi_sparse, np.dot(Sigma_sparse, Phi_sparse_trans))
    tmp1 = beta_old * Phi_trans[i,:]
    tmp2 = beta_old**2 * np.dot(Phi_trans[i,:],tmp0)
    tmp = tmp1 - tmp2
    Q = np.dot(tmp,t)
    
    S = np.dot(tmp,Phi[:,i])
    
    idx_fin = np.where(np.isfinite(alpha_old))[0]
    idx_inf = np.where(np.isinf(alpha_old))[0]

    alpha_sp = np.array(alpha_old[idx_fin])
    if np.isinf(alpha_old[i]):
        q = Q
        s = S
    else:
        q = alpha_old[i]*Q / (alpha_old[i] - S)
        s = alpha_old[i]*S / (alpha_old[i] - S)    
        
    gamma = np.array([1.-alpha_sp[v]*Sigma_sparse[v,v] for v in range(len(alpha_sp))],dtype=float)
    alpha_update = s**2/(q**2-s)
    
    N = float(len(t))
    t_pred = np.dot(Phi_sparse,mean_sparse)
    t_pred = np.reshape(t_pred,(-1,1))
    
    inv_beta = np.linalg.norm(t-t_pred)**2 / (N - M + np.sum([alpha_sp[v]*Sigma_sparse[v,v] for v in range(len(alpha_sp))]))
    beta = 1./inv_beta
    
    q2_smaller_equal_s = np.where(q**2<=s)[0]
    q2_larger_s = np.where(q**2>s)[0]
    
    alpha = np.array(alpha_old)
    
    if q**2 > s: # q**2 > s and alpha is finite/infinte - update with 7.101
        alpha[i] = alpha_update
    elif q**2 < s and np.isfinite(alpha_old[i]): # q**2 < s and alpha is finite set new alpha to infinite
        alpha[i] = np.inf
    if np.isinf(alpha).all():
        alpha[0] = 1.
    if np.where(alpha<0)[0].any():
        
        print("to update with 7.101 {}".format(sorted(list(q2_larger_s))))
        print("q**2>s -> alpha {}".format(alpha[q2_larger_s]))
        print("alpha_old {}".format(alpha_old))
        print("alpha {}".format(alpha))
        print("q2 <= s {}".format(q2_smaller_equal_s))
        print("alpha finite {}".format(alpha_fin))
        print("intersection q2 <= s and alpha finite {}".format(np.intersect1d(q2_smaller_equal_s,alpha_fin)))
        print("s {}".format(s))
        print("q**2 {}".format(q**2))
        raise
    
    return alpha, beta

def get_updated_hyperparameters_sparse(t,mean_sparse,Phi_sparse,Phi_sparse_trans,Phi,Phi_trans,Sigma_sparse,
                                       N,M,alpha_old,beta_old):
    """Making use of section 7.2.2 in Bishop's book.
    
    Considering all alpha at the same time

    """

    tmp0 = np.dot(Phi_sparse, np.dot(Sigma_sparse, Phi_sparse_trans))
    
    tmp1 = beta_old * Phi_trans
    tmp2 = beta_old**2 * np.dot(Phi_trans,tmp0)
    
    tmp = tmp1 - tmp2
    Q = np.dot(tmp,t)
    Q = np.reshape(Q,(-1,))
    S = np.array([np.dot(tmp[i,:],Phi[:,i]) for i in range(M)])
    
    q = np.zeros(M)
    s = np.zeros(M)
    alpha_inf = np.where(np.isinf(alpha_old))[0]
    alpha_fin = np.where(np.isfinite(alpha_old))[0]
    alpha_sp = alpha_old[alpha_fin]
    q[alpha_inf] = Q[alpha_inf]
    q[alpha_fin] = alpha_sp*Q[alpha_fin] / (alpha_sp - S[alpha_fin])
    s[alpha_inf] = S[alpha_inf]
    s[alpha_fin] = alpha_sp*S[alpha_fin] / (alpha_sp - S[alpha_fin])
    
    gamma = np.array([1.-alpha_sp[v]*Sigma_sparse[v,v] for v in range(len(alpha_sp))],dtype=float)
    alpha_update = s**2/(q**2-s)

    N = float(len(t))
    t_pred = np.dot(Phi_sparse,mean_sparse)
    t_pred = np.reshape(t_pred,(-1,1))
    
    inv_beta = np.linalg.norm(t-t_pred)**2 / (N - M + np.sum([alpha_sp[v]*Sigma_sparse[v,v] for v in range(len(alpha_sp))]))
    beta = 1./inv_beta 
        
    q2_smaller_equal_s = np.where(q**2<=s)[0]
    q2_larger_s = np.where(q**2>s)[0]
    # q**2 > s and alpha is finite/infinte - update with 7.101
    alpha = np.array(alpha_old)
    alpha[q2_larger_s] = alpha_update[q2_larger_s]
    
    # q**2 < s and alpha is finite set new alpha to infinite
    test_alpha = np.array(alpha)
    test_alpha[np.intersect1d(q2_smaller_equal_s,alpha_fin)] = np.inf
    if (test_alpha).all(): #in case that all remaiing fin alphas are to be set to inf keep one finite value
        alpha[np.intersect1d(q2_smaller_equal_s,alpha_fin[1:])] = np.inf
    
    if np.where(alpha<0)[0].any():
        
        print("to update with 7.101 {}".format(sorted(list(q2_larger_s))))
        print("q**2>s -> alpha {}".format(alpha[q2_larger_s]))
        print("alpha_old {}".format(alpha_old))
        print("alpha {}".format(alpha))
        print("q2 <= s {}".format(q2_smaller_equal_s))
        print("alpha finite {}".format(alpha_fin))
        print("intersection q2 <= s and alpha finite {}".format(np.intersect1d(q2_smaller_equal_s,alpha_fin)))
        print("s {}".format(s))
        print("q**2 {}".format(q**2))
        raise
    
    return alpha, beta

def get_quality_measures(t,weights,Phi,alpha,beta):
    """
    Returns just a bunch of quality/difference measures, such as mse, loglikelihood,...

    """
    t_pred = np.dot(Phi,weights)
    t_pred = np.reshape(t_pred,(-1,1))
    N,M = Phi.shape
    
    hyper = np.sum(np.log(alpha)) + N * np.log(beta)
    dt = t-t_pred
    evidence = beta * np.sum(dt**2)
    ln2pi = np.log(2*np.pi)
    regularizer = np.dot(alpha,weights**2)
    
    log_post = - .5*(M+N)*ln2pi + .5*N*np.log(beta) + .5*np.sum(np.log(alpha) - alpha*(weights**2)) - .5*evidence
    total_square_error = np.sqrt(np.sum(dt**2))
    mean_square_error = np.sqrt(np.mean(dt**2))
    median_square_error = np.sqrt(np.median(dt**2))
    min_delta = np.amin(dt)
    max_delta = np.amax(dt)
    return {"log_post":log_post,"tse":total_square_error,"mse":mean_square_error,"min":min_delta,"max":max_delta,"median_se":median_square_error}

def iterate(niter,alpha_init,beta_init,Phi,t,verbose=False,fix_beta=False,sequential=False,n_steps_beta=1):
    logbook = {"log_post":[],"alphas":[],"beta":[],"weights":[],"mse":[],"tse":[],"min":[],"max":[]}
    N,M = Phi.shape
    for i in range(niter):
        
        alpha_fin = np.where(np.isfinite(alpha_init))[0]
        alpha_inf = np.where(np.isinf(alpha_init))[0]
        if verbose: 
            print("\niteration {}/{}".format(i+1,niter))
            print("    > active phis {}...".format(len(alpha_fin)))

        Phi_sparse = Phi[:,alpha_fin]
        alpha_sparse = alpha_init[alpha_fin]
        inv_Sn_sparse, Sn_sparse = get_covariance_after_evidence(alpha_sparse,beta_init,Phi_sparse)
        weights_new_sparse = get_mean_after_evidence(beta_init,Sn_sparse,Phi_sparse,t)

        if np.isnan(weights_new_sparse).any():
            print("\nSn {}".format(Sn_sparse))
            print("\ninv_Sn {}".format(inv_Sn_sparse))
            print("\nalpha_init {}".format(alpha_init_sparse))
            print("\nbeta_init {}".format(beta_init))
            raise
        
        qualities = get_quality_measures(t,weights_new_sparse,Phi_sparse,alpha_sparse,beta_init)
        logbook["log_post"].append(qualities["log_post"])
        logbook["mse"].append(qualities["mse"])
        logbook["tse"].append(qualities["tse"])
        logbook["min"].append(qualities["min"])
        logbook["max"].append(qualities["max"])
        logbook["alphas"].append(alpha_init)
        logbook["beta"].append(beta_init)
        logbook["weights"].append(weights_new_sparse)
        
        if verbose: print("    > log posterior {} mse {} tse {} min {} max {}".format(round(logbook["log_post"][i],6),round(logbook["mse"][i],6),round(qualities["tse"],6),round(qualities["min"],6),round(qualities["max"],6)))

        Phi_sparse_trans = Phi_sparse.T
        
        if sequential:
            # update a single alpha
            i_alpha = i%M
            print("    > i_alpha {}".format(i_alpha))
            alpha_new, beta_new = get_updated_hyperparameters_sparse_sequential(i_alpha,t,weights_new_sparse,Phi_sparse,Phi_sparse_trans,
                                                                    Phi,Phi.T,Sn_sparse,N,M,alpha_init,beta_init)
        else:
        
            # update all alpha at one
            alpha_new, beta_new = get_updated_hyperparameters_sparse(t,weights_new_sparse,Phi_sparse,Phi_sparse_trans,
                                                                    Phi,Phi.T,Sn_sparse,N,M,alpha_init,beta_init)
        
        if i<niter-1:

            alpha_init = copy.deepcopy(alpha_new)

            if (not fix_beta) and i%n_steps_beta==0 and i>0:
                beta_init = copy.deepcopy(beta_new)
                #print("not fix_beta {} niter%n_steps_beta = {} % {} = {}".format(not fix_beta,niter,n_steps_beta,niter%n_steps_beta))
                if verbose: print("    > updated beta = {} ...".format(beta_init))
        else:
            alpha_curr = alpha_init
            beta_curr = beta_init
            weights_new = np.zeros(M)
            weights_new[alpha_fin] = weights_new_sparse
    return alpha_curr, beta_curr, weights_new, logbook

def predict(X,t,weights,alpha,beta,basis):
    """Calculates log p(t|x,w,alpha,beta).
    
    """
    N_x, N_t = len(X), len(t)
    M = len(basis)
    log_p = np.zeros((N_x,N_t))
    Xg, Tg = np.meshgrid(X,t,indexing='ij')
    
    idx_fin = np.where(np.isfinite(alpha))[0]
    hyper = np.sum(np.log(1./alpha[idx_fin])) + np.log(1./beta)
    
    regularizer = .5 * np.sum(np.dot(alpha[idx_fin],weights[idx_fin]**2))
    
    for i_x in range(N_x):
        phi = np.array([basis[v](X[i_x]) for v in idx_fin],dtype=float)
        tmp = np.dot(phi,weights[idx_fin])
        for i_t in range(N_t):
            evidence = beta * np.sum( (t[i_t]-tmp)**2 )
            log_p[i_x,i_t] =  - .5 * (hyper + evidence + regularizer)
    return log_p, Xg, Tg

#################################################
################## Plotting #####################
#################################################

def show_obs(all_t,all_X,all_y,y_lb,y_ub):
    antiders = my_globals["antiders"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    N = float(len(antiders))
    c = 0
    for antider_type in antiders:
        ax.plot(all_X[antider_type],all_t[antider_type],'o',c=plt.cm.bone(c/N),label="observed (antider {})".format(antider_type))
        ax.plot(all_X[antider_type],all_y[antider_type],'-',c=plt.cm.magma(c/N),label="true (antider {})".format(antider_type),linewidth=2)
        c += 1
    
    ax.grid()
    ax.set_xlabel("x (model input)",fontsize=22)
    ax.set_ylabel("y (model output)",fontsize=22)
    ax.tick_params(labelsize=20)
    plt.legend(loc=0)
    plt.title("Observations to be modelled",fontsize=24)
    plt.tight_layout()
    plt.show()

def show_posterior(weights_new,alpha_curr,beta_curr,basis,Phi,y_lb,y_ub,obs_mapper,all_X,all_y,all_t,beta,
                        Nx=100,Nt=50,ytol=.1,scale="log"):
    lb,ub = my_globals["lb"], my_globals["ub"]

    X_pred = np.linspace(lb,ub,Nx)
    t_pred = np.linspace(y_lb-ytol,y_ub+ytol,Nt)

    log_p_bayes, Xg, Tg = {}, {}, {}

    for antider in my_globals["antiders"]:
        tmp_basis = [basis[v] for v in sorted(basis.keys(),key = lambda x: x[0]) if v[1]==antider]
        log_p_bayes[antider], Xg[antider], Tg[antider] = predict(X_pred,t_pred,weights_new,alpha_curr,beta_curr,tmp_basis)
        

    for antider in my_globals["antiders"]:
        tmp_Phi = Phi[obs_mapper[antider],:]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(all_X[antider],all_t[antider],'o',label="observed (antider {}, beta = {})".format(antider,beta),alpha=0.3)
        ax.plot(all_X[antider],all_y[antider],'-k',label="true".format(antider),
            linewidth=7,alpha=0.7)
        ax.plot(all_X[antider],np.dot(tmp_Phi,weights_new),'-m',
            label="inferred (beta = {})".format(np.around(beta_curr,decimals=2)),
            linewidth=3,alpha=0.9)
        if scale!="log":
            cax = ax.contourf(Xg[antider],Tg[antider],np.exp(log_p_bayes[antider]),cmap=plt.cm.magma,alpha=0.6)
        else:
            cax = ax.contourf(Xg[antider],Tg[antider],log_p_bayes[antider],cmap=plt.cm.magma,alpha=0.6)
        cb = fig.colorbar(cax)
        if scale == "log":
            cb.ax.set_ylabel(r"log P(t|X,w,$\alpha$,$\beta$)",fontsize=20)
        else:
            cb.ax.set_ylabel(r"P(t|X,w,$\alpha$,$\beta$)",fontsize=20)
        cb.ax.tick_params(labelsize=18)

        ax.set_xlabel("x (model input)",fontsize=22)
        ax.set_ylabel("y (model output)",fontsize=22)
        ax.tick_params(labelsize=20)
        plt.title("Antider {}".format(antider),fontsize=24)
        ax.grid()
        plt.legend(loc=0)
        plt.tight_layout()
        
    plt.show()

import matplotlib.pylab as plt
import scipy.special as sps

def show_confidence(weights_new,alpha_curr,beta_curr,basis,Phi,y_lb,y_ub,obs_mapper,all_X,all_y,all_t,beta,
                        Nx=100,confidence=.95):
    lb,ub = my_globals["lb"], my_globals["ub"]

    X_pred = np.linspace(lb,ub,Nx)
    
    conf_deltas, t_pred = {}, {}
    sig = 1./np.sqrt(beta_curr)
    for antider in my_globals["antiders"]:
        tmp_basis = [basis[v] for v in sorted(basis.keys(),key = lambda x: x[0]) if v[1]==antider]
        
        idx_fin = np.where(np.isfinite(alpha_curr))[0]
        
        phi = np.array([tmp_basis[v](X_pred) for v in idx_fin],dtype=float).T
        
        tmp = np.dot(phi,weights_new[idx_fin])
        tmp_confi = sps.erfinv(confidence)*sig*np.sqrt(2.)
        conf_deltas[antider] = tmp_confi
        t_pred[antider] = tmp

    for antider in my_globals["antiders"]:
                
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(all_X[antider],all_t[antider],'o',label="observed (antider {}, beta = {})".format(antider,beta),alpha=0.3)
        ax.plot(all_X[antider],all_y[antider],'-k',label="true".format(antider),
            linewidth=7,alpha=0.7)
        ax.fill_between(X_pred,t_pred[antider]-conf_deltas[antider],t_pred[antider]+conf_deltas[antider],alpha=0.2,
                        color='b',label="{}% confidence interval".format(confidence*100))
        ax.plot(all_X[antider],t_pred[antider],'-m',
            label="inferred (beta = {})".format(np.around(beta_curr,decimals=2)),
            linewidth=3,alpha=0.9)
        
        ax.set_xlabel("x (model input)",fontsize=22)
        ax.set_ylabel("y (model output)",fontsize=22)
        ax.tick_params(labelsize=20)
        plt.title("Antider {}".format(antider),fontsize=24)
        ax.grid()
        plt.legend(loc=0)
        plt.tight_layout()
        
    plt.show()