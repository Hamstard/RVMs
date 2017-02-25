
# Simultaneous Regression of Derivatives / Antiderivatives using Relevance Vector Machines

This is a compilation of functions I've accumulated toying around with Relevance Vector Machines (RVMs) doing regression. RVMs are based on linear Bayesian regression 

$$ y_i = \vec{w}^T\cdot\vec{\phi}(\vec{x}_i) + \varepsilon $$

where $\vec{x}_i$ is the input for your model of your observation $y_i$ with the vector of basis function $\vec{\phi}$, corresponding weights $\vec{w}$ and a noise contribution $\varepsilon$. The, in my opinion, two most amazing parts of RVMs are:

1. they are highly sparse, with irrelevant basis vectors being "pruned out" and
2. you can actually regress multiple functions by properly setting up and partitioning the design matrix and the corresponding weights vector.

Thus using RVMs allows you to generate a model obeying Dr. Anita Faul's motto:

> A model is simple until proven otherwise.

So if you can think of a basis set appropriate for modelling your observations you are in business! Once you have your observations and specified your basis set the "only" thing left to do is to understand how to set up the *design matrix* $\Phi$, where each component $\Phi_{i,j}$ $ = \phi_{j}(\vec{x}_i)$ for $i\in[0,N]$ and $j\in[0,M]$ with $N$ the number of observations an $M$ the number of basis functions. The  specification of the basis set is totally up to you. For example if you want to regress a spectrum you measured and you already know what kind of signals should be underlying you can specify those as your *basis functions*. Note, however, that RVMs don't restrict the sign of the weights for the basis functions.

Recommended reading: 
* Tipping and Faul's original research papers (2001-2003)
* Bishop's book "Pattern Recognition and Machine Learning" section 7.2

## 1. Specify the basis set and the observations

Let's get started and generate some observations! Here two kind of basis sets are implemented, one is based on cosines and one Lorentzian type function. In this example we will regress a single function which has a cosine basis.


```python
%matplotlib notebook
import rvm_regression as rvm
import numpy as np

# specification of the observation itself - which shall later be regressed
obs_weights = {0:1,
               1:1,} # basis function "0" will have weight = 1 and basis function "1" will also have weight = 1, the rest will be 0.

beta = 100. # noise level
N = 100 # number of observations
lb, ub = 0, 2*np.pi # input domain
antiders = np.array([0],dtype=int) # only use the 0th antiderivative

# defining the basis set - for cosines we have cos(p[0]*pi*x + p[1]) where p are the values of "basis_coefs"
# this is the most crucial part for your specific problem, a bad basis set won't yield plausible fits (note: you can have very large basis sets)
basis_coefs = {0:np.array([0,0]),
               1:np.array([1,0]),
               2:np.array([2,0]),
               3:np.array([3,0])}
basis_type = "Cos" # "Cos" or "Lorentzian"

# set up the basis functions
rvm.set_basis_params(basis_coefs,basis_type,lb=lb,ub=ub,antiders=antiders)
basis, fun_names = rvm.get_basis(N_spline=1000) # N_spline is not relevant here but is used to generate integrals / derivatives of basis functions using scipy.interpolate.UnivariateSpline

# set up all inputs and outputs (noisy and true)
all_t, all_X, all_y = {}, {}, {}
for antider_type in antiders:
    all_t[antider_type], all_X[antider_type], all_y[antider_type] = rvm.generate_observations(lb,ub,obs_weights,N=N,beta=beta,antider_type=antider_type)

y_lb = min([np.amin(v) for v in all_t.values()])
y_ub = max([np.amax(v) for v in all_t.values()])

# plot the observations to be modelled
rvm.show_obs(all_t,all_X,all_y,y_lb,y_ub)
```

## 2. Setup the design matrix

Now the almighty *design matrix* $\Phi$...


```python
Phi, obs_mapper = rvm.get_design_matrix(all_X)
print("Phi {}".format(Phi.shape))

t = np.array([all_t[k] for k in sorted(antiders)])
t = np.reshape(np.ravel(t),(-1,1))
print("t {}".format(t.shape))
```

## 3. Run the RVM regression

Well this wasn't so bad afterall. Now let's specify whether or not to update $\beta$ with `fix_beta`, every how many steps to update it with `n_steps_beta`, whether or not all alphas are being updated simultaneously `sequential`, how many RVM iterations to do `niter` and what the initial hyper parameters (`alpha_init` and `beta_int`) should be.


```python
fix_beta = False # if False the noise level is adjusted every n_steps_beta
n_steps_beta = 1
sequential = False # if True one weight prior is updated at a time, if False all are updated at the same time (may be less stable)
niter = 5 # number of iteration steps for the RVM regression
alpha_init = np.ones(Phi.shape[1]) # initial hyperparameters for the weight priors
beta_init = 42 # initial noise level
       
alpha_curr, beta_curr, weights_new, logbook = rvm.iterate(niter,alpha_init,beta_init,Phi,t,verbose=True,
                                                         fix_beta=fix_beta,sequential=sequential,n_steps_beta=n_steps_beta)
```

## 4. Create some neat pictures

Now we are done regressing and can enjoy and optimized model for our observations with our basis set specified for the task. Below we find the regressed observations and amazingly enough, for this simple case, it turns out that the RVMs recover the almost exactly the original noise level!


```python
rvm.show_posterior(weights_new,alpha_curr,beta_curr,basis,Phi,y_lb,y_ub,obs_mapper,all_X,all_y,all_t,beta,Nx=100,Nt=50,scale="log")
```


```python
rvm.show_confidence(weights_new,alpha_curr,beta_curr,basis,Phi,y_lb,y_ub,obs_mapper,all_X,all_y,all_t,beta,
                    Nx=100,confidence=.95)
```

## 5. Regressing multiple functions at the same time

In this last step  we do all the steps as we did above for the single function, with the difference that we here regress __three functions__ simultaneously, the *original function* as well as its* first derivative* and *first antiderivative*! 

It turns out that RVMs don't care and the process is exactly the same! The only modification required is the selection of basis function given an observation, i.e. if you want to regress the antiderivative of a function you have to select the basis set which contains the respective antiderivatives of the basis functions. This is legal since we really just multiply $\vec{w}$ with $\Phi$ where the weights are the same for the a basis function independent of the derivative/antiderivative (if everything is done properly).


```python
%matplotlib notebook
%reset
import rvm_regression as rvm
import numpy as np

# specification of the observation itself - which shall later be regressed
obs_weights = {0:1,
               1:1,}

beta = 100.
N = 100
lb, ub = 0, 2*np.pi # the domain
antiders = np.array([-1,0,1],dtype=int) # generating observations for the first derivative, the function itself and its first antiderivative

# defining the basis set

basis_coefs = {0: np.array([[1,0,1]]),
                1: np.array([[1,1.5,1],
                             [1,2,.1]]),
                2: np.array([[1,1.5,1],
                             [1,2,1]])} # these parameters are totally not arbitrary!
basis_type = "Lorentzian" # well we already did the "Cos" version...

# basis functions available for regression

rvm.set_basis_params(basis_coefs,basis_type,lb=lb,ub=ub,antiders=antiders)
basis, fun_names = rvm.get_basis(N_spline=1000)

all_t, all_X, all_y = {}, {}, {}
for antider_type in antiders:
    all_t[antider_type], all_X[antider_type], all_y[antider_type] = rvm.generate_observations(lb,ub,\
                                                obs_weights,N=N,beta=beta,antider_type=antider_type)

y_lb = min([np.amin(v) for v in all_t.values()])
y_ub = max([np.amax(v) for v in all_t.values()])

rvm.show_obs(all_t,all_X,all_y,y_lb,y_ub)

# set up RVM
Phi, obs_mapper = rvm.get_design_matrix(all_X)
print("N (# observations) = {}, M (# basis functions) = {} ".format(*Phi.shape))

t = np.array([all_t[k] for k in sorted(antiders)])
t = np.reshape(np.ravel(t),(-1,1))

# regression
verbose = True
fix_beta = False
sequential = False
n_steps_beta = 1
niter = 5
alpha_init = np.ones(Phi.shape[1])
beta_init = 42
       
alpha_curr, beta_curr, weights_new, logbook = rvm.iterate(niter,alpha_init,beta_init, \
                                            Phi,t,verbose=verbose,fix_beta=fix_beta,
                                            sequential=sequential,n_steps_beta=n_steps_beta)

# plot
rvm.show_posterior(weights_new,alpha_curr,beta_curr,basis,Phi,y_lb,y_ub,obs_mapper,\
                   all_X,all_y,all_t,beta,Nx=100,Nt=50,scale="log")

rvm.show_confidence(weights_new,alpha_curr,beta_curr,basis,Phi,y_lb,y_ub,obs_mapper,all_X,all_y,all_t,beta,
                    Nx=100,confidence=.95)
```

That's it! Thanks for reading I hope this was useful for you and you enjoyed it!

Eric
