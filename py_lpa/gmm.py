import pandas as pd
pd.options.display.float_format = '{:,.4f}'.format
import numpy as np
import sklearn.mixture as mix
import scipy.stats as stats
import seaborn as sns
from scipy.stats import norm
from time import sleep
from tqdm import tqdm
from datetime import datetime
import itertools

"""
Disclaimer: This code is based on the code for multivariate Gaussian Mixture Models provided here by 
Xavier Bourret Sicotte:
https://xavierbourretsicotte.github.io/gaussian_mixture.html
"""


class GaussianMixture(object):
    """ 
    A class used to represent Gaussian Mixture Models

    Attributes
    ___________
    K: int
        the number auf gaussian distributions (i.e. latent categorical variables) that make up the data
    
    means_: array of type float
        means of each gaussian distribution

    covariances_: array of type float
        covariances of each gaussian distribution

    pi_: 

    log_likelihoods: 


    """
    
    def __init__(self, K = 3):
        self.K = K
        self.means_ = None
        self.covariances_ = None
        self.pi_ = None
        self.log_likelihoods = None
        self.constraints = None
        self.run_time = 0


    def initialization(self, *X, D, K, G, mean_constraint=None):
        """
        Initializes all components of the Gaussian Mixture.



        Parameters
        -----------
        X: a list of length G 
            list of all datasets

        D: int
            number of input dimensions

        K: int
            number of latent categorical variables = number of Gaussian Mixtures
        
        G: int
            number of datasets = manifest groups 

        mean_constraint: None, 'groups' - see fit method for description


        Returns
        ----------

        pi
            an array of starting probabilities for each class and dataset, 
            per default each class is assigned the same probability

        mu 
            an array of starting means for each class and dataset,
            per default these are random draws from the respective dataset

        sigma
            an array of starting covariances for each class and dataset
            per default these are diagonal matrices with 1 on the diagonal

        ng
            an array containing the number of observations in each dataset

        W
            a list of length G with all zero arrays of dimension ng x K
            this is weighting matrix that is used for facilitating the updating of parameters
        """
        # if there are not multiple datasets, unlist the data
        if len(X)==1:
            X=X[0]
        
        sigma = np.array([[np.eye(D)]*K]*G)
        pi = np.zeros((G,K))
        W = list()
        ng = [x.shape[0] for x in X]
        mu=np.zeros((G,K,D))


        for g in range(G):
            W.append(np.zeros((ng[g],K)))
            pi[g,:] = np.array([1 / K] * K)

        # if means are not constrained (or there is only one dataset) chhose starting value
        # at random from each dataset
        if mean_constraint == None:
            for g in range(G):
                mu[g,:,:] = X[g][np.random.choice(ng[g],K,replace = False)]

        # if means are constrained to be the same across datasets, choose starting value 
        # at random from all datasets
        if mean_constraint == 'groups':
            X = np.vstack(X)
            mu = np.tile(X[np.random.choice(X.shape[0],K,replace = False)],(G,1)).reshape(G,K,D)


        return (pi, mu, sigma, ng, W)

    def estep(self, X, mu, sigma, pi):
        """
        Expectation Step of the Algorithm.

        Parameters
        ------------

        X: array or list of arrays
            the data - this could be one dataset or a list consisting of multiple datasets with the same number of variables

        mu: array of floats
            an array that contains the current mean values, has to have  dimension(G,K,D) 
            i.e. number of datasets/groups x number of latent classes/Gaussians x number of manifest variables
        
        sigma: array of floats
            aan array that contains the current covariance values, has to have dimension (G, K, D, D)
            i.e. number of datasets/groups x number of latent classes/Gaussians x number of manifest variables x number of manifest variables

        pi: array of floats
            an array that contains the current proportions for the latent lasses/Gaussians, has to have dimension (G,K)
            i.e. number of datasets/groups x number of latent classes/Gaussians



        Returns
        ----------

        l: int
            loglikelihood of the data given mu, sigma and pi

        W: array or list of arrays
            weights for each observation in each dataset for each latent class/Gaussian -> used to update the parameters in the M-step
        
        W_s: array or list of arrays
            sum over the the observations of W, helper variable used for later calculations

        """

        K= self.K
        # probability density function of a multivariate normal distribution
        P = lambda m ,s: stats.multivariate_normal.pdf(x=X, mean = m, cov = s, allow_singular=True)

        # calculate the likelihood of each datapoint taken the current mean, cov and proporitons for given
        W = np.zeros((X.shape[0],K))
        for k in range(K):
            W[:, k] = pi[k] * P(mu[k], sigma[k])
        l = np.sum(np.log(np.sum(W, axis = 1)))
        W = (W.T / W.sum(axis = 1)).T

        # sum over the number of observations, helper variable used for later calculations
        W_s = np.nansum(W, axis = 0)

        return ([l,W,W_s])


    def mstep_mu(self, X, W, W_s, constraint=None):
        """
        Maximization Step - Update the mean
        
        Parameters
        ------------
        X, W, W_s - see estep method for description

        constraint: see 'mean_constraint' in the fit method for description
        
        Returns:
        ------------
        mu: array of floats
            updated means, has to have dimension (G,K,D)
        
        """
        # number of manifest variables
        D = X[0].shape[1]
        # number of datasets/groups
        G = len(X)
        # number of latent ariables
        K = self.K
        
        # no constraint
        if constraint == None:
            mu=np.zeros((G, K,D))
            update_mu = lambda W_s,W,X: (1. / W_s) * np.sum(W.T * X.T, axis = 1).T 
            for g in range(G):
                mu[g] = list(map(update_mu, W_s[g],W[g].T, itertools.repeat(X[g])))
        

        # equality within profile across groups
        elif constraint == 'groups':
            nom=np.zeros((G, K,D))
            denom=np.zeros((G,K,D))
            for g in range(G):
                for k in range(K):
                    nom[g][k] = np.sum((W[g][:,k] * X[g].T).T * np.sum(W_s[g]),axis=0)
                    denom[g][k] = np.sum(W[g][:,k]* np.sum(W_s[g]))

            mu = np.sum(nom, axis=0)/np.sum(denom,axis=0)   
            mu = np.array([mu]*G)

        else: 
            raise Exception('No implementation for this combination of constraints')

        
        return(mu)

    def mstep_sigma(self, X,W,W_s, mu,var_constraint='classes', cov_constraint='zero'):
        """
        Maximization Step - Update Covariance.

        Parameters:
        -------------
        X, W - see fit method for description

        mu: array of floats
            array of the updated means obtained from mstep_mu

        var_constraints, cov_constraints - see fit method for description

        Returns:
        -------------
        sigma: array of floats
            updated covaraince matrices, has to have dimension (G,K,D,D)
        
        """
        # number of manifest variables
        D = X[0].shape[1]
        # number of groups/datasets
        G = len(X)
        #number of latent variables/Gaussians
        K = self.K

        sigma = np.zeros((G,K,D,D))

        # standard Gaussian Mixture Model
        if var_constraint == None and cov_constraint == None:
            for g in range(G):
                for k in range(K):
                    sigma[g][k] = ((W[g][:,k] * ((X[g] - mu[g][k]).T)) @ (X[g] - mu[g][k])) / W_s[g][k]
        
        # variances unconstrained but zero covariance
        elif var_constraint == None and cov_constraint == 'zero':
            diag = np.zeros((G,K,D))

            for g in range(G):
                nom =0
                for k in range(K):
                    for d in range(D):
                        diag[g,k]= np.sum(W[g][:,k] * ((X[g]-mu[g,k])**2).T,axis=1)/np.sum(W[g][:,k])
            sigma = np.array(list(map(np.diag, diag.reshape(G*K,D)))).reshape(G,K,D,D)

         # "standard LPA"
         # variances are restricted to be equal across profiles and covariance is zero
        elif var_constraint == 'classes' and cov_constraint == 'zero':
            sigma = np.array((G,K,D,D))
            diag = np.zeros((G,D))
            for g in range(G):
                nom =0
                for k in range(K):
                    nom = nom + np.sum(W[g][:,k] * ((X[g]-mu[g][k])**2).T,axis=1)
                denom=np.sum(np.sum(W[g],axis=1),axis=0)
                diag[g] = nom/denom
            sigma = np.repeat(list(map(np.diag, diag)),K,axis=0).reshape(G,K,D,D)
        
        
        # variances are constrained to be the same across classes and groups
        # covariances are zero
        # together with mean_constraint='groups' this corresponds to dispersion invariance 
        elif var_constraint =='classes-groups' and cov_constraint == 'zero':
            sigma = np.array((G,K,D,D))
            nom =0
            for g in range(G):
                for k in range(K):
                    nom = nom + np.sum(W[g][:,k] * ((X[g]-mu[g][k])**2).T,axis=1)
            denom = np.sum(list(map(np.sum,W)))
            diag = np.diag(nom/denom)
            sigma = np.repeat([diag],K*G,axis=0).reshape(G,K,D,D)

        else: 
            raise Exception('No implementation for this combination of constraints')

        return sigma



    def mstep_pi(self, X,W,W_s, pi_constraint=None):

        """
        Maximization Step - Update Proportions.

        Parameters
        --------------
        X, W, W_s, pi_constraint - see fit method for description

        Returns:
        -------------
        pi: array of floats
            proportions of each latent class/Gaussian in each group/dataset, has to have dimension (G,K)
        
        """
        # number of manifest variables
        D = X[0].shape[1]
        # number of groups/datasets
        G = len(X)
        #number of latent variables/Gaussians
        K = self.K

        pi = np.zeros((G,K))
        
        # let proportions vary freely across groups and classes
        if pi_constraint == None:
            for g in range(G):
                for k in range(K):
                    pi[g][k] = W_s[g][k] / X[g].shape[0]

        # restrict proportions to be the same for each group/dataset
        elif pi_constraint == 'groups':
            nom = np.zeros((G,K))
            denom = 0
            for g in range(G):
                nom[g] = np.sum(W[g],axis=0)
                denom = denom + X[g].shape[0]
            pi = np.sum(nom,axis=0)/denom
            pi = np.array([pi] * G)

        else: 
            raise Exception('No implementation for this combination of constraints')


        return pi

    def run_EM(self, X,  rstarts=100, max_iter=100, tol=0.001,  n_solutions=1, init_mu=None, init_cov=None, init_pi=None, mean_constraint=None, var_constraint='groups', cov_constraint='zero', pi_constraint=None):
        """
        Function to run the actual Expectation Maximization Algorithm.

        Parameters
        -------------
        X: array or list of arrays
            the data - this could be one dataset or a list consisting of multiple datasets with the same number of variables
        
        rstarts: int
            number of times the model should run with a new set of random starting values each time
        
        max_iter: int
            maximum number of iterations to run the algorithm
        
        tol: float
        tolerance for log likelihood to determine convergence, i.e. if the log likelihood improved by less than tol from one run to the other the algorithm stops

        n_solutions: int
            number of solutions to return, defaults to 1 

        init_mu: array of floats
            [Optional]: an array to use as starting values for the mean, has to have dimension (G,K,D) 
            i.e. number of datasets/groups x number of latent classes/Gaussians x number of manifest variables
        
        init_cov: array of floats
            [Optional]: an array to use a starting values for the covariance, has to have dimension (G, K, D, D)
            i.e. number of datasets/groups x number of latent classes/Gaussians x number of manifest variables x number of manifest variables

        init_pi: array of floats
            [Optional]: an array to use as starting proportions for the latent lasses/Gaussians, has to have dimension (G,K)
            i.e. number of datasets/groups x number of latent classes/Gaussians

        .... for description of remaining parameters see the 'fit'-function


        Returns
        --------
        solutions: a list of containing the final values of 
                    - proportions of each class/Gaussian
                    - the means
                    - the covariances
                    - the log likelihood values for each iteration
                    - the maximum likelihood value that was found
        
        """
        try:
            len(np.unique([x.shape[1] for x in X]))==1
        except:
            raise Exception('The samples need to have the same number of manifest variables.')
        
        # number of datasets/groups
        G = len(X)
        # number of manifest variables
        D = X[0].shape[1]
        # number of latent classes/Gaussians
        K = self.K
        # variable to store maximum likelihood values 
        max_l = np.empty(0)
        solutions = []

        for s in tqdm(range(rstarts)):
            # initalisation
            pi,mu,sigma,W, ng = self.initialization([x for x in X], D=D, K=K, G=G)

            # override initialisations if starting values are provided
            if init_mu is not None:
                mu = init_mu
            if init_cov is not None:
                sigma = init_cov
            if init_pi is not None:
                pi = init_pi
            
            log_likelihoods = []

            while len(log_likelihoods) < max_iter:

                # Expectation Step
                eout = list(map(self.estep, X, mu, sigma, pi))       
                ll, W, W_s = [[i for i in element if i is not None] for element in  list(itertools.zip_longest(*eout))]
                l = np.sum(ll)

                log_likelihoods.append(l)

                # check for convergence
                if (len(log_likelihoods)>2 and np.abs(l - log_likelihoods[-2]) < tol): break
                      
                # Maximization Step
                mu = self.mstep_mu(X,W,W_s, constraint=mean_constraint)
                sigma = self.mstep_sigma(X,W,W_s,mu,var_constraint=var_constraint, cov_constraint=cov_constraint)
                pi = self.mstep_pi(X,W,W_s, pi_constraint=pi_constraint)
  

            max_l = np.append(max_l, max(log_likelihoods))
            solutions.append([pi,mu,sigma,log_likelihoods, max(log_likelihoods)])

        # find the best solution(s)
        best_solutions = np.argpartition(max_l,-n_solutions)[-n_solutions:]
       
        solutions = [solutions[i] for i in best_solutions]
        return(solutions)


    def fit(self,*X, rstarts=1, max_iter=100, tol=0.001, first_stage_iter=None, n_final_stage=1, mean_constraint=None, var_constraint=None, cov_constraint=None,pi_constraint=None):
        """
        Function to fit the model.

        Parameters
        --------------
        rstarts: int
            number of times the model should run with a new set of random starting values each time

        max_iter: int
            maximum number of iterations that the algorithm runs
        
        tol: float
            tolerance for log likelihood to determine convergence, i.e. if the log likelihood improved by less than tol from one run to the other the algorithm stops

        first_stage_iter: int
            [Optional]: this parameter can be used to together with #n_final_stage to introduce a 'first stage': This means, first #rstarts random starts are created and the algorithm runs for
            #first_stage_iter iterations. Then, the #n_final_stage solutions with the highest log likelihood are selected and only these advance to the 'second stage'. In the second stage another
            #max_iter - #first_stage_iter iterations of the algorithm are run for the second stage solutions.
        
        n_final_stage: int
            [Optional]: number of solutions to advance to the second stage
        
        mean_constraint: None, 'groups'
            [Optional]: A constraint to impose on the means of the Gaussians.
                - None: Means are allowed to vary freely across all dimensions.
                - 'groups': Means are allowed to vary across latent classes/profiles but are constrained to be equal across groups.

        var_constraint: 'classes', 'classes-groups'
            [Optional]: A constraint to impose on the variances i.e. the diagonals of the covariance matrices of the Gaussians.
                - 'classes': Variances are constrained to be the same for each latent class/profile. However, if there are multiple groups/datasets they can vary across these.
                - 'classes-groups': Variances are constrained to be the same for each latent class and each group.
        
        cov_constraint: None, 'zero'
            [Optional]: A constraint to impose on the covariances, i.e. the off-diagonal elemnts of the covariance matrices of the Gaussians.
            - None: Covariances are allowed to vary freely
            - 'zero': Covariances are restricted to be 0
        
        pi_constraint: None, 'groups'
            [Optional]: A constraint to impose on the proportions of the Gaussians/latent classes
            - None: No restriciton on the proportions
            -'groups': Proportions are restricted to be the same for all groups/datasets
        
        
        Returns
        -----------


        
        
        """
        # initialize time measurement
        t1 = datetime.now()

       # the case where we do have a first and second stage
        if first_stage_iter is not None:
           
            if rstarts <= n_final_stage:
                raise Exception('Number of Random Starts must be bigger than number of desired final stage solutions')

            if n_final_stage <= 1:
                raise Exception('When a number of frist stage iterations is specified, the number of desired final stage solutions needs to be bigger than 1.')

            # run first stage
            initial_solutions = self.run_EM(X,rstarts=rstarts, max_iter=first_stage_iter, tol=tol, n_solutions =n_final_stage, mean_constraint=mean_constraint,var_constraint=var_constraint, cov_constraint=cov_constraint,pi_constraint=pi_constraint)
            fpi, fmu, fsigma, flog_likelihoods, fmax_loglikelihood= [[i for i in element if i is not None] for element in  list(itertools.zip_longest(*initial_solutions))]

            
            # run second stage
            final_solutions = list(map(self.run_EM, itertools.repeat(X),itertools.repeat(1),itertools.repeat(max_iter-first_stage_iter),itertools.repeat(tol), itertools.repeat(1),fmu,fsigma,fpi,itertools.repeat(mean_constraint),itertools.repeat(var_constraint), itertools.repeat(cov_constraint), itertools.repeat(pi_constraint)))
            final_solutions = [[i for i in element if i is not None] for element in  list(itertools.zip_longest(*final_solutions))]
            [fsolutions] = final_solutions
            pi, mu, sigma, log_likelihoods, max_loglikelihood = [[i for i in element if i is not None] for element in  list(itertools.zip_longest(*fsolutions))]

            # find best solution and retain values
            s = np.argmax(np.array(max_loglikelihood))
            pi = pi[s]
            mu = mu[s]
            sigma = sigma[s]
            log_likelihoods = [flog_likelihoods[s][:-1],log_likelihoods[s]]
        else:
            solutions = self.run_EM(X,rstarts=rstarts, max_iter=max_iter, tol=tol, mean_constraint=mean_constraint,var_constraint=var_constraint, cov_constraint=cov_constraint,pi_constraint=pi_constraint)
            pi, mu, sigma, log_likelihoods, max_loglikelihood = [[i for i in element if i is not None] for element in  list(itertools.zip_longest(*solutions))]
            [pi]=pi
            [mu]=mu
            [sigma]=sigma
            [log_likelihoods] = log_likelihoods
        # calculate runtime
        t2 = datetime.now()
        run_time = t2-t1

        self.means_ = mu
        self.covariances_ = sigma
        self.log_likelihoods = log_likelihoods
        self.pi = pi
        self.constraints = [['Constraint on the Mean', mean_constraint], ['Constraint on the Variance', var_constraint],['Constraint on the Covariance',cov_constraint],['Constraint on Pi', pi_constraint]]
        self.run_time = run_time
           # opt_loglikelihoods = max(log_likelihoods)