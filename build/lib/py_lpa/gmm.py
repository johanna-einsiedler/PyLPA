import numpy as np
from scipy import stats
from tqdm import tqdm
from datetime import datetime
import itertools
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


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
        self.means_std = None
        self.covariances_ = None
        self.pi_ = None
        self.log_likelihoods = None
        self.constraints = None
        self.run_time = 0
        self.X=None
        self.rstarts=None


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
                #if (not np.any(W_s)): break     
                # Maximization Step
                mu = self.mstep_mu(X,W,W_s, constraint=mean_constraint)
                sigma = self.mstep_sigma(X,W,W_s,mu,var_constraint=var_constraint, cov_constraint=cov_constraint)
                pi = self.mstep_pi(X,W,W_s, pi_constraint=pi_constraint)
  
            #print(log_likelihoods)
            max_l = np.append(max_l, max(log_likelihoods))
            solutions.append([pi,mu,sigma,log_likelihoods, max(log_likelihoods)])

        # find the best solution(s)
        best_solutions = np.argpartition(max_l,-n_solutions)[-n_solutions:]
       
        solutions = [solutions[i] for i in best_solutions]
        return(solutions)


    def fit(self,*X, rstarts=1, max_iter=100, tol=0.001, first_stage_iter=None, n_final_stage=1, init_mu=None, init_cov=None, init_pi=None, mean_constraint=None, var_constraint=None, cov_constraint=None,pi_constraint=None):
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
               init_mu: array of floats
            [Optional]: an array to use as starting values for the mean, has to have dimension (G,K,D) 
            i.e. number of datasets/groups x number of latent classes/Gaussians x number of manifest variables
        
        init_cov: array of floats
            [Optional]: an array to use a starting values for the covariance, has to have dimension (G, K, D, D)
            i.e. number of datasets/groups x number of latent classes/Gaussians x number of manifest variables x number of manifest variables

        init_pi: array of floats
            [Optional]: an array to use as starting proportions for the latent lasses/Gaussians, has to have dimension (G,K)
            i.e. number of datasets/groups x number of latent classes/Gaussians

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
            initial_solutions = self.run_EM(X,rstarts=rstarts, max_iter=first_stage_iter, init_mu=init_mu, init_cov=init_cov, init_pi=init_pi, tol=tol, n_solutions =n_final_stage, mean_constraint=mean_constraint,var_constraint=var_constraint, cov_constraint=cov_constraint,pi_constraint=pi_constraint)
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

        # standardize the mean
        mu_std = np.zeros_like(mu)
        for g in range(len(X)):
            mu_std[g] = (mu[g]-np.mean(X[g],axis=0))/np.array(list(map(np.diagonal, np.sqrt(sigma[g]))))
        # calculate runtime
        t2 = datetime.now()
        run_time = t2-t1

        self.means_ = mu
        self.means_std_ = mu_std
        self.covariances_ = sigma
        self.log_likelihoods = log_likelihoods
        self.pi_ = pi
        self.constraints = [['Constraint on the Mean', mean_constraint], ['Constraint on the Variance', var_constraint],['Constraint on the Covariance',cov_constraint],['Constraint on Pi', pi_constraint]]
        self.run_time = run_time
        self.X = X
        self.rstarts = rstarts
        self.max_iter = max_iter
        self.tol = tol
           # opt_loglikelihoods = max(log_likelihoods)

    def predict_proba(self,*x0):
        """
        Function to predict posterior probabilites for each observation to belong to a specific Gaussian/Latent Class

        Parameters
        --------------
        x0: Dataset of observations to calculate posterior probabilities for

        Returns
        -------------
        all_probs: A list of dimension G (i.e. number of groups/datasets) x N (i.e. number of observations) x K (i.e. number of Gaussians/Latent Classes)
        with the probabilty for every datapoint to belong to a certain Gaussian/Class
        """
        # for each observation predict probability of belonging to a specific Gaussian/Latent Class
        all_probs =[]
        for g in range(len(x0)):
            probs = np.array([ stats.multivariate_normal.pdf(x0[g], mean = self.means_[g][k], cov = self.covariances_[g][k]) for k in range(self.K) ])
            all_probs.append(probs.T)
        return all_probs

    def predict(self,*x0):
        """
        Function to get assigned class/Gaussian for each observation

        Parameters
        --------------
        x0: Dataset of observations to calculate class assignment for

        Returns
        -------------
        all_probs: A list of dimension G (i.e. number of groups/datasets) x N (i.e. number of observations) with the number of the class/Gaussian an
        observation was assigned to
       
        """    
        all_probs =[]
        for g in range(len(x0)):
            probs = np.array([ stats.multivariate_normal.pdf(x0[g], mean = self.means_[g][k], cov = self.covariances_[g][k]) for k in range(self.K) ])
            all_probs.append(np.argmax(probs,axis=0))
        return all_probs

    def bootstrap_raw(self,*X,nbs, rstarts=0, max_iter=0, tol=0, mean_constraint=0, var_constraint=0, cov_constraint=0, pi_constraint=0):
        """
        Function to boostrap standard errors
        Based on O’Hagan, A., Murphy, T.B., Scrucca, L. et al. Investigation of parameter uncertainty in clustering using a Gaussian mixture model via jackknife, bootstrap and weighted likelihood bootstrap. Comput Stat 34, 1779–1813 (2019). https://doi.org/10.1007/s00180-019-00897-9
        
        i.e we draw samples of the  same size as the original data with replacement from the original data and then perform LPA with the optimal values obtained from the original LPA as starting values
        we den calculate the sample standard deviation from the solutions

        Parameters:
        ------------------
        X: data to sample from for the bootstrap

        nbs: number of bootstrap samples to draw

        ... the remaining are the same parameters as for the original EM algorithm, per default the values specified in the fit function are chosen

        Returns:
        ----------------
        sd_solutions:  a vector with estimated standard deviations for pi, mu and sigma for each group/dataset
        
        
        
        """
        G = len(X)
        preds = []
        for g in range(G):
            preds.append(self.predict_proba(X[g])[0])
        bs_pi =np.zeros((nbs,G,self.K))
        bs_mu = np.zeros((nbs,G,self.K,X[0].shape[1]))
        bs_cov = np.zeros((nbs,G,self.K, X[0].shape[1],X[0].shape[1]))
        if(rstarts == 0):
            rstarts =self.rstarts
        if(max_iter == 0):
            max_iter = self.max_iter
        if(tol == 0):
            tol = self.tol
        if(mean_constraint == 0):
            mean_constraint = self.constraints[0][1]
        if(var_constraint==0):
            var_constraint=self.constraints[1][1]
        if(cov_constraint==0):
            cov_constraint=self.constraints[2][1]
        if(pi_constraint ==0):
            pi_constraint=self.constraints[3][1]
        
        mu = self.means_
        cov = self.covariances_
        pi = self.pi_


        for g in range(G):
            for i in range(nbs):
                # draw a random sample from X with replacement
                bindex = np.random.choice(X[g].shape[0],X[g].shape[0],replace=True)
                bsample = np.array(np.take(X[g],bindex,axis=0)).astype(float)
                bpost = np.take(preds[g],bindex,axis=0)
                #print(bpost.shape)
                #print(len(preds))
                try:
                    bsrun = self.run_EM([bsample], rstarts=rstarts, max_iter=max_iter, tol =tol, mean_constraint=mean_constraint, var_constraint=var_constraint,cov_constraint=cov_constraint, pi_constraint=pi_constraint,init_pi=bpost)
                except:
                    i=i-1
                    pass
                bmu = bsrun[0][1][0]
                # calculate pairwise distance
                #print(mu[0])
                #print(bmu)
                D = distance_matrix(mu[0],bmu)
                org, reorder = linear_sum_assignment(D)
                bmu = bmu[reorder]

                #print(bsrun[0][0])
                #print(reorder)
                bs_pi[i,g,:] = bsrun[0][0][0][reorder]
                bs_mu[i,g,:]=bmu
                bs_cov[i,g,:]=bsrun[0][2][0][reorder]

        pi_sd = np.sqrt(1/(bs_pi.shape[0]-1)*np.sum((bs_pi-np.mean(bs_pi,axis=0))**2,axis=0))
        #mu_sd = np.sqrt(1/(bs_mu.shape[0]-1)*np.sum((bs_mu-np.mean(bs_mu,axis=0))**2,axis=0))
        mu_sd = np.sqrt(1/(bs_mu.shape[0]-1)*np.sum(np.sqrt((bs_mu-np.mean(bs_mu,axis=0))**2),axis=0))
        #mu_sd =((bs_mu.shape[0]-1)/bs_cov.shape[0]*np.sum(bs_mu-np.mean(bs_mu,axis=0),axis=0))**2
        cov_sd = np.sqrt(1/(bs_cov.shape[0]-1)*np.sum((bs_cov-np.mean(bs_cov,axis=0))**2,axis=0))
        #cov_sd =((bs_cov.shape[0]-1)/bs_cov.shape[0]*np.sum(bs_cov-np.mean(bs_cov,axis=0),axis=0))**2

        sd_solutions = [pi_sd, mu_sd,cov_sd]
        return sd_solutions

    def bootstrap_fitted(self,*X,nbs, rstarts=0, max_iter=0, tol=0, mean_constraint=0, var_constraint=0, cov_constraint=0, pi_constraint=0):
        """
        Function to boostrap standard errors
        Based on Basford K. E.; McLachlan G. J. (1997): Standard errors of fitted component means of normal mixtures. https://www.researchgate.net/publication/37625647_Standard_errors_of_fitted_component_means_of_normal_mixtures?enrichId=rgreq-00d4b1f6ca093f66ea93fd49d03d6f43-XXX&enrichSource=Y292ZXJQYWdlOzM3NjI1NjQ3O0FTOjEwNDU1Njc3MTkzODMxOEAxNDAxOTM5Njg1OTQ1&el=1_x_2&_esc=publicationCoverPdf
        
        i.e we draw samples of the  same size as the original data with replacement from the fitted distributions and then perform LPA with the optimal values obtained from the original LPA as starting values
        we den calculate the sample standard deviation from the solutions
        
        Parameters:
        ------------------
        X: data to sample from for the bootstrap

        nbs: number of bootstrap samples to draw

        ... the remaining are the same parameters as for the original EM algorithm, per default the values specified in the fit function are chosen

        Returns:
        ----------------
        sd_solutions:  a vector with estimated standard deviations for pi, mu and sigma for each group/dataset
        
        
        
        """
        G = len(X)
        preds = []
        for g in range(G):
            preds.append(self.predict_proba(X[g])[0])
        D = X[0].shape[1]
        bs_pi =np.zeros((nbs,G,self.K))
        bs_mu = np.zeros((nbs,G,self.K,X[0].shape[1]))
        bs_cov = np.zeros((nbs,G,self.K, X[0].shape[1],X[0].shape[1]))
        if(rstarts == 0):
            rstarts =self.rstarts
        if(max_iter == 0):
            max_iter = self.max_iter
        if(tol == 0):
            tol = self.tol
        if(mean_constraint == 0):
            mean_constraint = self.constraints[0][1]
        if(var_constraint==0):
            var_constraint=self.constraints[1][1]
        if(cov_constraint==0):
            cov_constraint=self.constraints[2][1]
        if(pi_constraint ==0):
            pi_constraint=self.constraints[3][1]
        
        mu = self.means_
        cov = self.covariances_
        pi = self.pi_

        # calculate number of samples to draw from each distribution
        num_samples = np.round(self.pi_ * X[0].shape[0])

        for g in range(G):
            for i in range(nbs):
                bsample = np.empty((0,int(D)))
                for k in range(self.K):
                #bindex = np.random.choice(X[g].shape[0],X[g].shape[0],replace=True)
                    bsample_part = np.random.multivariate_normal(self.means_[g][k],self.covariances_[g][k], size=num_samples[0][k].astype(int))
                    bsample = np.vstack([bsample, bsample_part])
                bsrun = self.run_EM([bsample], rstarts=rstarts, max_iter=max_iter, tol =tol, mean_constraint=mean_constraint, var_constraint=var_constraint,cov_constraint=cov_constraint, pi_constraint=pi_constraint,init_pi=self.pi_)
                
                bmu = bsrun[0][1][0]
                # calculate pairwise distance
                #print(mu[0])
                #print(bmu)
                Dist = distance_matrix(mu[0],bmu)
                org, reorder = linear_sum_assignment(Dist)
                bmu = bmu[reorder]

                #print(bsrun[0][0])
                #print(reorder)
                bs_pi[i,g,:] = bsrun[0][0][0][reorder]
                bs_mu[i,g,:]=bmu
                bs_cov[i,g,:]=bsrun[0][2][0][reorder]

        pi_sd = np.sqrt(1/(bs_pi.shape[0]-1)*np.sum((bs_pi-np.mean(bs_pi,axis=0))**2,axis=0))
        #mu_sd = np.sqrt(1/(bs_mu.shape[0]-1)*np.sum((bs_mu-np.mean(bs_mu,axis=0))**2,axis=0))
        mu_sd = np.sqrt(1/(bs_mu.shape[0]-1)*np.sum(np.sqrt((bs_mu-np.mean(bs_mu,axis=0))**2),axis=0))
        #mu_sd =((bs_mu.shape[0]-1)/bs_cov.shape[0]*np.sum(bs_mu-np.mean(bs_mu,axis=0),axis=0))**2
        cov_sd = np.sqrt(1/(bs_cov.shape[0]-1)*np.sum((bs_cov-np.mean(bs_cov,axis=0))**2,axis=0))
        #cov_sd =((bs_cov.shape[0]-1)/bs_cov.shape[0]*np.sum(bs_cov-np.mean(bs_cov,axis=0),axis=0))**2

        sd_solutions = [pi_sd, mu_sd,cov_sd]
        return sd_solutions

    def BIC(self):
        """
        Function to calculate the Bayesian Information Criterion for the maximum likelihood solution
        Parameters:
        ------------------
        self:
        A fitted Gaussian Mixture Model

        ----------------
        BIC: the Bayesian Information Criterion

        """
        max_loglik = [item for sublist in self.log_likelihoods for item in sublist][-1]
        N = self.X[0].shape[0]
        dfmeans = len(np.unique(self.means_))
        dfvars = len(np.unique(self.covariances_[self.covariances_!=0]))
        dfclasses = self.pi_.shape[1]-1
        df = dfmeans + dfvars+ dfclasses
        return -2*max_loglik + np.log(N)*df

    def AIC(self):
        """
        Function to calculate the Akaike Information Criterion for the maximum likelihood solution
        Parameters:
        ------------------
        self:
        A fitted Gaussian Mixture Model

        ----------------
        AIC: the Akaike Information Criterion

        """
        max_loglik = [item for sublist in self.log_likelihoods for item in sublist][-1]
        N = self.X[0].shape[0]
        dfmeans = len(np.unique(self.means_))
        dfvars = len(np.unique(self.covariances_[self.covariances_!=0]))
        dfclasses = self.pi_.shape[1]-1
        df = dfmeans + dfvars+ dfclasses
        return 2*df-2*max_loglik

    def CAIC(self):
        """
        Function to calculate the Consistent Akaike Information Criterion for the maximum likelihood solution
        Parameters: ozdogman, H. Model selection and Akaike’s information criterion (AIC): the
        general theory and its analytical extensions. Psychometrika 52, 345–370 (1987). 61. Schwartz, G. Estimating the dimension of a model. Ann. Stat. 6.
        ------------------
        self:
        A fitted Gaussian Mixture Model

        ----------------
        CAIC: the Consistent Akaike Information Criterion

        """
        max_loglik = [item for sublist in self.log_likelihoods for item in sublist][-1]
        N = self.X[0].shape[0]
        dfmeans = len(np.unique(self.means_))
        dfvars = len(np.unique(self.covariances_[self.covariances_!=0]))
        dfclasses = self.pi_.shape[1]-1
        df = dfmeans + dfvars+ dfclasses
        return -2*max_loglik + df*(np.log(N)+1)


    def SABIC(self):
        """
        Function to calculate the Sample Size Adjusted BIC for the maximum likelihood solution
        Sclove, S. L. Application of model-selection criteria to some problems with
        multivariate analysis. Psychometrika 52, 333–343 (1987).
        ------------------
        self:
        A fitted Gaussian Mixture Model

        ----------------
        SABIC: the sample size adjusted BIC

        """
        max_loglik = [item for sublist in self.log_likelihoods for item in sublist][-1]
        N = self.X[0].shape[0]
        dfmeans = len(np.unique(self.means_))
        dfvars = len(np.unique(self.covariances_[self.covariances_!=0]))
        dfclasses = self.pi_.shape[1]-1
        df = dfmeans + dfvars+ dfclasses
        return -2*max_loglik + df*(np.log((N+2)/24))