'''A self-contained class that estimates ranking from an adjacency matrix.
Scalar parameters are estimated with an unsophisticated (but effective) Metropolis-Hastings algorithm.
Rank orders are estimated with a combination of slice, leapfrog, and Metropolis-Hastings algorithms.
'''

import numpy as np 
import numba
import math
import random
import sqlite3

import sys,os,time

import ctypes
from multiprocessing import Process, Queue, Array


#######################
## Utility functions:
#######################
## Utility and combinitorial functions used by
## other functions.
#######################

@numba.jit(nopython=True)
def bounds_reflect(x,lower,upper):
    if lower > upper:
        return(x)
    if lower == upper:
        return(lower)
    while True:
        if x > upper:
            x = 2 * upper - x
        elif x < lower:
            x = 2 * lower - x
        else:
            return(x)
    return(x)

@numba.jit(nopython=True)
def step_size_update(accepted,preferred = 0.6):
    '''calculates a step size update'''
    rate = accepted.mean()

    veryLow = preferred*0.5
    low = preferred*0.8
    high = preferred + (1-preferred)*0.2
    veryHigh = preferred + (1-preferred)*0.5

    if rate < veryLow:
        return(0.1)
    elif rate < low:
        return(0.5)
    elif rate < high:
        return(1.0)
    elif rate < veryHigh:
        return(2.0)
    else:
        return(10.0)



def urandomInt():
    '''get a 24-bit random integer from OS's urandom'''
    b = os.urandom(3)
    i = b[0] + b[1]*256 + b[2]*256*256
    return(i)

@numba.jit(nopython=True)
def set_numba_rseed(rseed):
    '''compiled in nopython mode, so this
    function, when called from python, will
    set the random number seed for subsequent
    numba compiled function calls'''
    np.random.seed(rseed)

@numba.jit(nopython=True)
def cov(x,y):
    '''sample covariance between x and y'''
    n = len(x)
    if n < 2:
        return(0.0)
    Kx = x[0] # center on first values
    Ky = y[0]
    Ex = 0.0
    Ey = 0.0
    Exy = 0.0
    for i in range(n):
        Ex += x[i] - Kx
        Ey += y[i] - Ky
        Exy += (x[i] - Kx) * (y[i] - Ky)
    return((Exy - Ex*Ey/n)/n)

@numba.jit(nopython=True)
def cov_mat(X):
    '''calculate the covariance matrix of X, whose rows are samples.
    (there's probably a more efficient way to do it but this suffices)'''
    n = X.shape[1]
    res = np.empty((n,n),dtype=np.float64)
    for i in range(n):
        res[i,i] = cov(X[:,i],X[:,i])
        for j in range(i):
            res[i,j] = cov(X[:,i],X[:,j])
            res[j,i] = res[i,j]
    return(res)




#######################
## Distributions:
#######################
## Numba-enabled random generators and density functions.
#######################


@numba.jit(nopython=True)
def lddirichlet(x,alpha):
    '''dirichlet log density function'''
    logProdOfGammas = 0.0
    alpha0 = 0.0
    logNumerator = 0.0
    n = x.shape[0]
    for i in range(n):
        logProdOfGammas += math.lgamma(alpha[i])
        alpha0 += alpha[i]
        logNumerator += math.log(x[i])*(alpha[i]-1)
    return(logNumerator + math.lgamma(alpha0) - logProdOfGammas)

@numba.jit(nopython=True)
def lddirichlet_sym(x,alpha):
    '''same as lddirichlet, but expects a scalar for alpha'''
    logProdOfGammas = 0.0
    alpha0 = 0.0
    logNumerator = 0.0
    n = x.shape[0]
    for i in range(n):
        logProdOfGammas += math.lgamma(alpha)
        alpha0 += alpha
        logNumerator += math.log(x[i])*(alpha-1)
    return(logNumerator + math.log(math.gamma(alpha0)) - logProdOfGammas)

@numba.jit(nopython=True)
def rdirichlet_sym(n,alpha):
    '''sample from a symmetric dirichlet distribution of
    dimension n with concentration parameter alpha'''
    s = 0.0
    res = np.empty(n,dtype=np.float64)
    for i in range(n):
        res[i] = np.random.gamma(alpha,1.0)
        s += res[i]
    for i in range(n):
        res[i] /= s
    return(res)


@numba.jit(nopython=True)
def ldbeta(x,alpha,beta):
    '''log density of the beta distribution.
    (this is much faster than np.stats.beta.logpdf)'''
    if x<0.0 or x>1.0:
        return(-np.inf)
    dx = np.empty(2,dtype=np.float64)
    dx[0] = x
    dx[1] = 1-x
    da = np.empty(2,dtype=np.float64)
    da[0] = alpha
    da[1] = beta
    return(lddirichlet(dx,da))



@numba.jit(nopython=True)
def cholesky_decomp(A):
    '''Cholesky decomposition of a matrix A. No checks.'''
    L = np.zeros(A.shape,np.float64)
    for i in range(A.shape[0]):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s += L[i,k] * L[j,k]
            if i==j:
                L[i,j] = math.sqrt(A[i,i]-s)
            else:
                L[i,j] = (A[i,j] - s) / L[j,j]
    return(L)

@numba.jit(numba.float64[:](numba.float64[:],numba.float64[:,:]),nopython=True)
def rmvnorm(mu,sigma):
    '''random draw from multivariate normal distribution
    with mean mu and covariance sigma. No checks.'''
    A = cholesky_decomp(sigma)
    n = len(mu)
    z = np.empty(n,dtype=np.float64)
    for i in range(n):
        z[i] = np.random.normal()
    return(mu + np.dot(A,z))


####
## support functions working with orders
####

@numba.jit(nopython=True)
def reduceOrder(o,rot=0):
    '''reduce an order vector to a vector of the same length,
    but combinatorially 'collapsed'.
    if rot!=0, circularly rotate vector by rot before reducing.'''
    n = o.shape[0]
    res = np.empty(n,dtype=np.int64)
    if rot==0:
        for i in range(n):
            res[i]=o[i]
    else:
        for i in range(n):
            ind = (i+rot) % n
            res[i] = o[ind]
    for i in range(n-1):
        for j in range(i+1,n):
            if res[j]>res[i]:
                res[j] -= 1
    return(res)

@numba.jit(nopython=True)
def expandOrder(r,rot=0):
    '''undo reduceOrder.
    if rot!=0, 'unrotate' by rot after expanding'''
    n = r.shape[0]
    res = np.empty(n,dtype=np.int64)
    if rot==0:
        # fill res with contents of o
        for i in range(n):
            res[i]=r[i]
        # expand res
        for i in range(n-1,0,-1):
            for j in range(i,n):
                if res[j]>=res[i-1]:
                    res[j] += 1
    else:
        # fill res with contents of o
        for i in range(n):
            ind = (i-rot) % n
            res[i]=r[ind]
        # expand res
        for i in range(n-1,0,-1):
            ind = (i+rot) % n
            for j in range(i,n):
                jnd = (j+rot) % n
                if res[jnd]>=res[ind-1]:
                    res[jnd] += 1
    return(res)


####
## Likelihood functions
####

@numba.jit(nopython=True)
def priorProb(dProbs,pAlpha=1.0):
    '''prior probability of dProbs (=[pBase,pDown,pUp]).
    pAlpha is concentration parameter to the Beta distribution.'''
    if dProbs[1] > dProbs[2]:
        return(-np.inf)
    res = ldbeta(dProbs[0],pAlpha,pAlpha)
    res += ldbeta(dProbs[1],pAlpha,pAlpha)
    res += ldbeta(dProbs[2],pAlpha,pAlpha)
    return(res)


@numba.jit(nopython=True)
def fillEdgeLogLikMatrix(ell,dProbs):
    '''fills edge log-likelihood matrix `ell` in place.
    ell is of shape (n*2-1, 2), with ell[d,e] representing the
    log likelihood of am[i,j]==e with dist[i,j]==d'''
    n = (ell.shape[0] + 1)/2
    ell[0,0] = -np.inf
    ell[0,1] = -np.inf
    lowProb  = dProbs[0]
    highProb = dProbs[0]
    for d in range(1,n):
        lowProb  *= dProbs[1]
        ell[d,1] = math.log(lowProb)
        ell[d,0] = math.log(1 - lowProb)
        highProb *= dProbs[2]
        ell[-d,1] = math.log(highProb)
        ell[-d,0] = math.log(1 - highProb)

@numba.jit(nopython=True)
def orderLogLik(p,ell,am):
    '''log likelihood of just the order vector p using cached edge
    log likelihoods. (turns out to be conditional likelihood for 
    pAlpha = 1.0)'''
    n = len(p)
    res = 0.0
    n = am.shape[0]
    for i in range(n-1):
        for j in range(i+1,n):
            d = p[i]-p[j]
            res += ell[d,am[i,j]]
            res += ell[-d,am[j,i]]
    return(res)

@numba.jit(nopython=True)
def fullLogLik(p,ell,dProbs,am,pAlpha=1.0):
    '''full log likelihood of the model using cached edge log likelihoods'''
    res = priorProb(dProbs,pAlpha)
    # each edge
    n = am.shape[0]
    for i in range(n-1):
        for j in range(i+1,n):
            d = p[i]-p[j]
            res += ell[d,am[i,j]]
            res += ell[-d,am[j,i]]
    return(res)

@numba.jit(nopython=True)
def fullLogLik_complete(p,dProbs,am,pAlpha=1.0):
    ell = np.zeros((n*2-1,2),dtype=np.float64)
    fillEdgeLogLikMatrix(ell,dProbs)
    return(fullLogLik(p,ell,dProbs,am,pAlpha))




####
## MCMC step functions
####
@numba.jit(nopython=True)
def oneTowards(x,y):
    '''returns 1 if x<y or x==0, -1 otherwise'''
    if x < y or x == 0:
        return(1)
    else:
        return(-1)

@numba.jit(nopython=True)
def order_step_slice(p,ell,am,meandims=4.0,maxdims=20,switchiter=1000,maxiter=50000):
    '''a single slice step on p'''
    n = len(p)

    rot = np.random.randint(n)

    ll = orderLogLik(p,ell,am)

    # figure out the dimensions to use
    ndims = min(min(np.random.poisson(meandims-1)+1,maxdims),n)
    dims = np.random.choice(n,ndims,replace=False)
    dims.sort()
    dimSizes = n - dims

    q = reduceOrder(p,rot) # rotated, reduced order
    qprime = q.copy() # keep candidate points in here
    y = ll-np.random.exponential() # same as y = log(Uniform(exp(ll)))

    # define initial hyperrectangle
    upper = dimSizes.copy()
    lower = np.zeros(ndims,dtype=np.int64)

    # start looking for a candidate
    iter = 0
    while True:
        if iter > maxiter:
            return(p,0)
        iter += 1
        #print(iter)
        # pick a proposal
        for d in range(ndims):
            qprime[dims[d]] = np.random.randint(lower[d],upper[d])
        # evaluate
        pprime = expandOrder(qprime,rot)
        llprime = orderLogLik(pprime,ell,am)
        if llprime >= y:
            # accept!
            #print(ndims)
            return((pprime,iter))
            #break
        # bad proposal: need to crop the hyperrectangle
        if iter <= switchiter:
            # find the gradient to crop on
            maxdim = -1
            maxgrad = -np.inf
            for d in range(ndims):
                s = oneTowards(qprime[d],q[d])
                qprime[d] += s
                llgrad = orderLogLik(expandOrder(qprime,rot),ell,am)
                qprime[d] -= s
                grad = llgrad-llprime
                if grad > maxgrad:
                    maxgrad = grad
                    maxdim=d
            # crop it
            if qprime[dims[maxdim]] >= q[dims[maxdim]]:
                upper[maxdim] = qprime[dims[maxdim]]+1
            else:
                lower[maxdim] = qprime[dims[maxdim]]
        else:
            # this is taking too long, crop along every dimension
            for d in range(ndims):
                if qprime[dims[d]] >= q[dims[d]]:
                    upper[d] = qprime[dims[d]]+1
                else:
                    lower[d] = qprime[dims[d]]                

@numba.jit(nopython=True)
def order_step_leap(p,ell,am,hist):
    '''a step method for orders that proposes a 'leap' in a direction
    determined by the difference in two random rows of `hist`, which
    should be a (m,n)-shaped array whose rows represent likely values
    of the posterior (e.g. a pre-populated list of ML-estimates or a
    running tally from multiple parallel chains).'''
    n = len(p)
    m = hist.shape[0]

    ll = orderLogLik(p,ell,am)

    rot = np.random.randint(n)
    q = reduceOrder(p,rot) # rotated, reduced order

    # pick rows
    rows = np.random.choice(m,2,replace=False)

    # propose new q
    diff = reduceOrder(hist[rows[0]],rot) - reduceOrder(hist[rows[1]],rot)
    qprime = np.empty_like(q)
    for i in range(n):
        qprime[i] = bounds_reflect(q[i] + diff[i],0,n-i)
    pprime = expandOrder(qprime,rot)
    llprime = orderLogLik(pprime,ell,am)
    if -np.random.exponential() < (llprime - ll):
        # accept!
        return(pprime,1)
    else:
        # reject!
        return(p,0)

@numba.jit(nopython=True)
def dProbs_step_MH(p,ell,dProbs,am,pAlpha=1.0,stepSize=0.3):
    '''a simple Metropolis-Hastings step for dProbs.
    Spherical normal proposal (reflected).'''
    # current log lik
    ll = fullLogLik(p,ell,dProbs,am,pAlpha)
    # generate a proposal
    dProbsprime = np.empty(3,dtype=np.float64)
    for i in range(3):
        dProbsprime[i] = bounds_reflect(dProbs[i] + stepSize * np.random.normal(),0.0,1.0)
    # check
    pell = np.empty_like(ell)
    fillEdgeLogLikMatrix(pell,dProbsprime)
    pll = fullLogLik(p,pell,dProbsprime,am,pAlpha)

    thresh = pll - ll
    if -np.random.exponential() < thresh:
        # accept
        return(dProbsprime,1)
    else:
        # reject
        return(dProbs,0)

@numba.jit(nopython=True)
def dProbs_step_MH_cov(p,ell,dProbs,am,sigma,pAlpha=1.0,stepSize=0.3):
    '''a simple Metropolis-Hastings step for dProbs.
    Proposals are multivariate normal with covariance sigma, scaled by stepSize'''
    # current log lik
    ll = fullLogLik(p,ell,dProbs,am,pAlpha)
    # generate a proposal
    dProbsprime = np.empty(3,dtype=np.float64)
    jump = rmvnorm(np.zeros(3,dtype=np.float64),sigma)
    for i in range(3):
        dProbsprime[i] = bounds_reflect(dProbs[i] + stepSize * jump[i], 0.0, 1.0)
    # check
    pell = np.empty_like(ell)
    fillEdgeLogLikMatrix(pell,dProbsprime)
    pll = fullLogLik(p,pell,dProbsprime,am,pAlpha)

    thresh = pll - ll
    if -np.random.exponential() < thresh:
        # accept
        return(dProbsprime,1)
    else:
        # reject
        return(dProbs,0)

@numba.jit(nopython=True)
def dProbs_step_MH_fromprior(p,ell,dProbs,am,pAlpha=1.0):
    '''a simple Metropolis-Hastings step for dProbs.
    Proposals are straightforward draws from prior.'''
    # current log lik
    ll = fullLogLik(p,ell,dProbs,am,pAlpha)
    # generate a proposal
    dProbsprime = rdirichlet_sym(3,pAlpha)
    while dProbsprime[1]>dProbsprime[2]:
        dProbsprime = rdirichlet_sym(3,pAlpha)

    # check
    pell = np.empty_like(ell)
    fillEdgeLogLikMatrix(pell,dProbsprime)
    pll = fullLogLik(p,pell,dProbsprime,am,pAlpha)

    lpforward = lddirichlet_sym(dProbsprime,pAlpha) 
    lpbackward = lddirichlet_sym(dProbs,pAlpha) 

    thresh = (pll + lpforward) - (ll + lpbackward)
    if -np.random.exponential() < thresh:
        # accept
        return(dProbsprime,1)
    else:
        # reject
        return(dProbs,0)


class OrderEstimator(object):
    '''a class for estimating rank orderings from a binary adjacency matrix'''

    def __init__(self,am,pAlpha=1.0):
        self.am = am
        self.n = am.shape[0]
        self.pAlpha = pAlpha

        # dict storing all the traces
        self.traces = {}
        self.tnum = 0

    def sample_mcmc_simple(self,niter,burnin=0,thin=1,verbose=False):
        # initialize parameters
        am = self.am
        n = self.n
        p = np.arange(n,dtype=np.int64)
        np.random.shuffle(p)
        dProbs = np.array([am.mean()*5,.85,.95])
        ell = np.zeros((n*2-1,2),dtype=np.float64)
        fillEdgeLogLikMatrix(ell,dProbs)

        # set up a trace list
        cnum = self.tnum
        self.traces[cnum] = {'perms':[],'dProbs':[]}
        self.tnum += 1

        for i in range(niter):
            for j in range(5):
                p,_ = order_step_slice(p,ell,am)
            dProbs,_ = dProbs_step_MH(p,ell,dProbs,am,stepSize=.03)
            if i%thin == 0:
                if verbose:
                    print('%d: (%d) %.3f | %3f %3f %3f' % ((os.getpid(),i,orderLogLik(p,ell,am))+tuple(dProbs)))
                if i>burnin:
                    self.traces[cnum]['perms'].append(p.copy())
                    self.traces[cnum]['dProbs'].append(dProbs.copy())

    def parallel_trace_recorder(self,db,queue,updateEvery,nChains,dProbs_cov,verbose,logFileName):
        '''a function that listens for trace rows on `queue` and
        writes them to the database `db` (creating tables as necessary).
        It also updates the shared array dProbs_cov every `updateEvery`
        rows that it writes.
        It will quit after it recieves `nChains` None objects from the queue.'''
        if logFileName is not None:
            logFile = open(logFileName,'a',1)

        n = self.n
        con = sqlite3.connect(db,timeout=20.0)
        self.traces[self.tnum] = con
        self.tnum += 1
        # make tables
        if verbose:
            print("%d: creating database and tables" % os.getpid())
            if logFileName is not None:
                logFile.write("%d: creating database and tables\n" % os.getpid())
                logFile.flush()
        cur = con.cursor()
        cur.execute('''
            create table if not exists deviance (
                chain int, iter int, deviance float)''')
        cur.execute('''
            create table if not exists dProbs (
                chain int, iter int,
                pBase float, pLow float, pHigh float)''')
        cur.execute('''
            create table if not exists perm (
                chain int, iter int,
                %s)''' % ','.join([('p%d int' % i) for i in range(n)]))
        cur.execute('''
            create table if not exists dProbs_cov(
                chain int, iter int,
                c00 float, c01 float, c02 float,
                           c11 float, c12 float,
                                      c22 float
            )''')
        con.commit()
        cur.close()
        noneCount = 0
        devianceRows = []
        dProbsRows = []
        permRows = []
        dProbs_covRows = []
        while noneCount < nChains:
            emission = queue.get()
            if emission is None:
                noneCount += 1
                if verbose:
                    print('%d: %d of %d chains finished' % (os.getpid(),noneCount,nChains))
                    if logFileName is not None:
                        logFile.write('%d: %d of %d chains finished\n' % (os.getpid(),noneCount,nChains))
                        logFile.flush()
                continue
            devianceRows.append(emission[0])
            dProbsRows.append(emission[1])
            permRows.append(emission[2])
            dProbs_covRows.append(emission[0][:2] + (dProbs_cov[0],dProbs_cov[1],dProbs_cov[2],dProbs_cov[3],dProbs_cov[4],dProbs_cov[8]))
            if len(devianceRows) >= updateEvery:
                # update covariance
                dProbsTrace = np.array([dp[2:] for dp in dProbsRows])
                with dProbs_cov.get_lock():
                    arr = np.frombuffer(dProbs_cov.get_obj()).reshape((3,3))
                    arr[:,:] = cov_mat(dProbsTrace)[:,:]
                # write everything to disk
                if verbose:
                    print('%d: writing %d rows to disk' % (os.getpid(),len(devianceRows)))
                    if logFileName is not None:
                        logFile.write('%d: writing %d rows to disk\n' % (os.getpid(),len(devianceRows)))
                        logFile.flush()
                cur = con.cursor()
                cur.executemany('insert into deviance values (?,?,?)', devianceRows)
                cur.executemany('insert into dProbs values (?,?,?,?,?)', dProbsRows)
                cur.executemany('insert into perm values (%s)' % ','.join(['?']*(n+2)), permRows)
                cur.executemany('insert into dProbs_cov values (?,?,?,?,?,?,?,?)', dProbs_covRows)
                con.commit()
                cur.close()
                devianceRows = []
                dProbsRows = []
                permRows = []

        if verbose:
            print('%d: writing FINAL %d rows to disk' % (os.getpid(),len(devianceRows)))
            if logFileName is not None:
                logFile.write('%d: writing FINAL %d rows to disk\n' % (os.getpid(),len(devianceRows)))
                logFile.flush()
        if len(devianceRows)>0:
            # write everything to disk
            cur = con.cursor()
            cur.executemany('insert into deviance values (?,?,?)', devianceRows)
            cur.executemany('insert into dProbs values (?,?,?,?,?)', dProbsRows)
            cur.executemany('insert into perm values (%s)' % ','.join(['?']*(n+2)), permRows)
            cur.executemany('insert into dProbs_cov values (?,?,?,?,?,?,?,?)', dProbs_covRows)
            con.commit()
            cur.close()
        return(True)

    def parallel_mcmc_sampler(self,queue,niter,burnin_isolated,burnin_shared,thin,record_throughout,updateEvery,nChains,chainNum,hist,dProbs_cov,verbose,logFileName):
        '''a sampler to run on a single process, dispatched by sample_mcmc_parallel'''
        if logFileName is not None:
            logFile = open(logFileName,'a',1)

        n = self.n
        hist_size = int(len(hist)/n/nChains)
        hist_frozen = None
        dProbs_cov_frozen = None
        
        # make sure all the random states are unique for this process
        np.random.seed(np.absolute(urandomInt()-os.getpid()))
        set_numba_rseed(np.absolute(urandomInt()-os.getpid()))

        # get the state
        am = self.am
        pAlpha = self.pAlpha
        p = np.arange(n,dtype=np.int64)
        np.random.shuffle(p)
        dProbs = np.array([am.mean()*5, 0.85, 0.95])
        dProbs += np.random.uniform(-0.01, 0.01, 3) # randomize just a tiny bit
        dProbs = np.absolute(dProbs) # just in case?
        ell = np.zeros((n*2-1,2),dtype=np.float64)
        fillEdgeLogLikMatrix(ell,dProbs)

        # initialize step sizes
        order_ssize_slice = 5.0
        order_ssize_slice_hist = np.zeros(updateEvery*5,dtype=np.int64)
        order_ssize_leap = 100
        order_ssize_leap_hist = np.zeros(updateEvery,dtype=np.int64)
        dProbs_ssize_MH = 0.1
        dProbs_ssize_MH_hist = np.zeros(updateEvery,dtype=np.int64)

        # iterate
        histIter = 0
        for i in range(niter):
            # some order slice steps
            for j in range(5):
                p,c = order_step_slice(p,ell,am,meandims=order_ssize_slice,maxdims=30)
                order_ssize_slice_hist[((i*5)+j) % (updateEvery*5)] = c 
            # order leap
            if i > burnin_isolated:
                succ_leaps = 0
                if i <= burnin_shared + burnin_isolated:
                    # leaps from dynamic history
                    for j in range(order_ssize_leap):
                        p,c = order_step_leap(p,ell,am,np.frombuffer(hist.get_obj(),dtype=np.int64).reshape((hist_size*nChains,n)))
                        if c>0:
                            succ_leaps += 1
                            if verbose:
                                print('%d: successful leap' % os.getpid())
                                if logFileName is not None:
                                    logFile.write('%d: successful leap\n' % os.getpid())
                                    logFile.flush()

                else:
                    # create frozen history if this is first time through
                    if hist_frozen is None:
                        hist_frozen = np.frombuffer(hist.get_obj(),dtype=np.int64).reshape((hist_size*nChains,n)).copy()
                    # leap from frozen history
                    for j in range(order_ssize_leap):
                        p,c = order_step_leap(p,ell,am,hist_frozen)
                        if c>0:
                            succ_leaps += 1
                            if verbose:
                                print('%d: successful leap' % os.getpid())
                                if logFileName is not None:
                                    logFile.write('%d: successful leap\n' % os.getpid())
                                    logFile.flush()
                order_ssize_leap_hist[i % updateEvery] = succ_leaps

            # dProbs MH steps
            if i <= burnin_isolated+burnin_shared:
                dProbs,c = dProbs_step_MH_cov(p,ell,dProbs,am,np.frombuffer(dProbs_cov.get_obj()).reshape((3,3)),pAlpha,stepSize=dProbs_ssize_MH)
            else:
                if dProbs_cov_frozen is None:
                    dProbs_cov_frozen = np.frombuffer(dProbs_cov.get_obj()).reshape((3,3)).copy()
                dProbs,c = dProbs_step_MH_cov(p,ell,dProbs,am,dProbs_cov_frozen,pAlpha,stepSize=dProbs_ssize_MH)
            dProbs_ssize_MH_hist[i % updateEvery] = c
            dProbs,c = dProbs_step_MH_fromprior(p,ell,dProbs,am,pAlpha)

            # update step sizes
            if (i <= burnin_isolated+burnin_shared) and (i % updateEvery == 0) and (i>0):
                order_ssize_slice = 1 + (order_ssize_slice-1)/step_size_update(order_ssize_slice_hist/1000.,.05)
                order_ssize_slice = max(min(order_ssize_slice,21.0),1.5)
                if i > burnin_isolated:
                    order_ssize_leap = int(1 + (order_ssize_leap-1)/step_size_update(order_ssize_leap_hist,.05))
                    order_ssize_leap = max(min(order_ssize_leap,800),2)
                dProbs_ssize_MH *= step_size_update(dProbs_ssize_MH_hist)
                dProbs_ssize_MH = max(min(dProbs_ssize_MH,3.0),0.0001)

            # record to hist and emit
            if i % thin == 0:
                ll = fullLogLik(p,ell,dProbs,am,pAlpha)
                # update hist
                # every chain gets a contiguous block of hist_size rows
                histRow = (hist_size * chainNum) + (histIter % hist_size)
                with hist.get_lock():
                    hist_np = np.frombuffer(hist.get_obj(),dtype=np.int64).reshape((hist_size*nChains,n))
                    hist_np[histRow,:] = p
                histIter += 1
                # queue trace rows
                if record_throughout or (i > burnin_isolated + burnin_shared):
                    devianceRow = (chainNum,i,-2*ll)
                    dProbsRow = (chainNum,i) + tuple(dProbs)
                    permRow = (chainNum,i) + tuple([int(x) for x in p])
                    #if verbose:
                    #    print('%d: Queuing 1 row' % os.getpid())
                    queue.put((devianceRow,dProbsRow,permRow))
                # print info
                if verbose:
                    print('%d: (iter %d) %.3f | %.3f %.3f %.3f | (%.3f, %d, %.4f)' % ((os.getpid(),i,ll) + tuple(dProbs) + (order_ssize_slice,order_ssize_leap,dProbs_ssize_MH)))
                    if logFileName is not None:
                        logFile.write('%d: (%d) %.3f | %.3f %.3f %.3f | (%.3f, %d, %.4f)\n' % ((os.getpid(),i,ll) + tuple(dProbs) + (order_ssize_slice,order_ssize_leap,dProbs_ssize_MH)))
                        logFile.flush()
        queue.put(None)

    def sample_mcmc_parallel(self,niter,burnin_isolated=0,burnin_shared=0,thin=1,record_throughout=False,updateEvery=50,hist_size=50,nChains=20,db=':memory:',verbose=False,logFileName=None):
        n = self.n

        # shared memory objects
        dProbs_cov = Array(ctypes.c_double,9)
        with dProbs_cov.get_lock():
            arr = np.frombuffer(dProbs_cov.get_obj()).reshape((3,3))
            # initialize to identity matrix
            arr[:,:] = np.array([[1,0,0],[0,1,0],[0,0,1]])[:,:]
        nHist = hist_size*nChains
        hist = Array(ctypes.c_int64,n*nHist)
        with hist.get_lock():
            hist_np = np.frombuffer(hist.get_obj(),dtype=np.int64).reshape((nHist,n))
            for i in range(nHist):
                hist_np[i,:] = np.arange(n)
                np.random.shuffle(hist_np[i,:])
        rowQueue = Queue()

        # start the chains
        processes = []
        for pi in range(nChains):
            print('Starting process %d' %pi)
            processes.append(Process(target=self.parallel_mcmc_sampler,kwargs = {
                'queue':rowQueue,
                'niter':niter,'burnin_isolated':burnin_isolated,'burnin_shared':burnin_shared,
                'thin':thin,'record_throughout':record_throughout,'updateEvery':updateEvery,
                'nChains':nChains,'chainNum':pi,
                'hist':hist,'dProbs_cov':dProbs_cov,
                'verbose':verbose,'logFileName':logFileName
            }))
            processes[-1].start()
            time.sleep(0.3)

        # add some stuff to self for later access
        self.last_queue = rowQueue
        self.last_hist = hist
        self.last_dProbs_cov = dProbs_cov
        self.last_processes = processes

        # start the recorder
        self.parallel_trace_recorder(db,rowQueue,updateEvery=updateEvery,nChains=nChains,dProbs_cov=dProbs_cov,verbose=verbose,logFileName=logFileName)




