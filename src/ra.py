import logging
import pickle
import sys
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator as RGI

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])

class RoboAdvisor:
    """
    The following parameters are used to specify the market environment and the client's risk aversion process.
    b_r: risk free rate, function from market state to return of risk-free asset B
    s_mu: rate of risky asset, function from market state to mean of risky asset S
    s_sig: volatility of risky asset, function from market state to standard deviation of risky asset S
    P: transition matrix of market state, 2x2 numpy array
    alpha: time discount factor for risk aversion
    sig_eps: magnitude of idiosyncratic shocks to risk aversion
    p_eps: probability of idiosyncratic shocks to risk aversion
    beta: behavioral bias factor describing the client's trend-chasing bias
    T: max time, n = 0, 1, 2, ..., T-1
    """
    def __init__(self, b_r, s_mu, s_sig, P, alpha, sig_eps, p_eps, beta, T, phi):

        self.b_r = b_r
        self.s_mu = s_mu
        self.s_sig = s_sig
        self.P = P
        self.alpha = alpha
        self.sig_eps = sig_eps
        self.p_eps = p_eps
        self.beta = beta
        self.T = T
        self.phi = phi
        self.tau = np.zeros(T, dtype=int)
        t = np.array(range(0, T, phi)) # interaction timestamps
        for i in t:
            self.tau[i:] = i

    # gamma_y depends on the sharpe ratio of the market
    def calc_gamma_y(self, y):
        return np.exp((self.s_mu[y] - self.b_r[y]) / self.s_sig[y])

    # eta is the time discount factor, alpha is the rate of time discount   
    def calc_eta(self, t):
        return np.exp(-self.alpha * (self.T-t))

    # Backwards induction, to find the optimal strategy given step number n and state history d.
    # Given the same market dymanics and risk aversion process, this is fully deterministic.
    # Therefore, we can precompute the optimal strategy for all possible state histories and step numbers. 
    # In actual operations, we can look up the optimal strategy from the precomputed table.
    # Note that we use approximation (to avoid integration over z) to speed up computation.
    def backwards_induction(self, verbose=True):
        # calculate the phi-fold convolution pdf for epsilon
        eps_space, eps_step = np.linspace(start=-2, stop=2, num=65, retstep=True, endpoint=True)
        pmf = scipy.stats.norm(loc=0, scale=self.sig_eps).pdf(eps_space) * eps_step
        pmf = pmf * self.p_eps
        pmf[eps_space.shape[0]//2] += 1-self.p_eps
        conv_pmf = np.copy(pmf)
        for _ in range(self.phi-1): # phi-fold convolution
            conv_pmf = scipy.signal.fftconvolve(conv_pmf,pmf,'same')

        # for any state d and number n, we calculate pi_n(d)
        delta_xi = 0.5
        delta_z = 0.1
        Y0_space = np.array([1,2])  # the economy state at the previous interaction time tau_n
        Y_space = np.array([1,2])   # the current economy state
        Z0_space = np.arange(-0.5, 0.5+delta_z, delta_z) # the sum of excess market returns between the two most recent interaction times, used to calculate xi2 given xi1
        Z1_space = np.arange(-0.5, 0.5+delta_z, delta_z) # Z is for the sum of excess market returns since the most recent interaction time, used to calculate xi2
        xi_space = np.arange(1e-5, 10, delta_xi) # most recent communicated risk aversion
        
        self.a = [None] * (self.T+1)
        self.b = [None] * (self.T+1)
        self.mu_a = [None] * self.T
        self.mu_az = [None] * self.T
        self.mu_b = [None] * self.T
        self.mu_bz = [None] * self.T
        self.mu_bz2 = [None] * self.T
        self.pi_d = [None] * self.T

        self.a[self.T] = RGI((Y0_space, Y_space, Z0_space, Z1_space, xi_space), np.ones((2, 2, Z0_space.size, Z1_space.size, xi_space.size)), bounds_error=False, fill_value=None)
        self.b[self.T] = RGI((Y0_space, Y_space, Z0_space, Z1_space, xi_space), np.ones((2, 2, Z0_space.size, Z1_space.size, xi_space.size)), bounds_error=False, fill_value=None)

        if verbose:
            logging.info(f"Start experiments")

        for n in reversed(range(self.T)):
            if verbose:
                logging.info(f"Backwards induction: calculating n = {n}...")
            def calc_mu_a(y0, y1, z0, z1, xi1):
                if n+1 == self.T:
                    res = 1.0
                elif self.tau[n+1] < n+1:
                    res = sum([self.P[y1-1, y2-1] * self.a[n+1]((y0, y2, z0, z1, xi1)) for y2 in [1,2]])
                elif self.tau[n+1] == n+1: # if n+1 is interaction time
                    # now we need to calculate xi2 based on xi1, Y, z1 (sum of excess market returns since last interaction time) and eps
                    func = lambda y2, i, eps2: self.a[n+1]((y2, y2, z1, 0.0, xi1 * np.exp(self.beta / self.phi * (z0 - z1)) * np.exp(eps2 + self.alpha * self.phi) * self.calc_gamma_y(y2)/self.calc_gamma_y(y0))) * conv_pmf[i]
                    res = sum([self.P[y1-1, y2-1] * sum([func(y2, i, eps) for i, eps in enumerate(eps_space)]) for y2 in [1,2]])
                return res

            def calc_mu_az(y0, y1, z0, z1, xi1):
                if n+1 == self.T:
                    res = self.s_mu[y1] - self.b_r[y1]
                elif self.tau[n+1] < n+1:
                    res = sum([self.P[y1-1, y2-1] * self.a[n+1]((y0, y2, z0, z1, xi1)) * (self.s_mu[y1]-self.b_r[y1]) for y2 in [1,2]])
                elif self.tau[n+1] == n+1:
                    gamma_c_tau_n = xi1 / np.exp(-self.beta * z0 / self.phi)
                    func = lambda y2, i, eps2: self.a[n+1]((y2, y2, z1, 0.0, gamma_c_tau_n * np.exp(eps2 + self.alpha * self.phi) * np.exp(-self.beta * z1 / self.phi) * self.calc_gamma_y(y2)/self.calc_gamma_y(y0))) * (self.s_mu[y1]-self.b_r[y1]) * conv_pmf[i]
                    res = sum([self.P[y1-1, y2-1] * sum([func(y2, i, eps) for i, eps in enumerate(eps_space)]) for y2 in [1,2]])
                return res
            
            def calc_mu_bz(y0, y1, z0, z1, xi1):
                if n+1 == self.T:
                    res = self.s_mu[y1]-self.b_r[y1]
                elif self.tau[n+1] < n+1:
                    res = sum([self.P[y1-1, y2-1] * self.b[n+1]((y0, y2, z0, z1, xi1)) * (self.s_mu[y1]-self.b_r[y1]) for y2 in [1,2]])
                elif self.tau[n+1] == n+1:
                    gamma_c_tau_n = xi1 / np.exp(-self.beta * z0 / self.phi)
                    func = lambda y2, i, eps2: self.b[n+1]((y2, y2, z1, 0.0, gamma_c_tau_n * np.exp(eps2 + self.alpha * self.phi) * np.exp(-self.beta * z1 / self.phi) * self.calc_gamma_y(y2)/self.calc_gamma_y(y0))) * (self.s_mu[y1]-self.b_r[y1]) * conv_pmf[i]
                    res = sum([self.P[y1-1, y2-1] * sum([func(y2, i, eps) for i, eps in enumerate(eps_space)]) for y2 in [1,2]])
                return res
            
            def calc_mu_bz2(y0, y1, z0, z1, xi1):
                if n+1 == self.T:
                    res = (self.s_mu[y1]-self.b_r[y1])**2 + self.s_sig[y1]**2
                elif self.tau[n+1] < n+1:
                    res = sum([self.P[y1-1, y2-1] * self.b[n+1]((y0, y2, z0, z1, xi1)) * ((self.s_mu[y1]-self.b_r[y1])**2 + self.s_sig[y1]**2) for y2 in [1,2]])
                elif self.tau[n+1] == n+1:
                    gamma_c_tau_n = xi1 / np.exp(-self.beta * z0 / self.phi)
                    func = lambda y2, i, eps2: self.b[n+1]((y2, y2, z1, 0.0, gamma_c_tau_n * np.exp(eps2 + self.alpha * self.phi) * np.exp(-self.beta * z1 / self.phi) * self.calc_gamma_y(y2)/self.calc_gamma_y(y0))) * ((self.s_mu[y1]-self.b_r[y1])**2 + self.s_sig[y1]**2) * conv_pmf[i]
                    res = sum([self.P[y1-1, y2-1] * sum([func(y2, i, eps) for i, eps in enumerate(eps_space)]) for y2 in [1,2]])
                return res

            def calc_pi(y0, y1, z0, z1, xi1):
                gamma_n = self.calc_eta(n)/self.calc_eta(self.tau[n]) * xi1 * self.calc_gamma_y(y1) / self.calc_gamma_y(y0)
                pi_res = 1/(gamma_n) * (self.mu_az[n]((y0, y1, z0, z1, xi1)) - (1+self.b_r[y1]) * gamma_n * (self.mu_bz[n]((y0, y1, z0, z1, xi1)) - self.mu_a[n]((y0, y1, z0, z1, xi1))*self.mu_az[n]((y0, y1, z0, z1, xi1))))/(self.mu_bz2[n]((y0, y1, z0, z1, xi1)) - (self.mu_az[n]((y0, y1, z0, z1, xi1)))**2)
                if pi_res < 0.0:
                    return 0.0
                elif pi_res > 1.0:
                    return 1.0
                else:
                    return pi_res
            
            def calc_a(y0, y1, z0, z1, xi1):
                if n+1 == self.T:
                    res = (1+self.b_r[y1]) + (self.s_mu[y1]-self.b_r[y1]) * self.pi_d[n]((y0, y1, z0, z1, xi1))
                elif self.tau[n+1] < n+1:
                    res = sum([self.P[y1-1, y2-1] * ((1+self.b_r[y1]) + (self.s_mu[y1]-self.b_r[y1]) * self.pi_d[n]((y0, y1, z0, z1, xi1))) * self.a[n+1]((y0, y2, z0, z1, xi1)) for y2 in [1,2]])
                elif self.tau[n+1] == n+1:
                    gamma_c_tau_n = xi1 / np.exp(-self.beta * z0 / self.phi)
                    func = lambda y2, i, eps2: ((1+self.b_r[y1]) + (self.s_mu[y1]-self.b_r[y1]) * self.pi_d[n]((y0, y1, z0, z1, xi1))) * self.a[n+1]((y2, y2, z1, 0.0, gamma_c_tau_n * np.exp(eps2 + self.alpha * self.phi) * np.exp(-self.beta * z1 / self.phi) * self.calc_gamma_y(y2)/self.calc_gamma_y(y0))) * conv_pmf[i]
                    res = sum([self.P[y1-1, y2-1] * sum([func(y2, i, eps) for i, eps in enumerate(eps_space)]) for y2 in [1,2]])
                return res
            
            def calc_b(y0, y1, z0, z1, xi1):
                if n+1 == self.T:
                    res = self.pi_d[n]((y0, y1, z0, z1, xi1))**2 * self.s_sig[y1]**2 + ((1+self.b_r[y1]) + (self.s_mu[y1]-self.b_r[y1]) * self.pi_d[n]((y0, y1, z0, z1, xi1)))**2
                elif self.tau[n+1] < n+1:
                    res = sum([self.P[y1-1, y2-1] * (self.pi_d[n]((y0, y1, z0, z1, xi1))**2 * self.s_sig[y1]**2 + ((1+self.b_r[y1]) + (self.s_mu[y1]-self.b_r[y1]) * self.pi_d[n]((y0, y1, z0, z1, xi1)))**2) * self.b[n+1]((y0, y2, z0, z1, xi1)) for y2 in [1,2]])
                elif self.tau[n+1] == n+1:
                    gamma_c_tau_n = xi1 / np.exp(-self.beta * z0 / self.phi)
                    func = lambda y2, i, eps2: (self.pi_d[n]((y0, y1, z0, z1, xi1))**2 * self.s_sig[y1]**2 + ((1+self.b_r[y1]) + (self.s_mu[y1]-self.b_r[y1]) * self.pi_d[n]((y0, y1, z0, z1, xi1)))**2) * self.b[n+1]((y2, y2, z1, 0.0, gamma_c_tau_n * np.exp(eps2 + self.alpha * self.phi) * np.exp(-self.beta * z1 / self.phi) * self.calc_gamma_y(y2)/self.calc_gamma_y(y0))) * conv_pmf[i]
                    res = sum([self.P[y1-1, y2-1] * sum([func(y2, i, eps) for i, eps in enumerate(eps_space)]) for y2 in [1,2]])
                return res
            
            self.mu_a[n] = RGI((Y0_space, Y_space, Z0_space, Z1_space, xi_space), np.array([calc_mu_a(i,j,k,l,m) for i in Y0_space for j in Y_space for k in Z0_space for l in Z1_space for m in xi_space]).reshape(len(Y0_space), len(Y_space), len(Z0_space), len(Z1_space), len(xi_space)), bounds_error=False, fill_value=None)
            self.mu_az[n] = RGI((Y0_space, Y_space, Z0_space, Z1_space, xi_space), np.array([calc_mu_az(i,j,k,l,m) for i in Y0_space for j in Y_space for k in Z0_space for l in Z1_space for m in xi_space]).reshape(len(Y0_space), len(Y_space), len(Z0_space), len(Z1_space), len(xi_space)), bounds_error=False, fill_value=None)
            self.mu_bz[n] = RGI((Y0_space, Y_space, Z0_space, Z1_space, xi_space), np.array([calc_mu_bz(i,j,k,l,m) for i in Y0_space for j in Y_space for k in Z0_space for l in Z1_space for m in xi_space]).reshape(len(Y0_space), len(Y_space), len(Z0_space), len(Z1_space), len(xi_space)), bounds_error=False, fill_value=None)
            self.mu_bz2[n] = RGI((Y0_space, Y_space, Z0_space, Z1_space, xi_space), np.array([calc_mu_bz2(i,j,k,l,m) for i in Y0_space for j in Y_space for k in Z0_space for l in Z1_space for m in xi_space]).reshape(len(Y0_space), len(Y_space), len(Z0_space), len(Z1_space), len(xi_space)), bounds_error=False, fill_value=None)
            self.pi_d[n] = RGI((Y0_space, Y_space, Z0_space, Z1_space, xi_space), np.array([calc_pi(i,j,k,l,m) for i in Y0_space for j in Y_space for k in Z0_space for l in Z1_space for m in xi_space]).reshape(len(Y0_space), len(Y_space), len(Z0_space), len(Z1_space), len(xi_space)), bounds_error=False, fill_value=None)
            self.a[n] = RGI((Y0_space, Y_space, Z0_space, Z1_space, xi_space), np.array([calc_a(i,j,k,l,m) for i in Y0_space for j in Y_space for k in Z0_space for l in Z1_space for m in xi_space]).reshape(len(Y0_space), len(Y_space), len(Z0_space), len(Z1_space), len(xi_space)), bounds_error=False, fill_value=None)
            self.b[n] = RGI((Y0_space, Y_space, Z0_space, Z1_space, xi_space), np.array([calc_b(i,j,k,l,m) for i in Y0_space for j in Y_space for k in Z0_space for l in Z1_space for m in xi_space]).reshape(len(Y0_space), len(Y_space), len(Z0_space), len(Z1_space), len(xi_space)), bounds_error=False, fill_value=None)
    

    """
    x0: starting wealth of the client
    y0: initial market state, either 1 or 2
    gamma_0: client's initial risk aversion
    calc_policy: whether to calculate the optimal policy
    verbose: whether to print the progress
    """
    def run(self, x0, y0, gamma_0, calc_policy=True, custom_schedule=None, verbose=True):
        if custom_schedule is not None and calc_policy:
            raise NotImplementedError("Backwards induction only compatible with uniform schedule for now.")

        if custom_schedule is not None:
            self.tau = custom_schedule

        self.x0 = x0
        self.y0 = y0
        self.gamma_0 = gamma_0

        self.Y = np.zeros(self.T, dtype=int)

        self.Z = np.zeros(self.T + 1)       # Z_0 is not defined, start from Z_1
        self.mu = np.zeros(self.T + 1)      # mu[i] is the mean of Z[i], determined by y[i-1]
        self.sig = np.zeros(self.T + 1)     # sig[i] is the std of Z[i], determined by y[i-1]
        self.r = np.zeros(self.T + 1)       # r[i] is the risk-free rate at time i, determined by y[i-1]
        self.R = np.zeros(self.T + 1)       # R[i]=1+r[i]

        self.X = np.zeros(self.T + 1)       # client's wealth
        self.pi = np.zeros(self.T)          # client's policy
        self.pi_tilde = np.zeros(self.T)    # client's policy (0 to 1)

        self.gamma_Y = np.zeros(self.T)
        self.eta = np.zeros(self.T)
        self.gamma_C = np.zeros(self.T)
        self.eps = np.ones(self.T + 1)
        self.gamma_id = np.zeros(self.T)

        self.gamma_Z = np.zeros(self.T) # behavior bias at interaction time, only defined at interaction times
        self.xi = np.zeros(self.T)      # communicated risk aversion
        self.gamma = np.zeros(self.T)   # robo-advisor's model of risk aversion

        for i in range(self.T):
            # market dynamics
            if i == 0:
                self.Y[i] = self.y0
                self.X[i] = self.x0
                self.mu[i+1] = self.s_mu[self.Y[i]]
                self.sig[i+1] = self.s_sig[self.Y[i]]
                self.r[i+1] = self.b_r[self.Y[i]]
                self.R[i+1] = 1 + self.r[i+1]
            else:
                # first generate the market state
                self.Y[i] = np.random.choice([1,2], size=1, p=[self.P[self.Y[i-1]-1,0], self.P[self.Y[i-1]-1,1]])
                self.mu[i+1] = self.s_mu[self.Y[i]]
                self.sig[i+1] = self.s_sig[self.Y[i]]
                self.r[i+1] = self.b_r[self.Y[i]]
                self.R[i+1] = 1 + self.r[i+1]

                # now generate the risky asset's price
                self.Z[i] = np.random.normal(loc=self.mu[i], scale=self.sig[i])
                # and the client's wealth
                self.X[i] = self.R[i] * self.X[i-1] + (self.Z[i] - self.r[i]) * self.pi[i-1]

            # then generate the client's risk aversion process and the communicated risk aversion
            self.gamma_Y[i] = self.calc_gamma_y(self.Y[i])
            self.eta[i] = self.calc_eta(i)
            if i > 0:
                self.eps[i] = np.random.choice([np.random.normal(loc=0, scale=self.sig_eps), 0], size=1, p=[self.p_eps, 1-self.p_eps])
                self.gamma_id[i] = self.gamma_id[i-1] * np.exp(self.eps[i])
                self.gamma_C[i] = self.gamma_Y[i] * self.eta[i] * self.gamma_id[i]
                
                if self.tau[i] == i: # interaction time
                    self.gamma_Z[i] = np.exp(-self.beta/(i-self.tau[i-1]) * sum([self.Z[k+1] - self.mu[k+1] for k in range(self.tau[i-1], i)]))
                    self.xi[i] = self.gamma_C[i] * self.gamma_Z[i]
                else:
                    self.xi[i] = self.xi[self.tau[i]]

            else:
                self.gamma_C[i] = self.gamma_0
                self.gamma_id[i] = self.gamma_C[i] / self.gamma_Y[i] / self.eta[i]

                self.gamma_Z[i] = 1 # no bias when initialized because no trend to chase at all
                self.xi[i] = self.gamma_C[i]
            
            # robot advisor's model of risk aversion
            self.gamma[i] = self.eta[i]/self.eta[self.tau[i]] * self.xi[i] * self.gamma_Y[i] / self.gamma_Y[self.tau[i]]

            if verbose:
                logging.info(f"Step {i}: look up in policy table for (Y0={self.Y[self.tau[i]]}, Y1={self.Y[i]}), Z0={sum([self.Z[k+1] - self.mu[k+1] for k in range(self.tau[i]-self.phi, self.tau[i])])}, Z1={sum([self.Z[k+1] - self.mu[k+1] for k in range(self.tau[i], i)])}, xi1={self.xi[i]}.")
            if calc_policy:
                self.pi_tilde[i] = self.pi_d[i]((
                    self.Y[self.tau[i]],
                    self.Y[i],
                    sum([self.Z[k+1] - self.mu[k+1] for k in range(self.tau[i]-self.phi, self.tau[i])]),
                    sum([self.Z[k+1] - self.mu[k+1] for k in range(self.tau[i], i)]),
                    self.xi[i]
                    ))
                if self.pi_tilde[i] < 0:
                    self.pi_tilde[i] = 0.0
                elif self.pi_tilde[i] > 1:
                    self.pi_tilde[i] = 1.0
                    
                self.pi[i] = self.pi_tilde[i] * self.X[i]
            else:
                self.pi[i] = 0.0
            
            if verbose:
                logging.info(f"Step {i}: current wealth: {self.X[i]}, optimal strategy: {self.pi[i]}.")

        # i = T
        self.X[self.T] = self.R[self.T] * self.X[self.T-1] + (self.Z[self.T] - self.mu[self.T]) * self.pi[self.T-1]
        if verbose:
            logging.info(f"Step {self.T}: current wealth: {self.X[self.T]}.")
    
    def personalization(self):
        # calculate the personalization effect
        return sum([np.abs((1/self.gamma[i] - 1/self.gamma_C[i])*self.gamma_C[i]) for i in range(self.T)])/self.T

    def save_data(self, filename):
        # save the data to a file
        with open(filename, 'wb') as f:
            pickle.dump(self, f)