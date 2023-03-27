import numpy as np

class ParamEstimator:
    """
    Initialize the personalized parameters of the client to be estimated
    gamma_0: initial risk aversion
    alpha: rate of time discount
    beta: rate of trend discount
    sig_eps: volatility of the idiosyncratic shock
    p_eps: probability of the idiosyncratic shock to be non-zero
    """
    def __init__(self, gamma_0, alpha, beta, sig_eps, p_eps) -> None:
        self.gamma_0 = gamma_0
        self.alpha = alpha
        self.beta = beta
        self.sig_eps = sig_eps
        self.p_eps = p_eps

    """
    Simulate the market and client's behavior, which is used to estimate the parameters later.
    r: risk-free rate
    sig: volatility of the risky asset
    mu: expected return of the risky asset
    N: number of periods of the estimation process
    """
    def simulate(self, r, sig, mu, N):
        # gamma_y depends on the sharpe ratio of the market
        def calc_gamma_y():
            return np.exp((mu - r) / sig)
        
        # eta is the time discount factor, alpha is the rate of time discount   
        def calc_eta(t):
            return np.exp(-self.alpha * (N-t))
        
        # initializations
        self.r = r
        self.sig = sig
        self.mu = mu
        self.N = N
        self.Z = np.zeros(self.N + 1)       # Z_0 is not defined, start from Z_1
        self.Z[0] = self.mu
        self.gamma_Y = np.zeros(self.N)
        self.eta = np.zeros(self.N)
        self.gamma_C = np.zeros(self.N)
        self.eps = np.ones(self.N+1)
        self.gamma_id = np.zeros(self.N)

        self.gamma_Z = np.zeros(self.N) # behavior bias
        self.xi = np.zeros(self.N)      # risk aversion used in decision making (observable)

        """
        Generate the observed risk aversionvalues
        """
        for i in range(self.N):
            # market dynamics
            if i > 0:
                # now generate the risky asset's price
                self.Z[i] = np.random.normal(loc=self.mu, scale=self.sig)

            # then generate the client's risk aversion process and the communicated risk aversion
            self.gamma_Y[i] = calc_gamma_y()
            self.eta[i] = calc_eta(i)
            if i > 0:
                self.eps[i] = np.random.choice([np.random.normal(loc=0, scale=self.sig_eps), 0], size=1, p=[self.p_eps, 1-self.p_eps])
                self.gamma_id[i] = self.gamma_id[i-1] * np.exp(self.eps[i])
                self.gamma_C[i] = self.gamma_Y[i] * self.eta[i] * self.gamma_id[i]
                
                self.gamma_Z[i] = np.exp(-self.beta * (self.Z[i] - self.mu))
                self.xi[i] = self.gamma_C[i] * self.gamma_Z[i]
            else:
                self.gamma_C[i] = self.gamma_0
                self.gamma_id[i] = self.gamma_C[i] / self.gamma_Y[i] / self.eta[i]

                self.gamma_Z[i] = 1
                self.xi[i] = self.gamma_C[i] * self.gamma_Z[i]
    
    """
    Estimate the parameters of the client's risk aversion process
    """
    def estimate(self, est_alpha_beta=False):
        self.dZ = np.zeros(self.N) # dZ[n] = Z[n] - Z[n-1]
        for i in range(self.N):
            if i > 0:
                self.dZ[i] = self.Z[i] - self.Z[i-1]
        self.Q = np.zeros(self.N) # Q[n] = xi[n]/xi[n-1]
        for n in range(1,self.N):
            self.Q[n] = np.log(self.xi[n]/self.xi[n-1])
        
        if est_alpha_beta:
            self.beta_hat = (np.sum([self.Q[2*i+1] for i in range((self.N-3)//2+1)]) - np.sum([self.Q[2*i+2] for i in range((self.N-3)//2+1)])) / (- np.sum([self.dZ[2*i+1] for i in range((self.N-3)//2+1)]) + np.sum([self.dZ[2*i+2] for i in range((self.N-3)//2+1)]))
            self.alpha_hat = (np.sum([self.Q[i] for i in range(1,self.N)]) + self.beta_hat * (self.Z[self.N-1]-self.mu))/(self.N-1)
        else:
            self.beta_hat = self.beta
            self.alpha_hat = self.alpha
            
        self.eps_hat = np.zeros(self.N+1)
        for i in range(1,self.N):
            self.eps_hat[i] = self.Q[i] + self.beta_hat * self.dZ[i] - self.alpha_hat
        
        mu2 = np.mean([self.eps_hat[n]**2 for n in range(1,self.N)])
        mu4 = np.mean([self.eps_hat[n]**4 for n in range(1,self.N)])

        self.sig_eps_hat = np.sqrt(mu4/3/mu2)
        self.p_eps_hat = mu2/(self.sig_eps_hat**2)
    
    def error(self, verbose=False):
        alpha_error = np.abs(self.alpha_hat - self.alpha)/self.alpha
        beta_error = np.abs(self.beta_hat - self.beta)/self.beta
        p_eps_error = np.abs(self.p_eps_hat - self.p_eps)/self.p_eps
        sig_eps_error = np.abs(self.sig_eps_hat - self.sig_eps)/self.sig_eps
        if verbose:
            print(f"[Relative error] alpha: {alpha_error}, beta: {beta_error}, p_eps: {p_eps_error}, sig_eps: {sig_eps_error}")
        return alpha_error, beta_error, p_eps_error, sig_eps_error