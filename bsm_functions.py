#
# Valuation of European call options in Black-Scholes-Merton model 
# incl. Vega function and implied volatility estimation # bsm_functions.py #
# Analytical Black-Scholes-Merton (BSM) Formula 
def bsm_call_value(S0, K, T, r, sigma):
    ''' Valuation of European call option in BSM model. Analytical formula.
    Parameters ========== S0 : float
    initial stock/index level
    K : float
    strike price
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate
    sigma : float volatility factor in diffusion term
    Returns =======
    value : float present value of the European call option
    '''
    from math import log, sqrt, exp; from scipy import stats;
    S0 = float(S0);
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    d2 = d1 - (sigma * sqrt(T));
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0)) 
    # stats.norm.cdf --> cumulative distribution function 
    # for normal distribution
    return value 
def bsm_put_value(S0, K, T, r, sigma):
    ''' Valuation of European call option in BSM model. Analytical formula.
    Parameters ========== S0 : float
    initial stock/index level
    K : float
    strike price
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate
    sigma : float volatility factor in diffusion term
    Returns =======
    value : float present value of the European call option
    '''
    from math import log, sqrt, exp; from scipy import stats;
    S0 = float(S0);
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    d2 = d1 - (sigma * sqrt(T));
    value = (-S0 * stats.norm.cdf(-d1, 0.0, 1.0) + K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0)) 
    # stats.norm.cdf --> cumulative distribution function 
    # for normal distribution
    return value 
    
# Vega function
def bsm_vega(S0, K, T, r, sigma): 
    ''' Vega of European option in BSM model.
    Parameters ========== S0 : float
    initial stock/index level
    K : float
    strike price
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate
    sigma : float volatility factor in diffusion term
    Returns =======
    vega : float
    partial derivative of BSM formula with respect to sigma, i.e. Vega
    '''
    from math import log, sqrt; from scipy import stats;
    S0 = float(S0);
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    vega = S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T);
    return vega
    
# Implied volatility function by Newton's method - call
def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100, tol=1e-6): 
    ''' Implied volatility of European call option in BSM model.
    Parameters ========== S0 : float
    initial stock/index level
    K : float
    strike price
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate sigma_est : float
    estimate of impl. volatility
    it : integer
    number of iterations
    Returns =======
    simga_est : float numerically estimated implied volatility
    '''
    cnt = 0;
    price = bsm_call_value(S0, K, T, r, sigma_est);
    vega = bsm_vega(S0, K, T, r, sigma_est);
    
    while (abs(price - C0) > tol) & (cnt <= it):
        sigma_est -= (price - C0) / vega;
        price = bsm_call_value(S0, K, T, r, sigma_est);
        vega = bsm_vega(S0, K, T, r, sigma_est);
        cnt += 1;

    return sigma_est

# Implied volatility function by Newton's method - put
def bsm_put_imp_vol(S0, K, T, r, P0, sigma_est, it=100, tol=1e-6): 
    ''' Implied volatility of European call option in BSM model.
    Parameters ========== S0 : float
    initial stock/index level
    K : float
    strike price
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate sigma_est : float
    estimate of impl. volatility
    it : integer
    number of iterations
    Returns =======
    simga_est : float numerically estimated implied volatility
    '''
    cnt = 0;
    price = bsm_put_value(S0, K, T, r, sigma_est);
    vega = bsm_vega(S0, K, T, r, sigma_est);
    
    while (abs(price - P0) > tol) & (cnt <= it):
        sigma_est -= (price - P0) / vega;
        price = bsm_put_value(S0, K, T, r, sigma_est);
        vega = bsm_vega(S0, K, T, r, sigma_est);
        cnt += 1;    
    
    return sigma_est