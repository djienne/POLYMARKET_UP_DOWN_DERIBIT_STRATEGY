Here is the complete extraction of the code and information presented in the video, organized by the logical steps of the tutorial.

### 1. The Breeden-Litzenberger Formula
**Concept:** The formula derives the Risk-Neutral Probability Density Function (PDF) from European call option prices.
**Formula:**
$$f_Q(K, \tau) = e^{r\tau} \frac{\partial^2 C(K, \tau)}{\partial K^2}$$

**Finite Difference Approximation (used in code):**
$$f_Q(K, \tau) \approx e^{r\tau} \frac{C(K+\Delta K, \tau) - 2C(K, \tau) + C(K-\Delta K, \tau)}{(\Delta K)^2}$$

---

### 2. Python Implementation

#### A. Imports and Parameters
The video sets up a Heston Model scenario to generate theoretical option prices to test the formula.

```python
import numpy as np
import scipy as sc
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import QuantLib as ql # Used later for benchmarking

# Initialise parameters
S0 = 100.0       # initial stock price
K = 150.0        # strike price
tau = 1.0        # time to maturity in years
r = 0.06         # annual risk-free rate

# Heston dependent parameters
kappa = 3        # rate of mean reversion of variance
theta = 0.20**2  # long-term mean of variance
v0 = 0.20**2     # initial variance
rho = 0.98       # correlation between returns and variances
sigma = 0.2      # volatility of volatility
lambd = 0        # risk premium of variance

# Heston condition check (2*kappa*theta > sigma^2) ensures variance remains positive
print(2*kappa*theta > sigma**2) 
# Output: True (0.24... > 0.04...)
```

#### B. Heston Semi-Analytical Pricing
To verify the Breeden-Litzenberger result, the video first calculates "exact" prices using the Heston characteristic function and rectangular integration.

```python
def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    # constants
    a = kappa * theta
    b = kappa + lambd
    
    # common terms w.r.t phi
    rspi = rho * sigma * phi * 1j
    
    # define d parameter given phi and b
    d = np.sqrt((rspi - b)**2 + (phi*1j + phi**2) * sigma**2)
    
    # define g parameter given phi, b and d
    g = (b - rspi + d) / (b - rspi - d)
    
    # calculate characteristic function by components
    exp1 = np.exp(r * phi * 1j * tau)
    term2 = S0**(phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g))**(-2 * a / sigma**2)
    exp2 = np.exp(a * tau * (b - rspi + d) / sigma**2 + v0 * (b - rspi + d) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau))) / sigma**2)
    
    return exp1 * term2 * exp2

def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    
    P, umax, N = 0, 100, 650 # Integration limits and steps
    dphi = umax / N # dphi is width
    
    for j in range(1, N):
        # rectangular integration
        phi = dphi * (2 * j + 1) / 2 # midpoint to calculate height
        numerator = heston_charfunc(phi - 1j, *args) - K * heston_charfunc(phi, *args)
        denominator = 1j * phi * K**(1j * phi)
        
        P += dphi * numerator / denominator
        
    return np.real((S0 - K * np.exp(-r * tau)) / 2 + P / np.pi)
```

#### C. Generating the Option Prices
The code generates prices for a range of strikes to build the volatility surface slice.

```python
# Generate strikes
strikes = np.arange(60, 180, 1.0) # Step size of 1.0 is crucial for finite difference

# Calculate prices using the Heston function defined above
option_prices = [heston_price_rec(S0, k, v0, kappa, theta, sigma, rho, lambd, tau, r) for k in strikes]

# Store in DataFrame
prices = pd.DataFrame([strikes, option_prices]).transpose()
prices.columns = ['strike', 'price']
```

#### D. Applying Breeden-Litzenberger (Curvature Calculation)
This is the core implementation of the formula using finite differences on the DataFrame.

```python
# Use 2nd order finite difference approximation
# The shift(1) is Previous Price, shift(-1) is Next Price
# Equation: (Price_next - 2*Price_current + Price_prev) / (dK^2)
# Here dK is 1.0, so division by 1**2 is implicit

prices['curvature'] = (-2 * prices['price'] + 
                       prices['price'].shift(1) + 
                       prices['price'].shift(-1)) / 1**2

# Note: The discount factor e^(rt) is applied later in the PDF calculation step.
```

#### E. Comparison with QuantLib (Benchmarking)
The video compares the manual rectangular integration with QuantLib's C++ implementation to check for numerical errors.

```python
# Setting up discount curve
today = ql.Date(28, 5, 2022)
ql.Settings.instance().evaluationDate = today
risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))

# Setting up Heston model
heston_process = ql.HestonProcess(risk_free_ts, dividend_ts, 
                                  ql.QuoteHandle(ql.SimpleQuote(S0)), 
                                  v0, kappa, theta, sigma, rho)
heston_model = ql.HestonModel(heston_process)
heston_engine = ql.AnalyticHestonEngine(heston_model)

# Calculate prices via QuantLib
option_prices_ql = []
for k in strikes:
    option = ql.EuropeanOption(ql.PlainVanillaPayoff(ql.Option.Call, k),
                               ql.EuropeanExercise(today + ql.Period(int(365*tau), ql.Days)))
    option.setPricingEngine(heston_engine)
    option_prices_ql.append(option.NPV())

# Compare MSE
mse = np.mean((np.array(option_prices) - np.array(option_prices_ql))**2)
print(f"Mean Squared Error: {mse}")
```

#### F. Recovering PDF and CDF
Using the curvature calculated earlier to get the full PDF and then integrating to get the Cumulative Distribution Function (CDF).

```python
# Drop NaN values generated by shift()
inter = prices.dropna()

# 1. Create Risk Neutral PDF (Apply discount factor)
# f_Q = e^(rt) * curvature
pdf_values = np.exp(r * tau) * inter['curvature']

# Interpolate to create a continuous function
pdf_func = sc.interpolate.interp1d(inter['strike'], pdf_values, kind='cubic')

# 2. Create Cumulative Distribution Function (CDF)
# CDF is the cumulative sum of the PDF
cdf_values = np.cumsum(pdf_values)
cdf_func = sc.interpolate.interp1d(inter['strike'], cdf_values, kind='linear')

# Plotting (Snippets)
# plt.plot(strikes, pdf_values)
# plt.plot(strikes, cdf_values)
```

#### G. Pricing Derivatives via Integration
Finally, the video demonstrates how to price complex derivatives (or verify vanilla ones) by integrating the payoff against the derived PDF.

```python
# Define integrand for a Call Option: (S_T - K) * PDF(S_T)
def integrand_call(x, K):
    return (x - K) * pdf_func(x)

# Define integrand for a Put Option: (K - S_T) * PDF(S_T)
def integrand_put(x, K):
    return (K - x) * pdf_func(x)

calls = []
puts = []

for k in strikes:
    # We must limit integration to the range of valid strikes (61 to 178 based on dropna)
    # Integral from K to Infinity (approx 178 here)
    call_int, err = sc.integrate.quad(integrand_call, k, 178, limit=1000, args=(k,))
    
    # Integral from -Infinity (approx 61 here) to K
    put_int, err = sc.integrate.quad(integrand_put, 61, k, limit=1000, args=(k,))
    
    # Apply discount factor to the integral result
    call_price = np.exp(-r * tau) * call_int
    put_price = np.exp(-r * tau) * put_int
    
    calls.append(call_price)
    puts.append(put_price)

# Store results in DataFrame
rnd_prices = pd.DataFrame([strikes, calls, puts]).transpose()
rnd_prices.columns = ['strike', 'Calls', 'Puts']
print(rnd_prices.head())
```

### Key Takeaways mentioned in the video:
1.  **Computational Efficiency:** Calculating option prices directly can be intense. Deriving the PDF allows you to price complex derivatives quickly by simply integrating the payoff against the PDF.
2.  **Model Independence:** Once you have the PDF (whether from Heston, Black-Scholes, or raw market data), the pricing method for derivatives is the same.
3.  **Numerical Stability:** Python's binary floating-point arithmetic can cause round-off errors when performing rectangular integration with very small step sizes, causing the calculated PDF to potentially dip below zero (which is theoretically impossible). Libraries like QuantLib (C++) handle this better.