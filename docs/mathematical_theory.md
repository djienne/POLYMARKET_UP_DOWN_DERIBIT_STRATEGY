# Mathematical Theory: Forecasting Market Probabilities from Deribit BTC Options

## Table of Contents

1. [Introduction & Problem Statement](#1-introduction--problem-statement)
2. [Why Options Encode Probabilities](#2-why-options-encode-probabilities)
3. [Data Pipeline](#3-data-pipeline)
4. [Heston Stochastic Volatility Model](#4-heston-stochastic-volatility-model)
5. [SSVI (Surface SVI) Model](#5-ssvi-surface-svi-model)
6. [Model Selection](#6-model-selection)
7. [Breeden-Litzenberger: Extracting the Probability Density](#7-breeden-litzenberger-extracting-the-probability-density)
8. [Terminal Probability via Monte Carlo](#8-terminal-probability-via-monte-carlo)
9. [From Probabilities to Trading Edge](#9-from-probabilities-to-trading-edge)
10. [Backtesting & Optimization](#10-backtesting--optimization)
11. [Computational Acceleration](#11-computational-acceleration)
12. [References](#12-references)

---

## 1. Introduction & Problem Statement

**Goal:** Extract forward-looking probability distributions from BTC option prices traded on Deribit, then use those probabilities to identify mispriced contracts on prediction markets like Polymarket.

**Core insight:** Every option price implicitly encodes a bet about the future. A call option with strike $100K is worthless unless BTC ends above $100K — so its price reveals what the market collectively believes about `P(BTC > $100K)`. By building a mathematical model of the entire volatility surface, we can extract the *full* probability distribution, not just point estimates.

### Pipeline Overview

```
 Deribit API       Quality        Stochastic Vol     Risk-Neutral      Monte Carlo       Polymarket
 (raw options) --> Filters    --> Calibration     --> Density        --> Barriers      --> Edge
                                  (Heston/SSVI)      (Breeden-         (first-passage)    Detection
                                                      Litzenberger)

 +-----------+    +---------+    +-------------+    +------------+    +------------+    +----------+
 | Fetch all |    | Remove  |    | Fit model   |    | f(K) =     |    | Simulate   |    | Compare  |
 | BTC opts  |--->| stale,  |--->| to implied  |--->| e^rT *     |--->| 1M Heston  |--->| model vs |
 | mark_iv,  |    | illiq,  |    | vol smile   |    | d²C/dK²    |    | paths for  |    | market   |
 | strikes,  |    | wide-   |    | across all  |    |            |    | touch prob |    | price    |
 | OI, fwd   |    | spread  |    | strikes     |    | Normalize  |    | P(min≤H)   |    |          |
 +-----------+    +---------+    +-------------+    +------------+    +------------+    +----------+
```

The pipeline runs in roughly 50 seconds end-to-end (dominated by Heston calibration across multiple expiries), producing calibrated probabilities with confidence intervals.

---

## 2. Why Options Encode Probabilities

### Risk-Neutral Pricing and the Fundamental Theorem

In standard finance theory, the price of any derivative equals its expected payoff under a special probability measure called **Q** (the *risk-neutral measure*), discounted at the risk-free rate:

```
Price = e^(-rT) * E_Q[payoff(S_T)]
```

This follows from the **First Fundamental Theorem of Asset Pricing**: the absence of arbitrage is equivalent to the existence of at least one equivalent martingale measure Q under which discounted asset prices are martingales. Under Q, all assets earn the risk-free rate on average.

The transition from the real-world (physical) measure P to Q is governed by the **Radon-Nikodym derivative** dQ/dP, which re-weights probabilities to absorb the risk premium. Formally, by **Girsanov's theorem**, if under P the asset follows:

```
dS/S = μ dt + σ dW^P
```

then under Q (with market price of risk λ = (μ − r)/σ):

```
dS/S = r dt + σ dW^Q       where dW^Q = dW^P + λ dt
```

The Brownian motion is tilted by the market price of risk. For BTC, λ is historically positive (risk premium for holding a volatile asset), so Q assigns lower probability to upside outcomes and higher probability to downside outcomes than the physical measure P.

### Q vs P: When the Distinction Matters

For our use case — comparing option-implied probabilities against prediction market prices — the distinction is largely moot:

- Both Deribit options and Polymarket contracts are priced under Q (or very close to it)
- Binary event pricing (`P(S_T > K)`) maps directly to digital option pricing under Q
- The relevant question is not "what will really happen?" but "are these two Q-measure prices consistent?"

Where P vs Q *does* matter: if using option-implied probabilities as real-world forecasts (e.g., "how likely is BTC to crash?"), the Q-probabilities overweight tails due to risk aversion. The variance risk premium — the difference between implied and realized variance — is persistently positive for BTC, meaning Q-implied vol exceeds realized vol on average by 10-20%.

### Digital Options and the CDF Connection

A **digital (binary) call** pays $1 if S_T > K and $0 otherwise. Its risk-neutral price is:

```
Digital_call(K) = e^(-rT) · Q(S_T > K) = e^(-rT) · [1 − F_Q(K)]
```

where F_Q is the risk-neutral CDF. This is the direct mathematical link between option markets and prediction markets. A Polymarket "YES" token on "BTC > $100K at expiry" is precisely a digital call at K = $100K.

Digital call prices can also be recovered from vanilla calls via:

```
Digital_call(K) = −dC/dK
```

This is a first derivative (the Breeden-Litzenberger formula uses the second derivative to get the density). The negative sign reflects that call prices decrease as strikes increase.

### Implied Volatility

A call option on BTC with strike K and expiry T has a Black-Scholes price determined by a single free parameter: **implied volatility** (IV). Given the market price of the option, we can invert Black-Scholes to recover the IV that the market is "implying."

If all strikes shared the same IV, the market would be assuming BTC follows a simple lognormal random walk (geometric Brownian motion). In reality, IV varies by strike — the **volatility smile**:

- **OTM puts** (low strikes) have higher IV — the market prices crash risk (left tail is heavier than lognormal)
- **OTM calls** (high strikes) often have slightly lower IV, though BTC can exhibit a symmetric or even right-skewed smile during bull markets
- The pattern is roughly a skewed "smile" or "smirk" shape

The smile's existence proves the market believes in non-Gaussian returns: fat tails, negative skewness, and stochastic volatility. This strike-dependent IV is the raw data we model.

### IV Inversion: Newton-Raphson on Vega

Recovering IV from an observed option price requires inverting the Black-Scholes formula. Since no closed form exists, we use **Newton-Raphson iteration**:

```
σ_{n+1} = σ_n − [C_BS(σ_n) − C_market] / vega(σ_n)
```

where vega (the sensitivity of option price to volatility) provides the Jacobian:

```
vega = F · √T · φ(d₁)
```

with φ the standard normal PDF. Newton-Raphson converges quadratically (error squares each iteration) because vega is strictly positive for non-degenerate options. The implementation uses a Brent's method fallback for robustness when Newton fails to converge (deep OTM options where vega is near-zero).

**Convergence criterion:** |C_BS(σ) − C_market| < 10⁻¹⁰ or |Δσ| < 10⁻¹⁰, whichever is satisfied first, with a maximum of 100 iterations.

### Out-of-the-Money Convention

We construct the volatility surface using **out-of-the-money (OTM) options** only:
- **Puts** for K < F (forward price)
- **Calls** for K > F

OTM options are preferred because:
1. **Liquidity:** Market makers quote tighter bid-ask spreads on OTM options
2. **Information content:** OTM options are pure time value (no intrinsic value component), so their prices cleanly reflect probability beliefs
3. **Numerical stability:** ITM options have large intrinsic value and small time value — small price errors translate to large IV errors
4. **No early exercise:** For European options (Deribit standard), OTM options avoid the early-exercise premium complications of American-style ITM puts

### Log-Moneyness

The natural coordinate for fitting is **log-moneyness**:

```
k = ln(K / F)
```

where F is the forward price. This makes the smile roughly symmetric around k = 0 (ATM) and normalizes across different price levels and expiries. In the SVI/SSVI framework, total implied variance w(k) = σ²T is a function of k alone, making the parameterization scale-invariant.

### Forward Price from Put-Call Parity

The forward price F for a given expiry is determined by **put-call parity**:

```
C(K) − P(K) = e^(-rT) · (F − K)
```

Deribit provides the forward price directly (`underlying_price`), derived from their mark price methodology which synthesizes put-call parity across multiple strikes. This is more robust than using a single strike pair.

For forward pricing (setting r = 0, which is standard for crypto since there is no risk-free lending rate in BTC), we have:

```
C(K) − P(K) = F − K
```

This simplifies all subsequent pricing formulas.

---

## 3. Data Pipeline

### Deribit API

The system fetches the complete BTC options chain via the Deribit public API (`/public/get_book_summary_by_currency`). Key fields per option:

| Field | Description |
|---|---|
| `mark_iv` | Exchange-computed implied volatility (%) |
| `underlying_price` | Forward price for this expiry |
| `strike` | Strike price |
| `open_interest` | Number of contracts outstanding |
| `bid_price`, `ask_price` | Best bid/ask in BTC terms |

**Mark IV vs mid IV:** Deribit's `mark_iv` is exchange-computed from their mark price (a proprietary smoothing of the orderbook), which is generally more stable than computing IV from raw bid/ask midpoints. We use `mark_iv` as the primary data source for calibration.

### OTM Surface Construction

From the raw chain, we select one option per strike:
- Put if K < F (OTM put)
- Call if K > F (OTM call)
- ATM: prefer call if both exist

**Put-call parity consistency:** Since we use the exchange forward price, OTM puts and calls at the same strike should yield the same IV (by put-call parity under the forward measure). Deviations indicate arbitrage opportunities or stale quotes, which are filtered.

### Quality Filters

Each filter eliminates data that would degrade the model fit:

| Filter | Default Threshold | Why |
|---|---|---|
| Open interest | ≥ 0 (relaxed) | Ensures some trading activity; stale quotes mislead the fitter |
| Bid-ask spread | ≤ 200% of mid | Wide spreads indicate illiquid or stale quotes |
| Moneyness K/F | ∈ [0.5, 1.5] | Deep ITM/OTM options are noisy and dominated by intrinsic value |
| IV range | ∈ [0.05, 5.0] | Rejects clearly erroneous or stale implied vols |
| Minimum points | ≥ 5 options | Need enough data to constrain a 3–5 parameter model |

Additionally, the surface must span **both sides** of the forward price — we need puts *and* calls to characterize both tails of the distribution.

**Rationale for moneyness bounds:** At K/F = 0.5 (50% moneyness), a put is extremely deep ITM with negligible time value. The IV extracted from such an option has very large relative error because price ≈ intrinsic value ± ε, and small ε maps to large σ uncertainty. At K/F = 1.5, an OTM call is far in the right tail where the density is negligible and option prices are near zero.

*(Filter thresholds are set in `config.yaml` → `filters` section.)*

---

## 4. Heston Stochastic Volatility Model

### The Model

In a simple random walk (GBM), price moves randomly but the *size* of those moves (volatility) is constant. The Heston model relaxes this: **volatility itself is random**, fluctuating around a long-term level with its own dynamics.

This captures two key features of real markets:
1. **Volatility clustering**: high-vol periods tend to persist (κ controls how fast vol reverts to normal)
2. **Leverage effect**: when price drops, volatility tends to rise (ρ < 0 captures this correlation)

### Stochastic Differential Equations

The Heston (1993) model evolves two coupled processes — price S and variance v under the risk-neutral measure Q:

```
dS_t = S_t √v_t dW₁

dv_t = κ(θ − v_t) dt + ξ√v_t dW₂

corr(dW₁, dW₂) = ρ
```

The first equation says price changes are proportional to current volatility (√v). The second equation (a **Cox-Ingersoll-Ross (CIR) process**) says variance mean-reverts toward θ with speed κ, while being randomly perturbed with intensity ξ.

Note: we work with **forward prices** (drift-free under Q), so there is no drift term in the price SDE. Under the forward measure (using the zero-coupon bond as numeraire), the forward price is a martingale.

**Cholesky decomposition** of the correlated Brownians:

```
dW₁ = dZ₁
dW₂ = ρ dZ₁ + √(1 − ρ²) dZ₂
```

where Z₁, Z₂ are independent standard Brownian motions. This decomposition is used in both the Monte Carlo simulation and the characteristic function derivation.

### The 5 Parameters

| Parameter | Range | Meaning | Typical BTC Values |
|---|---|---|---|
| **v₀** | (0, 4] | Current instantaneous variance. √v₀ ≈ current ATM implied vol | 0.04–0.50 (20%–70% vol) |
| **κ** (kappa) | (0, 20] | Mean-reversion speed. Half-life = ln(2)/κ | 1–10 (half-life: days to weeks) |
| **θ** (theta) | (0, 4] | Long-run variance level. √θ ≈ long-run implied vol | 0.04–0.25 (20%–50% vol) |
| **ξ** (xi) | (0, 15] | Vol-of-vol. Controls smile curvature and kurtosis | 0.5–5.0 |
| **ρ** (rho) | (−1, 1) | Price-vol correlation. Controls smile skew | −0.3 to −0.7 |

**Parameter identification from the smile shape:**
- **v₀** is pinned by ATM IV: v₀ ≈ σ²_ATM
- **ρ** controls the **skew** (asymmetry of the smile): ∂σ/∂k|_{k=0} ∝ ρξ/(2√v₀)
- **ξ** controls the **convexity** (curvature of the smile): ∂²σ/∂k²|_{k=0} ∝ ξ²
- **κ and θ** jointly control the **term structure**: how the smile flattens with maturity (faster mean reversion → quicker flattening)

### CIR Variance Process

The variance process v_t follows a CIR (Cox-Ingersoll-Ross) diffusion with known analytical properties:

**Transition density:** The conditional distribution of v_t given v_s (for t > s) is a scaled non-central chi-squared:

```
v_t | v_s  ~  (ξ²(1 − e^(−κ(t−s))) / (4κ)) · χ²_d(λ)
```

where:
- Degrees of freedom: d = 4κθ/ξ²
- Non-centrality: λ = 4κ e^(−κ(t−s)) v_s / (ξ²(1 − e^(−κ(t−s))))

**Moments:**

```
E[v_t | v_s] = θ + (v_s − θ) e^(−κ(t−s))

Var[v_t | v_s] = v_s · (ξ²/κ)(e^(−κ(t−s)) − e^(−2κ(t−s))) + θ · (ξ²/(2κ))(1 − e^(−κ(t−s)))²
```

The conditional mean shows exponential decay toward θ with rate κ. The half-life of a variance shock is ln(2)/κ — for κ = 5, this is about 0.14 years (≈ 50 days).

### Feller Condition

The variance process can theoretically hit zero if vol-of-vol is too high relative to mean reversion:

```
Feller condition: 2κθ > ξ²

Equivalently: d = 4κθ/ξ² > 2
```

When the Feller condition holds (d > 2), the origin is **inaccessible** — the CIR process stays strictly positive with probability 1. When violated (d ≤ 2), the process can reach zero but is immediately reflected.

**Empirical reality for BTC:** The Feller condition is frequently violated in calibration (d ≈ 0.5–1.5 is common), because BTC's high vol-of-vol (ξ) relative to mean-reversion (κθ) pushes the variance process toward zero. This is not a model failure — it reflects real market dynamics where short-dated BTC vol can collapse to near-zero during quiet periods. We handle this numerically by reflecting variance at zero: `v = max(v, 0)`.

### Characteristic Function (The Key Formula)

The power of the Heston model is that it has a **closed-form characteristic function** — a formula that encodes the entire probability distribution in the frequency domain.

The characteristic function of ln(S_T/F) under the forward measure is:

```
φ(u) = E_Q[e^(iu · ln(S_T/F))] = exp[A(u) + B(u) · v₀]
```

This **affine structure** (exponential-affine in v₀) arises because the Heston model belongs to the class of **affine diffusions** (Duffie, Pan & Singleton, 2000). The functions A(u) and B(u) satisfy a system of **Riccati ODEs**:

```
dB/dτ = −½(u² + iu) + (κ − ρξiu)B + ½ξ²B²      B(0) = 0
dA/dτ = κθ · B                                     A(0) = 0
```

These ODEs have the closed-form solution given by Heston (1993). The implementation follows the **"Little Heston Trap" formulation** from Albrecher et al. (2007), which avoids numerical instabilities (branch-cut discontinuities in the complex logarithm) present in the original formula:

```
d  = √[(ρξui − κ)² + (ui + u²)ξ²]

g  = (κ − ρξui + d) / (κ − ρξui − d)

A  = (κθ/ξ²) · [(κ − ρξui + d)τ − 2 ln((1 − g·e^(dτ))/(1 − g))]

B  = (κ − ρξui + d)/ξ² · (1 − e^(dτ))/(1 − g·e^(dτ))
```

Here τ = T (time to maturity) and i = √(−1).

**Why the "Little Heston Trap" matters:** The original Heston formula uses the reciprocal definition of g (swapping numerator/denominator), which can cause |g·e^(dτ)| > 1 and push the complex logarithm onto a different Riemann sheet, producing discontinuous jumps in φ(u). The Albrecher formulation ensures |g·e^(dτ)| < 1 for all τ > 0, keeping the logarithm on its principal branch.

**Moment properties from the characteristic function:**

```
E[ln(S_T/F)]   = −i · φ'(0)     (should be ≈ −½v₀T for short T)
Var[ln(S_T/F)]  = −φ''(0) + [φ'(0)]²

Implied skewness ∝ ρξ/√v₀
Implied kurtosis ∝ ξ²/v₀
```

### Option Pricing via Fourier Inversion

Given the characteristic function, European option prices are recovered via Fourier inversion. We use the **Lewis (2001) formulation**, which is numerically superior to the original Heston two-integral approach:

```
C(K) = F − (√(KF)/π) ∫₀^∞ Re[e^(−iu·k) · φ(u − i/2) / (u² + 1/4)] du
```

where k = ln(K/F) is log-moneyness. This single-integral formula avoids the subtraction of two large nearly-equal numbers that plagues the Heston P₁/P₂ formulation.

**Numerical integration details:**

| Setting | Value | Rationale |
|---|---|---|
| Integration domain | [0, u_max] | u_max = 100; integrand decays as O(1/u²) |
| Quadrature | Rectangular midpoint rule | Simple, stable, well-suited for oscillatory integrands |
| Grid points | 512 (pricing), 1024 (calibration) | Pricing needs speed; calibration needs accuracy |
| Truncation error | < 10⁻⁸ for typical params | φ(u) decays exponentially for large u when v₀ > 0 |

**Price bounds:** The result is clamped to `[max(F − K, 0), F]` — a call's value lies between its intrinsic value and the forward price.

**Put prices** are recovered via forward put-call parity: `P(K) = C(K) − (F − K)`.

### IV Extraction from Heston Prices

After computing C_Heston(K) for each market strike, we invert Black-Scholes to get the model-implied IV:

```
C_BS(K, σ_model) = C_Heston(K)  →  solve for σ_model
```

This inversion uses the same Newton-Raphson/Brent procedure as the market IV extraction (Section 2). The calibration objective then compares σ_model(kᵢ) vs σ_market(kᵢ) directly in IV space, which is more numerically stable than comparing prices (where deep OTM options have near-zero prices and huge relative errors).

### Calibration (Finding the 5 Parameters)

**Objective:** Find (v₀, κ, θ, ξ, ρ) that minimize the weighted sum of squared errors between model-implied and market-observed implied volatilities:

```
SSE = Σᵢ wᵢ · (σ_model(kᵢ) − σ_market(kᵢ))²
```

where kᵢ are the log-moneyness values and wᵢ are weights (uniform by default, or vega-weighted for ATM emphasis).

**Weighting schemes:**
- **Uniform weights** (default): all strikes contribute equally. Best when the smile is well-sampled
- **Vega weights** (wᵢ ∝ vega_i): emphasize ATM options where IV is most precisely determined. Useful for noisy data but can under-fit wings

**ATM penalty:** For short-dated options (TTM < 36 days), an additional penalty anchors the fit at ATM:

```
SSE += λ_atm · N · (σ_model(0) − σ_market_atm)²
```

where λ_atm = 10 for short-dated, 50 for very short-dated (TTM < 7 days), and 0 otherwise. This addresses the identifiability problem where multiple (κ, θ, v₀) combinations can produce similar wing shapes but different ATM levels. The penalty acts as a Bayesian prior anchoring v₀ to the observed ATM variance.

**Optimization strategy:**

1. **Black-Scholes initialization:** Bootstrap initial guess from ATM vol (v₀ ≈ σ²_ATM) and smile shape:
   - ρ estimated from smile skew: ρ ≈ 2·(σ(k=−0.1) − σ(k=+0.1)) / (0.2·ξ) (linear approximation)
   - ξ estimated from smile curvature: ξ ≈ √(2·(σ(k=−0.1) + σ(k=+0.1) − 2·σ_ATM) / 0.01)
   - θ ≈ v₀ (assume variance is near equilibrium)
   - κ ≈ 2.0 (moderate mean reversion as default)

2. **Multi-start Differential Evolution:** Run DE from 5 diverse starting points spanning the parameter space. DE is a population-based global optimizer that evolves candidate solutions through mutation, crossover, and selection:
   - Population size: 15 × (number of parameters) = 75
   - Mutation strategy: `best1bin` (mutate toward best solution)
   - Crossover probability: 0.7
   - Mutation factor: (0.5, 1.0) dithered
   - Maximum generations: 1000
   - Tolerance: 10⁻⁸

   DE is robust to local minima — critical because the Heston objective has many. The SSE surface has ridges along κ-θ (the mean-reversion speed and level are partially interchangeable for short maturities) and along ξ-ρ (vol-of-vol and correlation jointly determine smile shape).

3. **TTM-adaptive bounds:** Short-dated options need wider parameter ranges because the smile can be extreme:
   - Standard: ξ ∈ [0.1, 5], κ ∈ [0.1, 10]
   - Short-dated (< 36 days): ξ ∈ [0.1, 10], κ ∈ [0.01, 15]
   - Very short-dated (< 7 days): ξ ∈ [0.1, 15], κ ∈ [0.001, 20]

   The physical reasoning: short-dated options see the current volatility regime (not the long-run average), so κ can be very small (variance doesn't need to mean-revert in 2 days) and ξ can be large (vol-of-vol is high relative to the short horizon).

4. **L-BFGS-B polish:** After DE finds the basin of attraction, a local gradient-based optimizer (Limited-memory BFGS with box constraints) refines the solution using approximate Hessian information. This typically improves SSE by 1-2 orders of magnitude from the DE solution.

5. **Early termination:** If any multi-start achieves SSE < 1×10⁻⁵, remaining starts are skipped (the fit is already excellent).

6. **Parallel execution:** Multi-starts run in separate processes via `ProcessPoolExecutor`, eliminating GIL contention. This provides near-linear speedup on multi-core systems.

### Goodness of Fit

- **R²:** Coefficient of determination in IV space:
  ```
  R² = 1 − Σ(σ_model − σ_market)² / Σ(σ_market − σ̄_market)²
  ```
  R² > 0.90 is excellent; R² > 0.80 is acceptable; R² < 0.50 triggers fallback to SSVI

- **RMSE:** Root mean squared error in IV units:
  ```
  RMSE = √(SSE / N)
  ```
  Typical values: 0.005–0.02 (0.5%–2% absolute IV error)

- **Max residual:** Largest absolute IV error at any strike. If max|σ_model − σ_market| > 0.10 · σ_market at any strike, the fit is flagged as inconsistent

- **Weighted residual analysis:** Residuals should be roughly uniformly distributed across strikes. Systematic patterns (e.g., all wing residuals positive) suggest model misspecification

---

## 5. SSVI (Surface SVI) Model

### Purpose

SSVI is a simpler **3-parameter** volatility surface model. It serves as a fallback when Heston calibration fails (too few data points, degenerate smile shapes, or numerical issues). Unlike Heston, SSVI is a **reduced-form model** — it directly parameterizes the IV surface without specifying underlying stochastic dynamics. This makes it more flexible (can fit any smile shape) but less informative (cannot generate Monte Carlo paths directly).

### Gatheral & Jacquier (2014) Formula

The SSVI parametrizes **total implied variance** w(k) = σ²T as a function of log-moneyness:

```
w(k) = (θ/2) · [1 + ρφk + √((φk + ρ)² + 1 − ρ²)]
```

To recover implied volatility: σ(k) = √(w(k) / T).

**Verification at ATM (k = 0):**

```
w(0) = (θ/2) · [1 + √(ρ² + 1 − ρ²)] = (θ/2) · [1 + 1] = θ
```

So θ is exactly the ATM total variance, confirming the parameter's interpretation.

**Asymptotic behavior:** For large |k|:

```
w(k) → (θ/2) · φ|k| · (1 + sign(k)·ρ)    as |k| → ∞
```

The wings grow linearly in |k| with slopes (θφ/2)(1 ± ρ). The left wing slope is steeper when ρ < 0 (negative skew), consistent with crash-risk pricing.

### The 3 Parameters

| Parameter | Range | Meaning | Effect on Smile |
|---|---|---|---|
| **θ** (theta) | > 0 | ATM total variance (σ²T at k = 0). Sets the overall level | Vertical shift |
| **ρ** (rho) | (−1, 1) | Skew. Negative = puts more expensive than calls (typical for BTC) | Tilt/asymmetry |
| **φ** (phi) | > 0 | Curvature. Higher = steeper smile wings | Wing steepness |

### No-Arbitrage Constraints

The SSVI surface must satisfy two classes of no-arbitrage conditions:

**1. Butterfly (static) arbitrage:** The density f(K) must be non-negative everywhere, which requires the second derivative of call prices w.r.t. strike to be non-negative. For SSVI, a sufficient condition is:

```
θ · φ · (1 + |ρ|) ≤ 4
```

This is enforced as a hard constraint during optimization. If violated, the implied density goes negative at some strikes — a logical impossibility that would allow free-money butterfly trades.

**More precisely**, the full no-butterfly condition requires checking that for all k:

```
g(k) = (1 − k·w'/(2w))² − (w'/2)²·(1/4 + 1/w) + w''/2  ≥  0
```

where w = w(k) and primes denote derivatives w.r.t. k. The θφ(1+|ρ|) ≤ 4 condition is a tractable sufficient condition that is tight for the SSVI parameterization.

**2. Calendar spread (dynamic) arbitrage:** For consistency across expiries, total variance must be non-decreasing in T for each fixed k:

```
∂w(k, T)/∂T ≥ 0    for all k, T
```

Since we calibrate SSVI per-slice (one expiry at a time), this condition is checked ex post but not enforced during optimization. Violations are rare in practice for BTC options.

### Connection to Local Volatility

Via Dupire's formula (1994), any IV surface implies a unique **local volatility** surface:

```
σ²_local(K, T) = (∂w/∂T) / [1 − (k/w)·(∂w/∂k) + (1/4)·(−1/4 − 1/w + k²/w²)·(∂w/∂k)² + (1/2)·(∂²w/∂k²)]
```

This local vol surface is used for the SSVI Monte Carlo simulation (Section 8): we simulate paths under a local volatility model calibrated to the SSVI surface, preserving consistency with the observed smile.

### Roger Lee's Moment Formula

The asymptotic slope of the smile constrains the existence of moments of S_T. By Roger Lee (2004):

```
p_max = (β_R / 2) − 1 + √(β_R²/4 + β_R)      (right wing slope β_R = lim w(k)/k as k→∞)
q_max = (β_L / 2) − 1 + √(β_L²/4 + β_L)      (left wing slope β_L = lim w(k)/|k| as k→−∞)
```

where E[S_T^(1+p)] < ∞ for p < p_max and E[S_T^(-q)] < ∞ for q < q_max.

For SSVI, β_R = (θφ/2)(1 + ρ) and β_L = (θφ/2)(1 − ρ). The butterfly condition ensures β_R, β_L ≤ 2, guaranteeing at least the first moment (mean) exists.

### Calibration

1. **Primary:** SLSQP (Sequential Least Squares Quadratic Programming) — a constrained local optimizer that respects the butterfly condition. Run from 5 diverse starting points (multi-start) with initial guesses:
   - θ₀ = σ²_ATM · T (from data)
   - ρ₀ ∈ {−0.5, −0.3, 0.0, 0.3, 0.5} (span skew possibilities)
   - φ₀ from smile curvature heuristic

2. **Global fallback:** If best local R² < 0.85, switch to Differential Evolution with the butterfly condition as a nonlinear constraint. DE uses `workers=-1` for parallel population evaluation

3. **Objective function** (relative error with regularization):
   ```
   L = Σᵢ wᵢ · [(σ_model(kᵢ) − σ_market(kᵢ)) / σ_market(kᵢ)]² + λ · (φ − φ₀)² / φ²_max
   ```
   The relative error formulation ensures OTM options (with lower absolute IV) are not dominated by ATM options. The regularization term (λ = 0.001) penalizes extreme φ values, preventing overfitting to noisy wing data. This acts as a Tikhonov (L2) regularizer on the curvature parameter.

4. **TTM-adaptive φ bounds:**
   - Standard: φ ∈ [0.001, 5]
   - Short-dated (< 36 days): φ ∈ [0.001, 20]
   - Very short-dated (< 7 days): φ ∈ [0.001, 200]

   Short-dated smiles can be extremely steep, requiring large φ to capture the curvature. Physically, this reflects the market pricing discrete near-term events (e.g., FOMC meetings, ETF decisions) that create jump-like behavior on short timescales.

---

## 6. Model Selection

Both Heston and SSVI Surface models are always calibrated independently:

1. **Heston** — Stochastic volatility with 5 parameters (v0, kappa, theta, xi, rho). Calibrated in parallel across expiries via `calibrate_to_expiry()` using up to 4 workers.

2. **SSVI Surface** — Gatheral & Jacquier (2014) joint fit across nearby expiries. Fitted separately via `fit_ssvi_surface_for_ttm()` with up to 4 workers.

3. **`trading_model` config** (`"ssvi_surface"` by default) selects which model's probabilities become `avg_prob` for downstream trading. No automatic fallback — if the preferred model fails, `avg_prob` is `None`.

4. **IV consistency check:** After Heston calibration, if any strike's model IV differs from market IV by > 10% relative error, the fit is flagged as inconsistent. This catches pathological fits where the optimizer converges to a parameter set that fits most strikes well but has a catastrophic outlier.

All raw probabilities (both models' MC, B-L, and params) are saved in the JSON output regardless of trading model selection, enabling backtesting comparison.

*(Configured via `config.yaml` → `model` section: `default_model`, `trading_model`, `iv_consistency_threshold`.)*

---

## 7. Breeden-Litzenberger: Extracting the Probability Density

### Core Insight

The **Breeden-Litzenberger (1978) formula** is the mathematical bridge between option prices and the probability density. It follows from a fundamental result in mathematical finance:

**Arrow-Debreu securities:** A security that pays $1 if and only if the final price is in [K, K+dK] has price f(K)·dK under forward pricing. The density f(K) is the price per unit of "state" — it tells you how expensive it is to buy insurance against each possible outcome.

**Derivation from call prices:**

Starting from the risk-neutral pricing formula for a European call:

```
C(K) = e^(-rT) · E_Q[(S_T − K)⁺] = e^(-rT) · ∫_K^∞ (s − K) · f(s) ds
```

Differentiating once with respect to K (by Leibniz rule):

```
dC/dK = −e^(-rT) · ∫_K^∞ f(s) ds = −e^(-rT) · [1 − F_Q(K)] = −e^(-rT) · Q(S_T > K)
```

This recovers the **digital call price** (survival function). Differentiating again:

```
d²C/dK² = e^(-rT) · f(K)
```

Therefore:

```
f(K) = e^(rT) · d²C/dK²
```

For forward pricing (r = 0, which we use since we work with forward prices directly):

```
f(K) = d²C_fwd/dK²
```

### Intuition: Butterfly Spreads

Consider a **butterfly spread**: buy a call at K−δ, sell two calls at K, buy a call at K+δ. The payoff is a narrow tent centered at K:

```
Payoff ≈ δ  if S_T is near K
         0   otherwise
```

The price of this butterfly is approximately `δ² · f(K)` — it's literally buying a "bet" that the price lands near K. The second derivative of call prices recovers f(K) because it's the limit of butterfly prices as δ → 0.

More precisely:

```
[C(K−δ) − 2C(K) + C(K+δ)] / δ² → d²C/dK² = f(K)    as δ → 0
```

This is the finite-difference approximation to the second derivative.

### Numerical Implementation

The extraction proceeds in 4 steps:

**Step 1: Create strike grid**
- 500 equally-spaced points (configurable via `breeden_litzenberger.strike_grid_points`)
- Range: forward × exp(±3σ√T), covering ±3 standard deviations of the implied distribution
- Grid spacing: ΔK = (K_max − K_min) / 499

The choice of 500 points balances resolution against spline stability. Too few points → coarse density; too many → spline oscillation in the wings.

**Step 2: Compute call prices**
For each grid strike K, query the calibrated model for IV σ(K), then compute the Black-Scholes forward call price:

```
d₁ = [ln(F/K) + ½σ²T] / (σ√T)
d₂ = d₁ − σ√T
C  = F·Φ(d₁) − K·Φ(d₂)
```

where Φ is the standard normal CDF. This is computed in a vectorized NumPy pass over all 500 strikes simultaneously.

**Convexity requirement:** The call price C(K) must be convex in K (d²C/dK² ≥ 0) for the density to be non-negative. This is guaranteed by construction if the IV smile satisfies the butterfly no-arbitrage condition.

**Step 3: Differentiate**
Fit a **cubic spline** to C(K) and evaluate its second derivative analytically. The spline provides smoother derivatives than finite differences:

- **Finite differences** amplify noise: the second-order FD approximation has error O(ΔK²), but its variance is O(σ²_noise / ΔK⁴) — noise is amplified by the fourth power of the inverse grid spacing
- **Cubic splines** impose C² continuity, effectively regularizing the second derivative. The spline's natural boundary conditions (zero second derivative at endpoints) are appropriate because the density vanishes at extreme strikes

**Spline boundary conditions:** Natural spline (S''(K_min) = S''(K_max) = 0), which is consistent with the density going to zero at the tails.

**Step 4: Clean and normalize**
- Clamp negative density values to zero (numerical artifacts from Gibbs-like oscillation at density discontinuities)
- Normalize: f(K) → f(K) / ∫f(K)dK so the density integrates to 1.0
- Validate:
  - Final integral should be within 5% of 1.0 pre-normalization (large deviations indicate grid truncation or model problems)
  - Mean should be within 10% of the forward price (the risk-neutral mean should equal F by martingale property)

### Moments from the Density

All distribution statistics are computed via **numerical integration** (Simpson's rule, which has error O(ΔK⁴) for smooth integrands):

| Statistic | Formula | Interpretation |
|---|---|---|
| Mean | μ = ∫ K · f(K) dK | Should ≈ F (forward price) by risk-neutrality |
| Variance | Var = ∫ (K − μ)² · f(K) dK | Dispersion of terminal price |
| Std deviation | σ = √Var | Approximate price uncertainty |
| Skewness | γ₁ = ∫ (K − μ)³ · f(K) dK / σ³ | Asymmetry: γ₁ < 0 → heavier left tail (crash risk) |
| Excess kurtosis | γ₂ = ∫ (K − μ)⁴ · f(K) dK / σ⁴ − 3 | Tail weight: γ₂ > 0 → fatter tails than normal |
| Mode | argmax f(K) | Most likely terminal price |

**Typical BTC RND characteristics:**
- Negative skewness (γ₁ ≈ −0.3 to −1.5): crash risk
- Positive excess kurtosis (γ₂ ≈ 1 to 10): fat tails
- Mode slightly above forward: slight right skew in the central mass despite heavy left tail

### Terminal Probabilities

The CDF is computed via cumulative trapezoid integration, then:

```
P(S_T > K) = ∫_K^∞ f(s) ds  =  1 − F_Q(K)

P(S_T < K) = F_Q(K) = ∫_{-∞}^K f(s) ds
```

**Percentiles** are found by inverting the CDF with linear interpolation: find the K where F_Q(K) = p. The interpolation error is O(ΔK) — negligible for a 500-point grid.

**Consistency check:** P(S_T > K) + P(S_T < K) = 1 by construction (the normalized density integrates to 1). However, the probability at exactly K is zero for a continuous distribution, so P(S_T = K) = 0.

---

## 8. Terminal Probability via Monte Carlo

### What We Compute

The code computes **terminal probabilities**: "Will BTC be above/below price H **at expiry**?" → P(S_T ≥ H) or P(S_T < H).

Two independent methods are used and then averaged (for SSVI Surface):
1. **Breeden-Litzenberger** — Analytical integration of the risk-neutral density extracted from the calibrated volatility surface
2. **Monte Carlo** — Simulate price paths under the local volatility surface derived from SSVI, check terminal price vs H

> **Note on first-passage vs terminal:** In barrier option pricing, "barrier probability" refers to first-passage — the probability of *touching* a price at any point before expiry, P(min S_t ≤ H). This is always ≥ the terminal probability because price can touch and bounce back. The code computes only terminal probabilities, not first-passage.

### Heston Monte Carlo Simulation

Since the Heston model has no closed-form first-passage formula (unlike GBM), we use **Monte Carlo simulation**: generate many random price paths and count how many touch the barrier.

**Euler-Maruyama discretization** of the coupled SDEs:

```
For each time step t → t+dt:

    Z₁, Z₂ ~ N(0, 1)                          # Independent standard normals
    Z₂_corr = ρ·Z₁ + √(1−ρ²)·Z₂              # Correlate the shocks (Cholesky)

    V_new = V + κ(θ − V)·dt + ξ·√V·√dt·Z₂    # Variance step (CIR process)
    V = max(V_new, 0)                           # Reflection at zero (absorption scheme)

    S_new = S · exp(−½V·dt + √V·√dt·Z₁)       # Log-Euler price step
```

**Why log-Euler for the price process:** The naive Euler scheme `S_new = S + S·√V·√dt·Z₁` can produce negative prices. The log-Euler scheme (exponential of the log-return increment) preserves positivity by construction: exp(anything) > 0.

**Variance floor (reflection scheme):** When V goes negative (Feller condition violated), we clamp to zero: V = max(V_new, 0). This is the simplest scheme and corresponds to the "absorption" boundary condition at V = 0. More sophisticated schemes exist:

| Scheme | Formula | Properties |
|---|---|---|
| **Reflection** (used) | V = max(V_new, 0) | Simple, first-order |
| **Full truncation** | V_step uses max(V, 0) in drift and diffusion | Better preserves moments |
| **QE (Quadratic Exponential)** | Exact sampling from non-central χ² | Second-order, no bias, more complex |

The QE scheme (Andersen, 2008) is theoretically superior but computationally more expensive per step. For our application (1M paths × ~1000 steps), the simple reflection scheme with small dt (5-minute steps) provides sufficient accuracy.

**Convergence properties:**
- **Weak order:** O(dt) — expected values converge linearly in the step size
- **Strong order:** O(√dt) — pathwise convergence at square-root rate

For our barrier computation (an expected value of an indicator function), weak convergence is the relevant metric. With dt = 5 minutes / 1440 minutes per day ≈ 0.0035 days, the weak error is O(0.0035) ≈ 0.35% — well within our tolerance.

**Time grid:** `n_steps_per_day` steps per day (default 288 = 5-minute intervals, configurable via `config.yaml` → `barrier.n_steps_per_day`). For a 3-day expiry at 288 steps/day, this means 864 total steps.

**Path count:** 1,000,000 paths (configurable via `barrier.n_simulations`), split into two batches for variance reduction.

### SSVI Local Volatility Monte Carlo

When SSVI is the selected model (no Heston dynamics available), barrier probabilities are computed via **local volatility Monte Carlo**. The SSVI surface is converted to a Dupire local volatility surface:

```
σ²_local(S, t) = [∂w/∂T] / [1 − (k/w)(∂w/∂k) + ¼(−¼ − 1/w + k²/w²)(∂w/∂k)² + ½(∂²w/∂k²)]
```

The MC simulation then uses:

```
S_new = S · exp(−½σ²_local(S,t)·dt + σ_local(S,t)·√dt·Z)
```

This is a one-factor model (no stochastic vol) but captures the smile-consistent dynamics implied by the SSVI surface. The local vol model reprices all European options exactly by construction (Dupire, 1994), so terminal probabilities match the Breeden-Litzenberger density. Barrier probabilities may differ from Heston because the path dynamics are different (deterministic vol vs stochastic vol).

### Antithetic Variates (Variance Reduction)

When `use_antithetic` is enabled (default), the total path count is split into two independent batches of 500K paths each. The results are concatenated before computing statistics. This batched approach provides some variance reduction through independent sampling while keeping memory usage bounded (each batch generates its own random numbers independently).

The theoretical ideal for antithetic variates — running the second batch with negated shocks −Z — would provide stronger variance reduction by ensuring symmetric sampling of the distribution:

```
θ̂_antithetic = ½[g(Z) + g(−Z)]

Var(θ̂_antithetic) = ½Var(g(Z)) + ½Cov(g(Z), g(−Z))
```

When g is monotone (as for barrier crossings), Cov(g(Z), g(−Z)) < 0, so variance is reduced. The current implementation trades theoretical optimality for simplicity and numerical robustness with the Heston model's correlated two-factor dynamics.

**Other variance reduction techniques** (not currently implemented but worth noting):
- **Importance sampling:** Shift the drift toward the barrier to generate more crossing events, then re-weight. Can reduce variance by 10-100× for rare events (P < 5%)
- **Stratified sampling:** Partition the random number space into strata and sample uniformly from each. Reduces variance by O(1/N²) vs O(1/N) for crude MC
- **Control variates:** Use the GBM closed-form as a control. If Y_GBM is the closed-form answer and Ŷ_GBM is the MC estimate, then θ̂_CV = θ̂_Heston − β(Ŷ_GBM − Y_GBM) has reduced variance

### Touch Probability Computation

After simulation, we have 1M final prices and running min/max for each path:

```
Down barrier H:  P(touch) = (1/N) · Σᵢ 𝟙{running_min_i ≤ H}
Up barrier H:    P(touch) = (1/N) · Σᵢ 𝟙{running_max_i ≥ H}
```

where 𝟙{·} is the indicator function.

**Standard error** (since each path is an independent Bernoulli trial):

```
SE = √(p̂(1 − p̂) / N)
```

This follows from the CLT applied to the sample mean of Bernoulli random variables.

**95% confidence interval** (Wald interval):

```
CI = [p̂ − 1.96·SE,  p̂ + 1.96·SE]
```

| N (paths) | p̂ = 0.50 SE | p̂ = 0.10 SE | p̂ = 0.01 SE |
|---|---|---|---|
| 100K | 0.0016 | 0.0009 | 0.0003 |
| 1M | 0.0005 | 0.0003 | 0.0001 |
| 10M | 0.00016 | 0.00009 | 0.00003 |

With 1M paths, a probability of 0.50 has SE ≈ 0.0005, giving a CI width of ±0.001 (±0.1 percentage points). This is more than sufficient for comparison against Polymarket prices (which are quoted in $0.01 increments, i.e., 1 percentage point resolution).

**Convergence rate:** MC converges at O(1/√N) regardless of dimension — this is both a strength (no curse of dimensionality) and a limitation (slow convergence). Doubling precision requires 4× more paths.

### Memory-Efficient Implementation

The compact simulation (`simulate_heston_paths_compact`) tracks only three vectors of length n_paths — final price, running min, and running max — rather than storing the full (n_paths × n_steps) matrix:

| Approach | Memory (500K paths × 8640 steps) |
|---|---|
| Full matrix | 500K × 8640 × 8 bytes ≈ 34 GB |
| Compact (3 vectors) | 3 × 500K × 8 bytes ≈ 12 MB |

This is possible because the barrier event depends only on path extrema, not the full path history. The running min/max are updated incrementally at each time step:

```
running_min = min(running_min, S_new)
running_max = max(running_max, S_new)
```

When Numba is available, the inner loop is JIT-compiled to machine code, providing significant speedup while maintaining identical numerical results (all random numbers are pre-generated with NumPy before entering the JIT kernel).

### GBM Analytical Benchmark

For **constant volatility** (GBM), there is a closed-form first-passage formula that serves as a sanity check.

**Down barrier (H < S₀):**

```
P(min_{0≤t≤T} S_t ≤ H) = Φ(−d₂) + (H/S₀)^(2μ/σ²) · Φ(−d₁)
```

where:

```
d₁ = [ln(S₀/H) + (μ + ½σ²)T] / (σ√T)
d₂ = [ln(S₀/H) + (μ − ½σ²)T] / (σ√T)
μ  = r − q  (risk-free rate minus dividend yield; typically 0 for crypto forwards)
```

**Up barrier (H > S₀):** By symmetry of the reflection principle:

```
P(max_{0≤t≤T} S_t ≥ H) = Φ(d₂') + (H/S₀)^(2μ/σ²) · Φ(d₁')
```

with d₁', d₂' using ln(H/S₀) instead of ln(S₀/H).

**Derivation sketch:** The formula follows from the **reflection principle** of Brownian motion. For a drifted BM X_t = μt + σW_t, the joint density of (X_T, min_{t≤T} X_t) is known in closed form via the method of images (reflecting the transition density around the barrier level). Integrating over the final position gives the marginal hitting probability.

The GBM formula provides a **lower bound** for the Heston barrier probability when the Heston ATM vol matches the GBM vol. Stochastic volatility generally increases barrier touch probabilities because:
1. Vol spikes increase the probability of large moves toward the barrier
2. The variance of variance creates fatter tails than GBM
3. Negative ρ means downward moves amplify volatility, further increasing downside barrier probabilities

### Discrete Monitoring Bias

The simulation checks barrier crossings only at discrete time steps (every ~5 minutes). Intra-step crossings are missed, causing a slight **underestimation** of touch probability.

**Quantification:** For GBM with constant vol σ and barrier distance d = |ln(S/H)|, the continuous-monitoring probability exceeds the discrete-monitoring probability by approximately:

```
ΔP ≈ f(H) · σ · √(dt) · √(2/π) · 0.5826
```

where f(H) is the density at the barrier and 0.5826 = ζ(1/2)/√(2π) (Broadie, Glasserman & Kou, 1997). For dt = 5 minutes ≈ 0.0035 days and σ = 50% annualized:

```
σ · √dt ≈ 0.50 · √(0.0035/365) ≈ 0.0016
```

This bias is < 0.2% for typical barrier distances, well within MC standard error.

**Brownian bridge correction:** A more precise correction uses the conditional probability of barrier crossing between two consecutive grid points, given the endpoints. For GBM between S_t and S_{t+dt} with barrier H:

```
P(min_{[t,t+dt]} S_u ≤ H | S_t, S_{t+dt}) = exp(−2·ln(S_t/H)·ln(S_{t+dt}/H) / (σ²·dt))
```

This can be applied as a random "coin flip" at each step to detect intra-step crossings. This correction is exact for GBM but approximate for Heston; it is not currently implemented due to the added complexity with stochastic volatility.

---

## 9. From Probabilities to Trading Edge

### Continuous Edge Function

Rather than fixed edge-ratio thresholds, the system uses a **continuous doubt-compression function** that adapts the required model confidence to the market's confidence level:

```
required_model_prob(p) = max(floor, 1 − (1 − p)^α)
```

where `p` is the Polymarket probability and `α` (alpha) is a curvature exponent. A trade is entered when the model probability exceeds this threshold.

**Intuition:** The function works on "doubt" — the probability the market is wrong, `(1 − p)`. The model must have even less doubt than the market, compressed by the exponent α. At high market confidence (p = 0.90, doubt = 0.10), the required model doubt is `0.10^α` — much smaller than 0.10 for α > 1. The `floor` parameter prevents trades where the model itself isn't confident enough regardless of market price.

### Mathematical Properties

Let h(p) = 1 − (1 − p)^α. This function has the following properties:

1. **Domain and range:** h: [0, 1] → [0, 1]
2. **Boundary values:** h(0) = 0, h(1) = 1
3. **First derivative:** h'(p) = α(1 − p)^(α−1) > 0 (strictly increasing)
4. **Second derivative:** h''(p) = −α(α−1)(1 − p)^(α−2)
   - For α > 1: h''(p) < 0 → **concave** (progressively harder to meet threshold at higher market prices)
   - For α < 1: h''(p) > 0 → **convex** (progressively easier at higher market prices)
   - For α = 1: h(p) = p → linear (identity function)
5. **Fixed point:** h(p) = p has a unique interior solution at p = 0 and p = 1. For α > 1, h(p) > p for all p ∈ (0, 1), so the required model probability always exceeds the market probability

**Sensitivity to α near market extremes:**

```
∂h/∂α = −(1 − p)^α · ln(1 − p)
```

This is maximized at intermediate doubt levels (p ≈ 0.5–0.8), meaning α has the strongest effect on entry decisions for moderately-priced markets — exactly where the edge function should be most discriminating.

### Per-Direction Parameterization

Alpha and floor are specified **independently for UP and DOWN** directions, since the two market sides can have different pricing characteristics:

| Parameter | Default | Meaning |
|---|---|---|
| `alpha_up` | 1.5 | Curvature for UP (above barrier) bets |
| `alpha_down` | 1.5 | Curvature for DOWN (below barrier) bets |
| `floor_up` | 0.65 | Min model prob floor for UP bets |
| `floor_down` | 0.65 | Min model prob floor for DOWN bets |

**Why per-direction?** Empirically, UP and DOWN markets on Polymarket exhibit different pricing inefficiencies:
- **DOWN markets** (e.g., "BTC below $80K") tend to be overpriced (fear premium), so lower α may be appropriate
- **UP markets** (e.g., "BTC above $120K") may require stronger model conviction due to the variance risk premium making Q-implied upside probabilities lower than physical probabilities

**Effect of α:**
- `α = 1.0`: Linear — model doubt must be ≤ market doubt (minimal edge required)
- `α = 1.5`: Default — demands proportionally more edge at high market confidence
- `α = 2.0`: Aggressive — requires significantly higher model confidence at high market prices

**Example at market_prob = 0.80 (doubt = 0.20):**
- α = 1.0 → required = max(0.65, 0.80) = 0.80
- α = 1.5 → required = max(0.65, 1 − 0.20^1.5) = max(0.65, 0.911) = 0.911
- α = 2.0 → required = max(0.65, 1 − 0.20^2.0) = max(0.65, 0.96) = 0.96

### Edge Ratio (Display Metric)

The edge ratio is still computed for display and comparison:

```
edge = model_probability / market_probability
```

An edge of 2.0× means the model thinks the event is twice as likely as the market does. However, the entry decision uses the continuous function above rather than a fixed edge-ratio threshold.

### Expected Value and Kelly Criterion

For a binary outcome token purchased at price p_market with true probability p_model:

**Expected value per dollar risked:**

```
EV = p_model · (1 − p_market) − (1 − p_model) · p_market
   = p_model − p_market
```

So the expected P&L per dollar of position is simply the probability difference. For edge = 2.0× with p_market = 0.30:

```
EV = 0.60 − 0.30 = +$0.30 per $1 of position (30% expected return)
```

**Kelly criterion** for optimal position sizing (fraction of capital to bet):

```
f* = (p_model · b − q_model) / b
```

where b = (1 − p_market) / p_market is the odds ratio and q_model = 1 − p_model. For the example above:

```
b = 0.70 / 0.30 = 2.333
f* = (0.60 · 2.333 − 0.40) / 2.333 = (1.40 − 0.40) / 2.333 = 0.429
```

Full Kelly suggests betting 42.9% of capital — aggressive. In practice, half-Kelly or quarter-Kelly is used to account for model uncertainty and estimation error in p_model. The system uses a fixed `order_size_pct` (default 5%) which is well below even quarter-Kelly for most situations, reflecting appropriate caution given model risk.

**The relationship between the edge function and Kelly:** The continuous edge function effectively implements a variable-threshold Kelly gate. By requiring p_model > h(p_market), we ensure a minimum implied Kelly fraction before entering. This is more conservative than pure Kelly (which would enter any positive-EV bet) and adapts the conservatism to the market price level.

### Entry Conditions

A trade is entered when **all** of the following hold:

| Condition | Default | Rationale |
|---|---|---|
| Model prob ≥ required(market_prob, α, floor) | α=1.5, floor=65% | Continuous edge function adapts threshold to market confidence |
| Time remaining ≥ minimum | 1 hour | Avoid entering too close to expiry where the market is nearly settled and spread costs dominate |
| No existing position in this market | — | One position at a time per market (no pyramiding) |

### Polymarket Execution

Polymarket uses a **CLOB (Central Limit Order Book)** with binary outcome tokens priced between $0 and $1:
- **YES token** at $0.65 means the market assigns 65% probability
- Buying YES and the event occurring pays $1.00 (profit of $0.35)
- Buying YES and the event not occurring pays $0.00 (loss of $0.65)

The expected return per dollar is:

```
E[return] = p_model · ($1.00 / $0.65 − 1) − (1 − p_model) · 1.0
          = p_model / p_market − 1
          = edge − 1
```

Entry prices use the orderbook **ask** (for buys) with VWAP (Volume-Weighted Average Price) for realistic fill simulation. The VWAP accounts for the fact that large orders consume multiple price levels in the orderbook:

```
VWAP = Σ(price_i · size_i) / Σ(size_i)    for levels consumed by the order
```

### Exit Strategies

1. **Take profit (TP):** A limit sell order at `entry_price × (1 + TP%)`. Default TP = 25%, so a $0.40 entry targets $0.50. The TP is capped at $0.99 (Polymarket maximum price) to avoid impossible limit orders.

2. **Trailing stop:** Activates after unrealized profit exceeds `trail_activation` (default 20%). Once active, a stop follows the peak price at a fixed distance of `trail_distance` (default 10 percentage points). If the best bid retraces from peak by more than the trail distance, the position is closed via limit sell at the current mid price.

   **Trail mechanics:**
   ```
   trail_trigger = peak_price − trail_distance
   if best_bid ≤ trail_trigger and trail_armed:
       execute limit sell at (best_bid + best_ask) / 2
   ```

3. **Expiry settlement:** If neither TP nor trailing stop triggers, the position settles at expiry:
   - $1.00 if the barrier event occurred (full win)
   - $0.00 if not (full loss)

   Settlement is binary — there is no partial payout. This makes position sizing and risk management critical, as each trade can lose 100% of the position.

---

## 10. Backtesting & Optimization

### Backtest Engine

The backtest replays the trading strategy against historical data (`probabilities.csv`) with realistic assumptions:

- **Latency modeling:** Configurable delay between signal generation and execution (default 2 minutes). During the delay, the signal must persist — if the edge disappears before execution, the trade is skipped. This prevents look-ahead bias from instantaneous execution
- **Orderbook pricing:** When available, uses actual bid/ask from historical orderbook snapshots. The look-back is non-anticipatory: the entry price is the ask at the time of execution (not signal generation), and exit prices use the bid
- **Friction:** Fee + spread per side (default 1.5%, modeling Polymarket's ~1% fee + ~0.5% spread impact)
- **One position at a time** per market (no pyramiding)
- **Mark-to-market tracking:** Capital curve is tracked per-trade for drawdown analysis

**P&L calculation per trade:**

```
For entry at price p_entry with size $S:
  Shares = S / p_entry

  If TP hit at price p_tp:
    P&L = Shares × (p_tp − p_entry) − friction × S × 2

  If trailing stop at price p_trail:
    P&L = Shares × (p_trail − p_entry) − friction × S × 2

  If held to expiry:
    P&L = Shares × (settlement − p_entry) − friction × S    (one-sided: no exit fee)
    where settlement ∈ {0.00, 1.00}
```

### Grid Search Optimization

The optimizer sweeps over parameter combinations in the continuous edge mode:

| Parameter | Typical Range | Grid |
|---|---|---|
| Alpha UP | 0.4 – 2.4 | 13 values |
| Alpha DOWN | 0.4 – 2.4 | 13 values |
| Floor UP | 0.35 – 0.80 | 10 values |
| Floor DOWN | 0.35 – 0.80 | 10 values |
| Take profit % | 5% – 40% | 8 values |
| Trail activation % | 15% – 30% | 4 values |
| Trail distance (pp) | 10 – 20 | 3 values |

**Total combinations:** Up to ~500K parameter sets (reduced by symmetry and early termination). Parallelized across `workers` cores using `ProcessPoolExecutor`.

**Legacy mode** also supports fixed edge-ratio thresholds (edge_up, edge_down) instead of the continuous function.

### Robustness Metrics

| Metric | Formula | What It Measures |
|---|---|---|
| **Sharpe ratio** | (μ̂ − r_f) / σ̂ | Risk-adjusted return per unit of volatility |
| **MtM Sharpe** | Sharpe of the mark-to-market equity curve (daily returns) | Smoothness of the capital curve |
| **Profit factor** | Σ(winning P&L) / |Σ(losing P&L)| | How much you win per dollar lost |
| **Max drawdown** | max_{t<s} [(equity_t − equity_s) / equity_t] | Worst peak-to-trough capital decline |
| **Win rate** | N_wins / N_trades | Fraction of profitable trades |
| **Bootstrap CI** | Resample markets with replacement (N_boot×), compute P&L distribution | Full distribution of possible outcomes |
| **Leave-one-out CV** | For each market, compute P&L using parameters optimized on remaining markets | Out-of-sample generalization |

**Sharpe ratio computation:**

```
Sharpe = mean(r_i) / std(r_i) · √(N_trades_per_year)
```

where r_i are per-trade returns (not annualized). The √(N_trades_per_year) scaling assumes IID returns and approximates annualization. With BTC daily markets and ~1-3 trades per week, N ≈ 100-150 per year, so the annualization factor is √100 ≈ 10.

**Caveat:** Sharpe ratios computed from <30 trades have wide confidence intervals. The standard error of the Sharpe ratio estimator is approximately:

```
SE(Sharpe) ≈ √((1 + Sharpe²/2) / (N − 1))
```

For Sharpe = 2.0 and N = 10 trades: SE ≈ 0.58. The 95% CI is [0.86, 3.14] — barely distinguishable from Sharpe = 0.

### Bootstrap Methodology

The bootstrap resamples **markets** (not individual trades) with replacement to preserve within-market trade dependencies:

```
For b = 1 to N_bootstrap:
    Sample N_markets with replacement from the market pool
    Re-run backtest on the resampled set
    Record total P&L_b

Report: mean, median, 5th/95th percentiles of {P&L_b}
```

**Why resample markets, not trades?** Trades within a single market (e.g., multiple entries/exits on "BTC > $100K Jan 31") are correlated because they share the same underlying dynamics. Resampling at the market level preserves this intra-market dependence structure (block bootstrap).

**Interpretation:**
- **5th percentile < 0:** There is a >5% chance the strategy loses money with these parameters. Proceed with caution.
- **95th percentile / 5th percentile ratio:** Measures the uncertainty range. A ratio > 10 suggests parameters are unstable.

### Leave-One-Out Cross-Validation

For N markets, perform N backtests, each holding out one market:

```
For i = 1 to N:
    Train set = all markets except market_i
    Test set = market_i
    Compute P&L_i on test set using parameters optimized on train set

CV_mean = mean(P&L_i)
CV_profitable = count(P&L_i > 0) / N
```

This estimates out-of-sample performance. If CV_mean ≪ in-sample P&L, the parameters are overfit.

### Parameter Stability Analysis

The optimizer computes parameter-level statistics across all top-performing combinations (top 20 by selected metric):

```
For each parameter p ∈ {alpha_up, alpha_down, floor_up, floor_down, tp, trail_act, trail_dist}:
    Collect values of p from top-20 results
    Report: median, mean, std, range

Suggested "robust" parameters = medians of the top performers
```

The median is preferred over the optimum because it is:
1. More stable to small perturbations in the data
2. Less likely to be at a boundary (edge of the grid)
3. A consensus of many good parameter sets rather than the single best (which may be a fluke)

### Multiple Testing and Overfitting Risk

With ~500K parameter combinations tested, the probability of finding an apparently profitable but actually random result is extremely high. This is the **multiple comparisons problem**.

**Bonferroni-corrected significance:** For 500K tests at α = 0.05, the corrected threshold is 0.05/500K = 10⁻⁷. Almost no backtest result will survive this correction with <20 markets.

**Practical mitigations:**
1. **Minimum trade count filter** (`--min-trades`): Reject parameter sets that trade fewer than N times (default 6). This eliminates sets that happen to profit on 1-2 lucky trades.
2. **Bootstrap 5th percentile:** Requires the strategy to be profitable even in adverse market resamples
3. **Cross-validation:** Tests performance on held-out data
4. **Stability analysis:** Consistent performance across *ranges* of parameters (not just one point) suggests real edge rather than overfitting
5. **Economic reasoning:** Parameters should make sense qualitatively (e.g., requiring more edge on expensive markets)

**The White reality check (2000):** A formal test of whether the best backtest result is statistically significant after accounting for data snooping. Not implemented but worth consideration for future work.

### Small Sample Caution

With fewer than ~10 historical markets, optimization results are inherently noisy:
- The absolute best parameter set may be overfit to specific market conditions
- Use the **median of top performers** (shown in stability analysis) rather than the single best
- The bootstrap 5th percentile is the most honest measure of downside risk
- The CV "Profitable" column shows on how many individual markets the parameters were profitable
- Reported Sharpe ratios and profit factors have wide confidence intervals and should be interpreted as rough guides, not precise estimates

---

## 11. Computational Acceleration

The system employs a multi-tier acceleration strategy:

| Tier | Component | Speedup | Used For | Algorithm |
|---|---|---|---|---|
| **QuantLib** | C++ analytics engine with Python bindings | ~10× vs pure Python | Heston pricing, IV extraction | Gauss-Laguerre quadrature |
| **Numba JIT (parallel)** | LLVM-compiled Python with multi-threaded prange | ~5–20× | Monte Carlo simulation kernel, SSE computation | Loop-level JIT + threading |
| **NumPy vectorization** | BLAS-backed array operations | ~10× vs loops | BL call pricing, density computation | SIMD via MKL/OpenBLAS |
| **ProcessPoolExecutor** | OS-level parallelism (bypasses GIL) | ~N× (N cores) | Multi-start DE calibration | Process-based |

**Fallback chain for Heston calibration:** QuantLib → Numba → pure Python. The system automatically detects which accelerators are available and selects the fastest.

**Computational complexity per expiry:**

| Stage | Complexity | Typical Time |
|---|---|---|
| API fetch | O(1) | ~0.5s |
| IV surface construction | O(N_options) | ~10ms |
| Heston calibration (DE) | O(N_gen × N_pop × N_strikes × N_quad) | ~5-30s |
| Breeden-Litzenberger | O(N_grid × N_quad) | ~50ms |
| Monte Carlo (1M paths) | O(N_paths × N_steps) | ~5-15s |
| **Total** | — | **~15-50s** |

The bottleneck is Heston calibration, which requires ~10⁶ characteristic function evaluations (1000 generations × 75 population × ~15 strikes).

*(Configured via `config.yaml` → `heston.use_quantlib`, `heston.enable_numba_fallback`.)*

---

## 12. References

### Core Theory

1. **Heston, S. (1993).** "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *The Review of Financial Studies*, 6(2), 327–343.

2. **Gatheral, J. & Jacquier, A. (2014).** "Arbitrage-Free SVI Volatility Surfaces." *Quantitative Finance*, 14(1), 59–71.

3. **Breeden, D. & Litzenberger, R. (1978).** "Prices of State-Contingent Claims Implicit in Option Prices." *The Journal of Business*, 51(4), 621–651.

### Numerical Methods

4. **Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007).** "The Little Heston Trap." *Wilmott Magazine*, January 2007.

5. **Lewis, A. (2001).** "A Simple Option Formula for General Jump-Diffusion and Other Exponential Levy Processes." *Envision Financial Systems and OptionCity.net*.

6. **Andersen, L. (2008).** "Efficient Simulation of the Heston Stochastic Volatility Model." *Journal of Computational Finance*, 11(3), 1–22. *(The QE scheme for variance process simulation.)*

7. **Broadie, M., Glasserman, P., & Kou, S. (1997).** "A Continuity Correction for Discrete Barrier Options." *Mathematical Finance*, 7(4), 325–349.

### Optimization and Statistics

8. **Storn, R. & Price, K. (1997).** "Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." *Journal of Global Optimization*, 11, 341–359.

9. **White, H. (2000).** "A Reality Check for Data Snooping." *Econometrica*, 68(5), 1097–1126. *(Multiple testing correction for backtested trading strategies.)*

### Foundational

10. **Dupire, B. (1994).** "Pricing with a Smile." *Risk*, 7(1), 18–20. *(Local volatility model.)*

11. **Duffie, D., Pan, J., & Singleton, K. (2000).** "Transform Analysis and Asset Pricing for Affine Jump-Diffusions." *Econometrica*, 68(6), 1343–1376. *(Affine structure and Riccati ODEs.)*

12. **Lee, R. (2004).** "The Moment Formula for Implied Volatility at Extreme Strikes." *Mathematical Finance*, 14(3), 469–480. *(Asymptotic wing behavior and moment existence.)*

13. **Cox, J., Ingersoll, J., & Ross, S. (1985).** "A Theory of the Term Structure of Interest Rates." *Econometrica*, 53(2), 385–408. *(The CIR process used for Heston variance.)*

14. **Kelly, J.L. (1956).** "A New Interpretation of Information Rate." *Bell System Technical Journal*, 35(4), 917–926. *(Optimal bet sizing under uncertainty.)*
