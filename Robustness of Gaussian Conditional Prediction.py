###########################################################################
# Robustness of Gaussian Conditional Prediction Beyond Gaussianity
# Numerical Experiments -- Python implementation
#
# Companion simulation code for the paper:
#
#   "Robustness of Gaussian Conditional Prediction Beyond Gaussianity:
#    Elliptical, Cumulant, and Copula Perspectives"
#
# The script reproduces:
#   (1) Exact robustness under Gaussian and Student-t scale-mixture models.
#   (2) Covariance-preserving symmetric Gaussian-mixture perturbations.
#   (3) Gaussian copula experiments (normal-score vs naive predictor).
#   (4) Sensitivity of excess risk to covariance parameters.
#   (5) Misspecified-copula experiment (t-copula with lognormal marginals).
#   (6) Figure: excess risk vs perturbation magnitude.
#   (7) Automatic LaTeX generation of Tables 1-4.
#
# Author: Francisco Maturana
# Date: September 2026
###########################################################################

import numpy as np
from scipy.stats import norm, t as student_t
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
# Common spatial setup
# =========================================================================

n = 20
Delta = 1.0
x_locs = np.arange(1, n + 1) * Delta
x0 = 0.0
sigma2_0 = 1.0
rho_0 = 5.0


def cov_exponential(h, sigma2=1.0, rho=5.0):
    """Exponential covariance function."""
    return sigma2 * np.exp(-np.abs(h) / rho)


def build_sigma(n=20, Delta=1.0, sigma2=1.0, rho=5.0, x0=0.0):
    """Build covariance matrix for (Z_0, Z_1, ..., Z_n)."""
    locations = np.concatenate(([x0], np.arange(1, n + 1) * Delta))
    distances = np.abs(locations[:, None] - locations[None, :])
    return sigma2 * np.exp(-distances / rho)


def kriging_weights(Sigma):
    """
    Return simple-kriging coefficients a, observation covariance Sigma_ZZ,
    and cross-covariance Sigma_0Z for a zero-mean model.
    """
    Sigma_0Z = Sigma[0, 1:]
    Sigma_ZZ = Sigma[1:, 1:]
    a = np.linalg.solve(Sigma_ZZ, Sigma_0Z)
    return a, Sigma_ZZ, Sigma_0Z


def mc_mean_se(samples):
    """Return Monte Carlo mean and standard error."""
    samples = np.asarray(samples, dtype=float)
    N = samples.size
    return samples.mean(), samples.std(ddof=1) / np.sqrt(N)


def check_positive_definite(A, name="matrix"):
    """Raise a clear error if A is not positive definite."""
    eigmin = np.linalg.eigvalsh(A).min()
    if eigmin <= 0:
        raise ValueError(
            f"{name} must be positive definite; "
            f"minimum eigenvalue = {eigmin:.6g}"
        )
    return eigmin


# =========================================================================
# 1. Exact robustness: Gaussian and Student-t scale mixture
# =========================================================================

def run_exact_robustness(N=40000, Sigma=None, seed=123):
    """Gaussian and covariance-matched Student-t scale-mixture experiments."""
    rng = np.random.default_rng(seed)

    if Sigma is None:
        Sigma = build_sigma(n, Delta, sigma2_0, rho_0, x0)

    m = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    a, _, _ = kriging_weights(Sigma)

    # Gaussian field
    Zg = rng.standard_normal(size=(N, m)) @ L.T
    Z0g = Zg[:, 0]
    Zg_obs = Zg[:, 1:]
    Zhat_G_gauss = Zg_obs @ a

    # Under Gaussianity, the conditional mean equals the Gaussian predictor.
    err_gauss = (Zhat_G_gauss - Z0g) ** 2
    R_star_gauss, se_star_gauss = mc_mean_se(err_gauss)
    R_G_gauss, se_G_gauss = R_star_gauss, se_star_gauss

    # Student-t Gaussian scale mixture, rescaled so Cov(Zt) = Sigma.
    nu = 5.0
    Y = rng.standard_normal(size=(N, m)) @ L.T
    U = rng.chisquare(df=nu, size=N)
    W = nu / U
    E_W = nu / (nu - 2.0)
    W_scaled = W / E_W

    Zt = np.sqrt(W_scaled)[:, None] * Y
    Z0t = Zt[:, 0]
    Zt_obs = Zt[:, 1:]
    Zhat_G_t = Zt_obs @ a

    # For this elliptical scale mixture, the conditional mean is the same
    # linear predictor.
    err_t = (Zhat_G_t - Z0t) ** 2
    R_star_t, se_star_t = mc_mean_se(err_t)
    R_G_t, se_G_t = R_star_t, se_star_t

    return {
        "R_star_gauss": R_star_gauss,
        "se_star_gauss": se_star_gauss,
        "R_G_gauss": R_G_gauss,
        "se_G_gauss": se_G_gauss,
        "R_star_t": R_star_t,
        "se_star_t": se_star_t,
        "R_G_t": R_G_t,
        "se_G_t": se_G_t,
        "Sigma": Sigma,
    }


# =========================================================================
# 2. Covariance-preserving symmetric Gaussian-mixture perturbations
# =========================================================================

def gaussian_log_density(Z, mean, Sigma_ZZ):
    """Log-density of N(mean, Sigma_ZZ) evaluated row-wise at Z."""
    Z = np.asarray(Z)
    mean = np.asarray(mean)
    diff = Z - mean[None, :]

    sign, logdet = np.linalg.slogdet(Sigma_ZZ)
    if sign <= 0:
        raise ValueError("Sigma_ZZ must be positive definite.")

    solved = np.linalg.solve(Sigma_ZZ, diff.T).T
    mah = np.einsum("ij,ij->i", diff, solved)

    n_dim = Sigma_ZZ.shape[0]
    log_coef = -0.5 * (n_dim * np.log(2.0 * np.pi) + logdet)
    return log_coef - 0.5 * mah


def maximum_shift_scale(Sigma, direction):
    """
    Return the boundary scale c_max for which

        Sigma - c^2 direction direction^T

    is positive semidefinite.

    For a positive-definite Sigma,
        c_max = 1 / sqrt(direction^T Sigma^{-1} direction).
    """
    direction = np.asarray(direction, dtype=float)

    if direction.shape != (Sigma.shape[0],):
        raise ValueError(
            f"direction must have shape ({Sigma.shape[0]},), "
            f"got {direction.shape}."
        )

    if np.allclose(direction, 0.0):
        raise ValueError("direction must be nonzero.")

    q = direction @ np.linalg.solve(Sigma, direction)
    return 1.0 / np.sqrt(q)


def make_admissible_shift(Sigma, direction, fraction=0.8):
    """
    Construct mu = fraction * c_max * direction, where c_max is the
    positive-semidefinite boundary. Requiring fraction < 1 keeps
    Sigma - mu mu^T strictly positive definite.
    """
    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction must lie strictly between 0 and 1.")

    c_max = maximum_shift_scale(Sigma, direction)
    mu = fraction * c_max * np.asarray(direction, dtype=float)

    Sigma_within = Sigma - np.outer(mu, mu)
    check_positive_definite(Sigma_within, "Sigma - mu mu^T")
    return mu


def simulate_mixture(N, mu, Sigma, rng):
    """
    Simulate the covariance-preserving symmetric Gaussian mixture

        X = S mu + eps,

    where P(S=+1)=P(S=-1)=1/2 and

        eps ~ N(0, Sigma - mu mu^T).

    Hence E[X] = 0 and Cov(X) = Sigma.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma_within = Sigma - np.outer(mu, mu)
    check_positive_definite(Sigma_within, "Sigma - mu mu^T")

    L_within = np.linalg.cholesky(Sigma_within)
    signs = rng.choice((-1.0, 1.0), size=N)
    eps = rng.standard_normal(size=(N, mu.size)) @ L_within.T

    return signs[:, None] * mu[None, :] + eps


def conditional_mean_mixture(Z_obs, mu, Sigma_within):
    """
    Exact E[Z_0 | Z] for the symmetric two-component Gaussian mixture.

    Both components have covariance Sigma_within and means +/- mu.
    """
    mu = np.asarray(mu, dtype=float)
    mu0 = mu[0]
    muZ = mu[1:]

    Sigma_0Z = Sigma_within[0, 1:]
    Sigma_ZZ = Sigma_within[1:, 1:]
    a_within = np.linalg.solve(Sigma_ZZ, Sigma_0Z)

    # Component-specific conditional means.
    m_plus = mu0 + (Z_obs - muZ) @ a_within
    m_minus = -mu0 + (Z_obs + muZ) @ a_within

    # Posterior component probabilities based on the observed subvector.
    log_f_plus = gaussian_log_density(Z_obs, muZ, Sigma_ZZ)
    log_f_minus = gaussian_log_density(Z_obs, -muZ, Sigma_ZZ)

    # Stable two-class softmax.
    mx = np.maximum(log_f_plus, log_f_minus)
    f_plus = np.exp(log_f_plus - mx)
    f_minus = np.exp(log_f_minus - mx)
    w_plus = f_plus / (f_plus + f_minus)

    return w_plus * m_plus + (1.0 - w_plus) * m_minus


def run_mixture(N=40000, Sigma=None, mu=None, seed=456):
    """
    Compare the Gaussian predictor with the exact conditional mean under

        X ~ 0.5 N(mu, Sigma - mu mu^T)
           + 0.5 N(-mu, Sigma - mu mu^T).

    The mixture has zero mean and total covariance Sigma.
    """
    rng = np.random.default_rng(seed)

    if Sigma is None:
        Sigma = build_sigma(n, Delta, sigma2_0, rho_0, x0)

    if mu is None:
        raise ValueError("mu must be specified.")

    mu = np.asarray(mu, dtype=float)

    if mu.shape != (Sigma.shape[0],):
        raise ValueError(
            f"mu must have shape ({Sigma.shape[0]},), got {mu.shape}."
        )

    Sigma_within = Sigma - np.outer(mu, mu)
    check_positive_definite(Sigma_within, "Sigma - mu mu^T")

    # Gaussian reference predictor uses the preserved total covariance Sigma.
    a, _, _ = kriging_weights(Sigma)

    # Simulate the covariance-preserving non-Gaussian mixture.
    X = simulate_mixture(N, mu, Sigma, rng)
    Z0 = X[:, 0]
    Zobs = X[:, 1:]

    Zhat_G = Zobs @ a

    # Exact conditional mean under the actual mixture law.
    Zstar = conditional_mean_mixture(Zobs, mu, Sigma_within)

    err_star = (Zstar - Z0) ** 2
    err_G = (Zhat_G - Z0) ** 2

    R_star, se_star = mc_mean_se(err_star)
    R_G, se_G = mc_mean_se(err_G)

    # Paired difference gives the excess-risk estimate and its MC uncertainty.
    risk_difference = err_G - err_star

    return R_star, se_star, R_G, se_G, risk_difference


# =========================================================================
# 3. Gaussian copula experiment: normal-score vs naive predictor
# =========================================================================

def run_gaussian_copula(
    N=40000,
    Sigma=None,
    seed=789,
    mu_F=0.0,
    sigma_F=1.0,
):
    """
    Gaussian-copula field with lognormal marginals.

    The normal-score procedure is exact for the conditional mean on the
    Gaussianised scale. The backtransformed predictor is not asserted to be
    the original-scale conditional expectation.
    """
    rng = np.random.default_rng(seed)

    if Sigma is None:
        Sigma = build_sigma(n, Delta, sigma2_0, rho_0, x0)

    m = Sigma.shape[0]

    # Convert covariance to a correlation matrix for the latent Gaussian field.
    sd = np.sqrt(np.diag(Sigma))
    Sigma_Y = Sigma / np.outer(sd, sd)
    L_Y = np.linalg.cholesky(Sigma_Y)

    # Kriging coefficients on the latent Gaussian scale.
    a_Y, _, _ = kriging_weights(Sigma_Y)

    Y = rng.standard_normal(size=(N, m)) @ L_Y.T
    Z = np.exp(mu_F + sigma_F * Y)

    Z0 = Z[:, 0]
    Zobs = Z[:, 1:]
    Yobs = Y[:, 1:]

    # Exact conditional mean on the Gaussianised scale.
    Yhat_G = Yobs @ a_Y

    # Conventional transform-predict-backtransform predictor.
    Z_NS = np.exp(mu_F + sigma_F * Yhat_G)

    # Naive linear predictor on the original scale, retained as a comparator.
    # This uses the same correlation-derived weights as in the original design;
    # it is not claimed to be the optimal linear predictor for lognormal Z.
    Z_naive = Zobs @ a_Y

    err_NS = (Z_NS - Z0) ** 2
    err_naive = (Z_naive - Z0) ** 2

    R_NS, se_NS = mc_mean_se(err_NS)
    R_naive, se_naive = mc_mean_se(err_naive)

    return R_NS, se_NS, R_naive, se_naive


# =========================================================================
# 4. Sensitivity analysis: rho x sigma2 grid
# =========================================================================

def run_sensitivity(
    direction,
    rho_grid=(2.0, 5.0, 10.0),
    sig2_grid=(0.5, 1.0, 2.0),
    strong_fraction=0.8,
    mild_fraction=0.25,
    N=20000,
    seed_base=1000,
):
    """
    Excess risk for a mild covariance-preserving mixture perturbation.

    At every (rho, sigma2) combination, the strong shift is defined as the
    same fraction of that covariance matrix's admissibility boundary. The
    mild shift is mild_fraction times that strong shift. This avoids changing
    admissibility merely because the covariance parameters change.
    """
    excess = np.zeros((len(rho_grid), len(sig2_grid)))
    se_exc = np.zeros_like(excess)

    for i, rho in enumerate(rho_grid):
        for j, s2 in enumerate(sig2_grid):
            Sigma = build_sigma(
                n=n,
                Delta=Delta,
                sigma2=s2,
                rho=rho,
                x0=x0,
            )

            mu_strong_local = make_admissible_shift(
                Sigma,
                direction,
                fraction=strong_fraction,
            )
            mu_mild_local = mild_fraction * mu_strong_local

            _, _, _, _, diff = run_mixture(
                N=N,
                Sigma=Sigma,
                mu=mu_mild_local,
                seed=seed_base + 10 * i + j,
            )

            excess[i, j] = diff.mean()
            se_exc[i, j] = diff.std(ddof=1) / np.sqrt(N)

    return excess, se_exc


# =========================================================================
# 5. Misspecified copula: true t-copula with lognormal marginals
# =========================================================================

def t_copula_oracle(
    Zobs_samples,
    Sigma_Y,
    nu,
    M_inner=200,
    seed=2024,
    mu_F=0.0,
    sigma_F=1.0,
):
    """
    Monte Carlo approximation to E[Z_0 | Z_1, ..., Z_n] under a t-copula
    with lognormal marginals.

    If T ~ t_nu(0, Sigma_Y), then for the partition (T_0, T_Z),

        T_0 | T_Z=t

    is univariate Student-t with df nu+n, location

        Sigma_0Z Sigma_ZZ^{-1} t,

    and scale squared

        ((nu + t' Sigma_ZZ^{-1} t) / (nu+n))
        * (Sigma_00 - Sigma_0Z Sigma_ZZ^{-1} Sigma_Z0).
    """
    rng = np.random.default_rng(seed)

    N = Zobs_samples.shape[0]
    n_obs = Zobs_samples.shape[1]

    Sigma_0Z = Sigma_Y[0, 1:]
    Sigma_ZZ = Sigma_Y[1:, 1:]

    a = np.linalg.solve(Sigma_ZZ, Sigma_0Z)
    schur = Sigma_Y[0, 0] - Sigma_0Z @ a
    nu_post = nu + n_obs

    # Z = F_LN^{-1}(F_t(T)), so T = F_t^{-1}(F_LN(Z)).
    F_Z = norm.cdf((np.log(Zobs_samples) - mu_F) / sigma_F)

    # Protect against numerical 0/1 values before applying the t quantile.
    eps = np.finfo(float).eps
    F_Z = np.clip(F_Z, eps, 1.0 - eps)
    T_Z = student_t.ppf(F_Z, df=nu)

    mu_star = T_Z @ a

    solved = np.linalg.solve(Sigma_ZZ, T_Z.T).T
    quad = np.einsum("ij,ij->i", T_Z, solved)

    scale2_star = (nu + quad) / nu_post * schur
    scale2_star = np.maximum(scale2_star, 1e-14)

    draws = student_t.rvs(
        df=nu_post,
        size=(N, M_inner),
        random_state=rng,
    )

    T0 = mu_star[:, None] + np.sqrt(scale2_star)[:, None] * draws

    U0 = student_t.cdf(T0, df=nu)
    U0 = np.clip(U0, eps, 1.0 - eps)
    Z0_draws = np.exp(mu_F + sigma_F * norm.ppf(U0))

    return Z0_draws.mean(axis=1)


def run_misspecified_copula(
    N=10000,
    Sigma=None,
    nu=4.0,
    seed=2025,
    M_inner=200,
    mu_F=0.0,
    sigma_F=1.0,
):
    """True t-copula; compare oracle, normal-score, and naive predictors."""
    rng = np.random.default_rng(seed)

    if Sigma is None:
        Sigma = build_sigma(n, Delta, sigma2_0, rho_0, x0)

    m = Sigma.shape[0]

    sd = np.sqrt(np.diag(Sigma))
    Sigma_Y = Sigma / np.outer(sd, sd)
    L_Y = np.linalg.cholesky(Sigma_Y)
    a_Y, _, _ = kriging_weights(Sigma_Y)

    # Simulate multivariate t latent variables with correlation Sigma_Y.
    z = rng.standard_normal(size=(N, m))
    W = rng.chisquare(df=nu, size=N)
    T = (z @ L_Y.T) * np.sqrt(nu / W)[:, None]

    # Transform t-copula uniforms to lognormal marginals.
    U = student_t.cdf(T, df=nu)
    eps = np.finfo(float).eps
    U = np.clip(U, eps, 1.0 - eps)
    Z = np.exp(mu_F + sigma_F * norm.ppf(U))

    Z0 = Z[:, 0]
    Zobs = Z[:, 1:]

    # Oracle conditional expectation under the true t-copula.
    Z_oracle = t_copula_oracle(
        Zobs,
        Sigma_Y,
        nu,
        M_inner=M_inner,
        seed=seed + 1,
        mu_F=mu_F,
        sigma_F=sigma_F,
    )

    # Normal-score predictor under the misspecified Gaussian-copula assumption.
    Y_latent = (np.log(Zobs) - mu_F) / sigma_F
    Yhat_G = Y_latent @ a_Y
    Z_NS = np.exp(mu_F + sigma_F * Yhat_G)

    # Naive original-scale comparator using the same spatial weights.
    Z_naive = Zobs @ a_Y

    err_oracle = (Z_oracle - Z0) ** 2
    err_NS = (Z_NS - Z0) ** 2
    err_naive = (Z_naive - Z0) ** 2

    R_oracle, se_oracle = mc_mean_se(err_oracle)
    R_NS, se_NS = mc_mean_se(err_NS)
    R_naive, se_naive = mc_mean_se(err_naive)

    return R_oracle, se_oracle, R_NS, se_NS, R_naive, se_naive


# =========================================================================
# 6. Excess-risk vs perturbation magnitude
# =========================================================================

def run_perturbation_curve(
    mu_strong,
    scales,
    N=20000,
    Sigma=None,
    seed=3030,
):
    """Excess risk as a function of perturbation scale."""
    if Sigma is None:
        Sigma = build_sigma(n, Delta, sigma2_0, rho_0, x0)

    excess = np.zeros(len(scales))
    se = np.zeros(len(scales))

    for k, s in enumerate(scales):
        _, _, _, _, diff = run_mixture(
            N=N,
            Sigma=Sigma,
            mu=s * mu_strong,
            seed=seed + k,
        )
        excess[k] = diff.mean()
        se[k] = diff.std(ddof=1) / np.sqrt(N)

    return excess, se


# =========================================================================
# Run all experiments
# =========================================================================

print("=" * 70)
print("Running experiments...")
print("=" * 70)

Sigma0 = build_sigma(n, Delta, sigma2_0, rho_0, x0)

# (1) Exact robustness
res_exact = run_exact_robustness(N=40000, Sigma=Sigma0)
print("[1/6] Exact robustness done.")

# (2) Covariance-preserving symmetric Gaussian mixtures
#
# Direction has nonzero entries at x0 and x10. Its magnitude is chosen as
# 80% of the positive-definiteness boundary for Sigma0.
mu_direction = np.zeros(n + 1)
mu_direction[0] = 1.0
mu_direction[10] = 1.0

strong_fraction = 0.8
mild_fraction = 0.25

mu_strong = make_admissible_shift(
    Sigma0,
    mu_direction,
    fraction=strong_fraction,
)
mu_mild = mild_fraction * mu_strong

print(
    "    Strong-shift nonzero value = "
    f"{mu_strong[0]:.6f}; "
    f"boundary fraction = {strong_fraction:.2f}"
)

R_star_mild, se_star_mild, R_G_mild, se_G_mild, diff_mild = run_mixture(
    N=40000,
    Sigma=Sigma0,
    mu=mu_mild,
    seed=456,
)

R_star_str, se_star_str, R_G_str, se_G_str, diff_str = run_mixture(
    N=40000,
    Sigma=Sigma0,
    mu=mu_strong,
    seed=457,
)

print("[2/6] Gaussian mixtures done.")

# (3) Gaussian copula
R_NS, se_NS, R_naive, se_naive = run_gaussian_copula(
    N=40000,
    Sigma=Sigma0,
)
print("[3/6] Gaussian copula done.")

# (4) Sensitivity analysis
rho_grid = (2.0, 5.0, 10.0)
sig2_grid = (0.5, 1.0, 2.0)

excess_sens, se_sens = run_sensitivity(
    mu_direction,
    rho_grid=rho_grid,
    sig2_grid=sig2_grid,
    strong_fraction=strong_fraction,
    mild_fraction=mild_fraction,
    N=20000,
)
print("[4/6] Sensitivity analysis done.")

# (5) Misspecified copula
(
    R_oracle,
    se_oracle,
    R_NS_mis,
    se_NS_mis,
    R_naive_mis,
    se_naive_mis,
) = run_misspecified_copula(
    N=10000,
    Sigma=Sigma0,
    nu=4.0,
)
print("[5/6] Misspecified copula done.")

# (6) Perturbation curve
scales = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])

excess_curve, se_curve = run_perturbation_curve(
    mu_strong,
    scales,
    N=20000,
    Sigma=Sigma0,
)
print("[6/6] Perturbation curve done.")


# =========================================================================
# Figure: excess risk vs perturbation magnitude
# =========================================================================

fig, ax = plt.subplots(figsize=(5.5, 3.6))

ax.errorbar(
    scales,
    excess_curve,
    yerr=1.96 * se_curve,
    fmt="o",
    capsize=3,
    label="Excess risk (mixture)",
)

# Descriptive quadratic fit to the smallest three nonzero perturbations.
p = np.polyfit(scales[:3], excess_curve[:3], 2)
xs = np.linspace(0.0, 1.05, 100)

ax.plot(
    xs,
    np.polyval(p, xs),
    "--",
    label="Quadratic fit (first three points)",
)

ax.set_xlabel(r"Perturbation scale $s$")
ax.set_ylabel(r"$R(\widehat Z_G)-R(Z^\star)$")
ax.set_title("Excess risk vs perturbation magnitude")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("excess_risk_vs_perturbation.pdf")
fig.savefig("excess_risk_vs_perturbation.png", dpi=300)

print("Figure saved to excess_risk_vs_perturbation.pdf / .png")


# =========================================================================
# LaTeX table generation
# =========================================================================

latex_table_1 = r"""\begin{table}[htbp]
    \centering
    \begin{tabular}{lcccc}
        \toprule
        Model & $R(Z^\star;x_0)$ & MC s.e. & $R(\widehat Z_G;x_0)$ & MC s.e. \\
        \midrule
        Gaussian & %.3f & %.3f & %.3f & %.3f \\
        Student-$t$ scale mixture & %.3f & %.3f & %.3f & %.3f \\
        \bottomrule
    \end{tabular}
    \caption{Estimated mean squared prediction errors for Gaussian prediction
    and the mean-square optimal predictor under two random field models with
    the same covariance structure. Values are based on $N=40{,}000$
    simulated replicates. Monte Carlo standard errors (MC s.e.) are computed
    as the sample standard deviation of the squared prediction errors divided
    by $\sqrt{N}$. In both cases the predictors coincide exactly under the
    model, consistent with the exact robustness results.}
    \label{tab:numerical}
\end{table}""" % (
    res_exact["R_star_gauss"],
    res_exact["se_star_gauss"],
    res_exact["R_G_gauss"],
    res_exact["se_G_gauss"],
    res_exact["R_star_t"],
    res_exact["se_star_t"],
    res_exact["R_G_t"],
    res_exact["se_G_t"],
)


latex_table_2 = r"""\begin{table}[htbp]
    \centering
    \begin{tabular}{lcccc}
        \toprule
        Experiment & Predictor & $R(\cdot;x_0)$ & MC s.e. & Risk difference \\
        \midrule
        Gaussian mixture (mild) & True $Z^\star$ & %.3f & %.3f & -- \\
        Gaussian mixture (mild) & Gaussian $\widehat Z_G$ & %.3f & %.3f & %.3f \\
        Gaussian mixture (strong) & True $Z^\star$ & %.3f & %.3f & -- \\
        Gaussian mixture (strong) & Gaussian $\widehat Z_G$ & %.3f & %.3f & %.3f \\
        Gaussian copula (lognormal) & Normal-score $Z_\mathrm{NS}$ & %.3f & %.3f & -- \\
        Gaussian copula (lognormal) & Naive Gaussian on $Z$ & %.3f & %.3f & %.3f \\
        \bottomrule
    \end{tabular}
    \caption{Mean squared prediction errors for covariance-preserving
    symmetric Gaussian-mixture perturbations and for a Gaussian copula random
    field with lognormal marginals. Monte Carlo standard errors are reported
    in the fourth column. For the mixture experiments, the risk difference is
    $R(\widehat Z_G)-R(Z^\star)$. For the Gaussian copula experiment, it is
    the risk of the naive original-scale predictor minus that of the
    transform--predict--backtransform normal-score predictor.}
    \label{tab:additional-numerical}
\end{table}""" % (
    R_star_mild,
    se_star_mild,
    R_G_mild,
    se_G_mild,
    R_G_mild - R_star_mild,
    R_star_str,
    se_star_str,
    R_G_str,
    se_G_str,
    R_G_str - R_star_str,
    R_NS,
    se_NS,
    R_naive,
    se_naive,
    R_naive - R_NS,
)


rows_sens = []
for i, rho in enumerate(rho_grid):
    cells = " & ".join(
        "%.3f" % excess_sens[i, j]
        for j in range(len(sig2_grid))
    )
    rows_sens.append("        %.1f & %s \\\\" % (rho, cells))

sens_body = "\n".join(rows_sens)
max_sens_se = se_sens.max()

latex_table_3 = (r"""\begin{table}[htbp]
    \centering
    \begin{tabular}{lccc}
        \toprule
        $\rho$ & $\sigma^2=0.5$ & $\sigma^2=1$ & $\sigma^2=2$ \\
        \midrule
%s
        \bottomrule
    \end{tabular}
    \caption{Excess risk $R(\widehat Z_G)-R(Z^\star)$ for the mild
    covariance-preserving Gaussian-mixture perturbation as a function of the
    range parameter $\rho$ and variance $\sigma^2$. At each covariance
    setting, the perturbation direction is fixed and its magnitude is scaled
    relative to that covariance matrix's positive-definiteness boundary.
    Each entry is based on $N=20{,}000$ replicates. The largest Monte Carlo
    standard error of the excess-risk estimates is %.4f.}
    \label{tab:sensitivity}
\end{table}""" % (sens_body, max_sens_se))


latex_table_4 = r"""\begin{table}[htbp]
    \centering
    \begin{tabular}{lccc}
        \toprule
        Predictor & $R(\cdot;x_0)$ & MC s.e. & Excess risk vs oracle \\
        \midrule
        Oracle $t$-copula conditional mean & %.3f & %.3f & -- \\
        Normal-score predictor (Gaussian copula) & %.3f & %.3f & %.3f \\
        Naive original-scale predictor & %.3f & %.3f & %.3f \\
        \bottomrule
    \end{tabular}
    \caption{Mean squared prediction errors under copula misspecification.
    The true copula is Student-$t$ with $\nu=4$, whereas the normal-score
    predictor assumes a Gaussian copula. The oracle uses the true $t$-copula
    conditional expectation, approximated by Monte Carlo. Results are based
    on $N=10{,}000$ replicates with $M=200$ inner Monte Carlo draws per
    observation.}
    \label{tab:misspec}
\end{table}""" % (
    R_oracle,
    se_oracle,
    R_NS_mis,
    se_NS_mis,
    R_NS_mis - R_oracle,
    R_naive_mis,
    se_naive_mis,
    R_naive_mis - R_oracle,
)


# =========================================================================
# Console output
# =========================================================================

print("\n" + "=" * 70)
print("Summary of results")
print("=" * 70)

print("\n[1] Exact robustness (Gaussian, Student-t):")
print(
    "    Gaussian  R(Z*) = %.4f (s.e. %.4f), "
    "R(Z_G) = %.4f (s.e. %.4f)"
    % (
        res_exact["R_star_gauss"],
        res_exact["se_star_gauss"],
        res_exact["R_G_gauss"],
        res_exact["se_G_gauss"],
    )
)
print(
    "    Student-t R(Z*) = %.4f (s.e. %.4f), "
    "R(Z_G) = %.4f (s.e. %.4f)"
    % (
        res_exact["R_star_t"],
        res_exact["se_star_t"],
        res_exact["R_G_t"],
        res_exact["se_G_t"],
    )
)

print("\n[2] Covariance-preserving mixture perturbations:")
print(
    "    Mild:   R(Z*) = %.4f, R(Z_G) = %.4f, excess = %.4f"
    % (R_star_mild, R_G_mild, R_G_mild - R_star_mild)
)
print(
    "    Strong: R(Z*) = %.4f, R(Z_G) = %.4f, excess = %.4f"
    % (R_star_str, R_G_str, R_G_str - R_star_str)
)
print(
    "    Mild excess paired MC s.e.   = %.6f"
    % (diff_mild.std(ddof=1) / np.sqrt(diff_mild.size))
)
print(
    "    Strong excess paired MC s.e. = %.6f"
    % (diff_str.std(ddof=1) / np.sqrt(diff_str.size))
)

print("\n[3] Gaussian copula (lognormal marginals):")
print("    R(Z_NS)    = %.4f (s.e. %.4f)" % (R_NS, se_NS))
print("    R(Z_naive) = %.4f (s.e. %.4f)" % (R_naive, se_naive))
print("    Risk difference naive - NS = %.4f" % (R_naive - R_NS))

print("\n[4] Sensitivity of excess risk (mild mixture):")
print("    rho \\ sigma^2     0.5        1.0        2.0")
for i, rho in enumerate(rho_grid):
    print(
        "    %.1f          %.4f     %.4f     %.4f"
        % (
            rho,
            excess_sens[i, 0],
            excess_sens[i, 1],
            excess_sens[i, 2],
        )
    )
print("    Maximum paired MC s.e. = %.6f" % max_sens_se)

print("\n[5] Misspecified copula (t-copula, nu=4):")
print(
    "    Oracle       R = %.4f (s.e. %.4f)"
    % (R_oracle, se_oracle)
)
print(
    "    Normal-score R = %.4f (s.e. %.4f), excess vs oracle = %.4f"
    % (R_NS_mis, se_NS_mis, R_NS_mis - R_oracle)
)
print(
    "    Naive        R = %.4f (s.e. %.4f), excess vs oracle = %.4f"
    % (R_naive_mis, se_naive_mis, R_naive_mis - R_oracle)
)

print("\n[6] Perturbation curve (excess risk vs scale s):")
for k, s in enumerate(scales):
    print(
        "    s = %.2f    excess = %.4f  (paired s.e. %.4f)"
        % (s, excess_curve[k], se_curve[k])
    )

print("\n\nLaTeX Table 1:\n")
print(latex_table_1)

print("\n\nLaTeX Table 2:\n")
print(latex_table_2)

print("\n\nLaTeX Table 3 (sensitivity):\n")
print(latex_table_3)

print("\n\nLaTeX Table 4 (misspecified copula):\n")
print(latex_table_4)
