"""Exact references for the estimation tests, independent of pysaem.

One-compartment bolus with lognormal CL and V and proportional error: the
marginal likelihood by adaptive Gauss-Hermite quadrature, and its maximiser.
Hazard outputs: frailty models by quadrature over a fine grid, and a marker
joint with a terminal event by adaptive Gauss-Hermite quadrature.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expi

TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
DOSE = 100.0
_X, _W = np.polynomial.hermite.hermgauss(21)


def conc(cl, v, t=TIMES):
    return DOSE / v * np.exp(-cl / v * t)


def simulate(n=60, cl=2.0, v=10.0, w_cl=0.3, w_v=0.2, b=0.1, seed=7, iiv_v=True, a=0.0,
             rho=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    y = np.empty((n, TIMES.size))
    for i in range(n):
        z1, z2 = rng.standard_normal(2)
        cli = cl * np.exp(w_cl * z1)
        vi = v * np.exp(w_v * (rho * z1 + np.sqrt(1 - rho**2) * z2)) if iiv_v else v
        f = conc(cli, vi)
        y[i] = f + (a + b * f) * rng.standard_normal(TIMES.size)
        rows.append({"ID": i + 1, "TIME": 0.0, "DV": ".", "AMT": DOSE})
        rows += [{"ID": i + 1, "TIME": t, "DV": yy, "AMT": "."} for t, yy in zip(TIMES, y[i])]
    return pd.DataFrame(rows), y


def _log_joint(eta, y, cl, v, w, b, iiv, rho=0.0):
    """log p(y_i, eta_i); eta has one column per parameter with variability.

    b is the proportional coefficient, or the pair (a, b) of a combined error.
    """
    a, b = (0.0, b) if np.ndim(b) == 0 else b
    e = np.zeros(eta.shape[:-1] + (2,))
    e[..., iiv] = eta
    f = conc(cl * np.exp(e[..., 0:1]), v * np.exp(e[..., 1:2]))
    g = a + b * f
    ll = -0.5 * (((y - f) / g) ** 2).sum(-1) - np.log(g).sum(-1) - 0.5 * y.size * np.log(2 * np.pi)
    z = eta / w
    if rho:
        quad = (z[..., 0] ** 2 - 2 * rho * z[..., 0] * z[..., 1] + z[..., 1] ** 2) / (1 - rho**2)
        lp = -0.5 * quad - np.log(w).sum() - 0.5 * np.log(1 - rho**2) - np.log(2 * np.pi)
    else:
        lp = -0.5 * (z**2).sum(-1) - np.log(w).sum() - 0.5 * w.size * np.log(2 * np.pi)
    return ll + lp


def loglik(theta, y, iiv=(0, 1), centers=None, correlated=False):
    """Marginal log-likelihood; theta = (CL, V, omegas of iiv parameters..., [rho,]
    [a,] b), rho being present when correlated."""
    iiv = list(iiv)
    q = len(iiv)
    rho = theta[2 + q] if correlated else 0.0
    theta = [t for k, t in enumerate(theta) if not (correlated and k == 2 + q)]
    cl, v, w, b = theta[0], theta[1], np.asarray(theta[2:2 + q]), _error(theta, q)
    nodes = np.stack(np.meshgrid(*[_X] * q, indexing="ij"), -1).reshape(-1, q)
    weights = np.prod(np.stack(np.meshgrid(*[_W] * q, indexing="ij"), -1).reshape(-1, q), 1)
    total = 0.0
    for i in range(y.shape[0]):
        c, L = centers[i] if centers is not None else _laplace(y[i], cl, v, w, b, iiv, rho)
        eta = c + np.sqrt(2) * nodes @ L.T
        lj = _log_joint(eta, y[i], cl, v, w, b, iiv, rho)
        logw = (np.log(weights) + (nodes**2).sum(1) + 0.5 * q * np.log(2)
                + np.log(np.diag(L)).sum())
        m = (lj + logw).max()
        total += m + np.log(np.exp(lj + logw - m).sum())
    return total


def _laplace(yi, cl, v, w, b, iiv, rho=0.0):
    """Mode of the joint density and the Cholesky factor of its Laplace covariance."""
    q = len(iiv)

    def f(e):
        return -_log_joint(e, yi, cl, v, w, b, iiv, rho)
    res = minimize(f, np.zeros(q), method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 2000})
    h = 1e-4
    hess = np.empty((q, q))
    for k in range(q):
        for m in range(q):
            dk, dm = np.eye(q)[k] * h, np.eye(q)[m] * h
            hess[k, m] = (f(res.x + dk + dm) - f(res.x + dk - dm) - f(res.x - dk + dm)
                          + f(res.x - dk - dm)) / (4 * h**2)
    return res.x, np.linalg.cholesky(np.linalg.inv(hess))


def centers(theta, y, iiv=(0, 1), correlated=False):
    q = len(iiv)
    rho = theta[2 + q] if correlated else 0.0
    theta = [t for k, t in enumerate(theta) if not (correlated and k == 2 + q)]
    return [_laplace(y[i], theta[0], theta[1], np.asarray(theta[2:2 + q]), _error(theta, q),
                     list(iiv), rho) for i in range(y.shape[0])]


def _error(theta, q):
    tail = theta[2 + q:]
    return tail[0] if len(tail) == 1 else (tail[0], tail[1])


def mle(start, y, iiv=(0, 1), correlated=False):
    """Maximum likelihood estimate by direct optimisation of the quadrature."""
    c = centers(start, y, iiv, correlated)
    k = 2 + len(iiv) if correlated else None

    def unpack(u):
        t = np.exp(u)
        if correlated:
            t[k] = np.tanh(u[k])
        return t
    u0 = np.log(np.where(np.arange(len(start)) == (k if correlated else -1), 1.0, start))
    if correlated:
        u0[k] = np.arctanh(start[k])
    res = minimize(lambda u: -loglik(unpack(u), y, iiv, c, correlated), u0, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 6000, "maxfev": 6000})
    return unpack(res.x), -res.fun


def random_intercept(n_subjects=80, n_obs=6, mu=10.0, omega=2.0, a=1.0, seed=3):
    rng = np.random.default_rng(seed)
    y = mu + omega * rng.standard_normal((n_subjects, 1)) + a * rng.standard_normal(
        (n_subjects, n_obs))
    df = pd.DataFrame({"ID": np.repeat(np.arange(n_subjects), n_obs),
                       "TIME": np.tile(np.arange(n_obs, dtype=float), n_subjects),
                       "DV": y.ravel()})
    return df, y


def random_intercept_mle(y):
    n_subjects, n_obs = y.shape
    means = y.mean(1)
    s2 = ((y - means[:, None]) ** 2).sum() / (n_subjects * (n_obs - 1))
    w2 = ((means - y.mean()) ** 2).sum() / n_subjects - s2 / n_obs
    return y.mean(), np.sqrt(w2), np.sqrt(s2)


def random_intercept_loglik(mu, omega, a, y):
    n_obs = y.shape[1]
    means = y.mean(1)
    ssw = ((y - means[:, None]) ** 2).sum(1)
    tot = a**2 + n_obs * omega**2
    return float((-0.5 * (n_obs * np.log(2 * np.pi) + (n_obs - 1) * np.log(a**2) + np.log(tot)
                          + ssw / a**2 + n_obs * (means - mu) ** 2 / tot)).sum())


def occasions(n_subjects=60, n_occasions=3, n_obs=4, mu=10.0, omega=2.0, gamma=1.0, a=0.5, seed=5):
    """Random intercept with a random effect per occasion: y = mu + eta_i + kappa_ik + eps."""
    rng = np.random.default_rng(seed)
    rows, y = [], []
    for i in range(n_subjects):
        eta = omega * rng.standard_normal()
        yi = []
        for k in range(n_occasions):
            kappa = gamma * rng.standard_normal()
            for j in range(n_obs):
                v = mu + eta + kappa + a * rng.standard_normal()
                rows.append({"ID": i, "TIME": 10.0 * k + j, "DV": v, "OCC": k + 1})
                yi.append(v)
        y.append(yi)
    return pd.DataFrame(rows), np.array(y)


def occasions_loglik(theta, y, n_occasions):
    """Exact log-likelihood of occasions(); theta = (mu, omega, gamma, a)."""
    mu, omega, gamma, a = theta
    m = y.shape[1]
    S = (a**2 * np.eye(m) + omega**2 * np.ones((m, m))
         + gamma**2 * np.kron(np.eye(n_occasions), np.ones((m // n_occasions,) * 2)))
    _, logdet = np.linalg.slogdet(S)
    r = y - mu
    return float(-0.5 * (y.shape[0] * (m * np.log(2 * np.pi) + logdet)
                         + np.einsum("ia,ab,ib->", r, np.linalg.inv(S), r)))


def censored_intercept_loglik(theta, y, cens):
    """Random intercept with left-censored records (value below y where cens), by
    numerical integration over the intercept."""
    from scipy.stats import norm
    mu, omega, a = theta
    grid = np.linspace(-8, 8, 1601)
    phi = mu + omega * grid
    dens = norm.pdf(grid) * (grid[1] - grid[0])
    total = 0.0
    for yi, ci in zip(y, cens):
        z = (yi[:, None] - phi[None, :]) / a
        logp = np.where(ci[:, None], norm.logcdf(z), norm.logpdf(z) - np.log(a)).sum(0)
        top = logp.max()
        total += top + np.log((np.exp(logp - top) * dens).sum())
    return float(total)


def censored_intercept_mle(start, y, cens):
    res = minimize(lambda lt: -censored_intercept_loglik((lt[0], np.exp(lt[1]), np.exp(lt[2])),
                                                         y, cens),
                   [start[0], np.log(start[1]), np.log(start[2])], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    return (res.x[0], np.exp(res.x[1]), np.exp(res.x[2])), -res.fun


# hazard outputs -------------------------------------------------------------------------------

# one random effect: the standard normal on a fine grid, since the integrand of a subject can be
# much narrower than the prior
_Z = np.linspace(-9, 9, 6001)
_Z_WEIGHT = np.exp(-0.5 * _Z**2) * (_Z[1] - _Z[0]) / np.sqrt(2 * np.pi)


def _marginal(logp):
    """Sum over subjects of log E[exp(logp_i(z))], z standard normal; logp (subjects, _Z)."""
    lw = logp + np.log(_Z_WEIGHT)
    m = lw.max(1, keepdims=True)
    return float((m[:, 0] + np.log(np.exp(lw - m).sum(1))).sum())


def counts(n=200, lam=0.3, omega=0.6, end=10.0, seed=1):
    """Repeated events of hazard lam*exp(eta) over (0, end]: rows opening the follow-up, one row
    per event, rows closing it at end. Returns the rows and the number of events per subject."""
    rng = np.random.default_rng(seed)
    rows, k = [], []
    for i in range(n):
        rate = lam * np.exp(omega * rng.standard_normal())
        t = np.cumsum(rng.exponential(1 / rate, 200))
        t = t[t <= end]
        k.append(t.size)
        rows += ([{"ID": i, "TIME": 0.0, "DV": 0}] + [{"ID": i, "TIME": x, "DV": 1} for x in t]
                 + [{"ID": i, "TIME": end, "DV": 0}])
    return pd.DataFrame(rows), np.array(k)


def counts_loglik(theta, k, end=10.0):
    """Marginal log-likelihood of counts(); theta = (lam, omega)."""
    lam, omega = theta
    rate = lam * np.exp(omega * _Z)[None, :]
    return _marginal(k[:, None] * np.log(rate) - rate * end)


def weibull(n=300, lam=0.05, k=1.6, omega=0.5, seed=2):
    """One event per subject of hazard lam*exp(eta)*k*t^(k - 1), censored at a time uniform in
    (4, 12). Returns the rows, the times and the event flags."""
    rng = np.random.default_rng(seed)
    eta = omega * rng.standard_normal(n)
    t = (rng.exponential(size=n) / (lam * np.exp(eta))) ** (1 / k)
    c = rng.uniform(4, 12, n)
    time, d = np.minimum(t, c), (t <= c).astype(int)
    rows = []
    for i in range(n):
        rows += [{"ID": i, "TIME": 0.0, "DV": 0}, {"ID": i, "TIME": time[i], "DV": d[i]}]
    return pd.DataFrame(rows), time, d


def weibull_loglik(theta, time, d):
    """Marginal log-likelihood of weibull(); theta = (lam, k, omega)."""
    lam, k, omega = theta
    rate = lam * np.exp(omega * _Z)[None, :]
    t = time[:, None]
    return _marginal(d[:, None] * np.log(rate * k * t ** (k - 1)) - rate * t**k)


def frailty_mle(loglik, start, *args):
    """Maximiser of loglik(theta, *args) over positive parameters, and the maximum."""
    res = minimize(lambda u: -loglik(np.exp(u), *args), np.log(start), method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 20000, "maxfev": 20000})
    return np.exp(res.x), -res.fun


# a marker X = X0 exp(g t), X0 and g lognormal, with a proportional error b, and a terminal event
# of hazard h0 exp(beta X) ("exp") or h0 + beta X ("linear"); the marker is observed every 2 time
# units until the event or 12
JOINT_NAMES = ("X0", "g", "omega_X0", "omega_g", "h0", "beta", "b")
_JOINT_X, _JOINT_W = np.polynomial.hermite.hermgauss(15)


def joint_cumulative(x0, g, h0, beta, t, link):
    """Integral of the hazard from 0 to t, in closed form."""
    if link == "exp":
        return h0 / g * (expi(beta * x0 * np.exp(g * t)) - expi(beta * x0))
    return h0 * t + beta * x0 * np.expm1(g * t) / g


def joint_simulate(n=150, link="exp", seed=1):
    """Rows (YTYPE 1 for the marker, 2 for the event) and, per subject, the marker's times and
    values, the end of follow-up and whether it is the event."""
    th = dict(zip(JOINT_NAMES, (5.0, 0.1, 0.3, 0.3, 0.01, 0.15, 0.1)))
    if link == "linear":
        th.update(h0=0.005, beta=0.005)
    rng = np.random.default_rng(seed)
    visits = np.arange(0.0, 13.0, 2.0)
    rows, subjects = [], []
    for i in range(n):
        x0 = th["X0"] * np.exp(th["omega_X0"] * rng.standard_normal())
        g = th["g"] * np.exp(th["omega_g"] * rng.standard_normal())
        e = rng.exponential()
        if joint_cumulative(x0, g, th["h0"], th["beta"], 12.0, link) > e:
            end = brentq(lambda t: joint_cumulative(x0, g, th["h0"], th["beta"], t, link) - e,
                         0.0, 12.0, xtol=1e-14, rtol=1e-14)
            event = 1
        else:
            end, event = 12.0, 0
        times = visits[visits <= end]
        y = x0 * np.exp(g * times) * (1 + th["b"] * rng.standard_normal(times.size))
        rows.append({"ID": i + 1, "TIME": 0.0, "DV": 0.0, "YTYPE": 2})
        rows += [{"ID": i + 1, "TIME": t, "DV": v, "YTYPE": 1} for t, v in zip(times, y)]
        rows.append({"ID": i + 1, "TIME": end, "DV": float(event), "YTYPE": 2})
        subjects.append((times, y, end, event))
    return pd.DataFrame(rows), subjects


def _joint_log(eta, subject, theta, link):
    """log p(y_i, T_i, eta_i); eta (..., 2) on the log scale of X0 and g."""
    x0_pop, g_pop, w0, wg, h0, beta, b = theta
    times, y, end, event = subject
    x0, g = x0_pop * np.exp(eta[..., 0:1]), g_pop * np.exp(eta[..., 1:2])
    f = x0 * np.exp(g * times)
    ll = (-0.5 * ((y - f) / (b * f)) ** 2 - np.log(b * f) - 0.5 * np.log(2 * np.pi)).sum(-1)
    x0, g = x0[..., 0], g[..., 0]
    with np.errstate(over="ignore", invalid="ignore"):
        ll = ll - joint_cumulative(x0, g, h0, beta, end, link)
        if event:
            x = x0 * np.exp(g * end)
            ll = ll + np.log(h0 * np.exp(beta * x) if link == "exp" else h0 + beta * x)
        z = eta / np.array([w0, wg])
        out = ll - 0.5 * (z**2).sum(-1) - np.log(w0 * wg) - np.log(2 * np.pi)
    # far nodes overflow (inf - inf): their weight is nil
    return np.where(np.isnan(out), -np.inf, out)


def _joint_laplace(subject, theta, link):
    def f(e):
        return -_joint_log(np.asarray(e), subject, theta, link)
    res = minimize(f, np.zeros(2), method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
    h = 1e-4
    hess = np.empty((2, 2))
    for k in range(2):
        for m in range(2):
            dk, dm = np.eye(2)[k] * h, np.eye(2)[m] * h
            hess[k, m] = (f(res.x + dk + dm) - f(res.x + dk - dm) - f(res.x - dk + dm)
                          + f(res.x - dk - dm)) / (4 * h**2)
    return res.x, np.linalg.cholesky(np.linalg.inv(hess))


def joint_centers(theta, subjects, link="exp"):
    return [_joint_laplace(s, theta, link) for s in subjects]


def joint_loglik(theta, subjects, link="exp", centers=None):
    """Marginal log-likelihood by adaptive Gauss-Hermite quadrature over the two random effects;
    theta in the order of JOINT_NAMES."""
    nodes = np.stack(np.meshgrid(_JOINT_X, _JOINT_X, indexing="ij"), -1).reshape(-1, 2)
    weights = np.outer(_JOINT_W, _JOINT_W).ravel()
    centers = centers if centers is not None else joint_centers(theta, subjects, link)
    total = 0.0
    for (c, L), s in zip(centers, subjects):
        lj = _joint_log(c + np.sqrt(2) * nodes @ L.T, s, theta, link)
        logw = np.log(weights) + (nodes**2).sum(1) + np.log(2) + np.log(np.diag(L)).sum()
        m = (lj + logw).max()
        total += m + np.log(np.exp(lj + logw - m).sum())
    return total


def joint_mle(start, subjects, link="exp"):
    """Maximiser of joint_loglik from start (positive parameters, quasi-Newton on their logs, the
    quadrature recentred once), and the maximum."""
    x = np.asarray(start, float)
    for _ in range(2):
        c = joint_centers(x, subjects, link)
        res = minimize(lambda u: -joint_loglik(np.exp(u), subjects, link, c), np.log(x),
                       method="BFGS", options={"gtol": 1e-4})
        x = np.exp(res.x)
    return x, joint_loglik(x, subjects, link)
