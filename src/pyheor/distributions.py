"""
Probability distributions for PSA sampling.

Each distribution supports the `sample(n)` method and can be parameterized
in multiple ways (e.g., Beta by mean/sd or alpha/beta).
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class Distribution(ABC):
    """Base class for probability distributions used in PSA."""
    
    @abstractmethod
    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n samples from the distribution."""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass


class Beta(Distribution):
    """Beta distribution.
    
    Parameters
    ----------
    mean : float, optional
        Mean of the distribution (0 < mean < 1).
    sd : float, optional
        Standard deviation.
    alpha : float, optional
        Alpha (shape1) parameter.
    beta : float, optional
        Beta (shape2) parameter.
        
    Notes
    -----
    Provide either (mean, sd) or (alpha, beta). When using (mean, sd),
    the method of moments is used to derive alpha and beta.
    
    Commonly used for:
    - Transition probabilities
    - Utility weights
    """
    
    def __init__(self, mean: Optional[float] = None, sd: Optional[float] = None,
                 alpha: Optional[float] = None, beta: Optional[float] = None):
        if alpha is not None and beta is not None:
            self.alpha = float(alpha)
            self.beta = float(beta)
        elif mean is not None and sd is not None:
            if sd == 0 or sd is None:
                # Degenerate: very tight around mean
                self.alpha = mean * 10000
                self.beta = (1 - mean) * 10000
            else:
                var = sd ** 2
                common = mean * (1 - mean) / var - 1
                if common <= 0:
                    raise ValueError(
                        f"Cannot parameterize Beta with mean={mean}, sd={sd}. "
                        f"Variance too large relative to mean*(1-mean)."
                    )
                self.alpha = mean * common
                self.beta = (1 - mean) * common
        else:
            raise ValueError("Provide either (mean, sd) or (alpha, beta)")
        
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(f"Alpha ({self.alpha:.4f}) and beta ({self.beta:.4f}) must be > 0")
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.beta(self.alpha, self.beta, size=n)
    
    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)
    
    def __repr__(self):
        return f"Beta(α={self.alpha:.2f}, β={self.beta:.2f}, mean={self.mean:.4f})"


class Gamma(Distribution):
    """Gamma distribution.
    
    Parameters
    ----------
    mean : float, optional
        Mean of the distribution.
    sd : float, optional
        Standard deviation.
    shape : float, optional
        Shape parameter (k).
    rate : float, optional
        Rate parameter (1/scale).
        
    Notes
    -----
    Commonly used for costs (always positive, right-skewed).
    """
    
    def __init__(self, mean: Optional[float] = None, sd: Optional[float] = None,
                 shape: Optional[float] = None, rate: Optional[float] = None):
        if shape is not None and rate is not None:
            self.shape = float(shape)
            self.rate = float(rate)
        elif mean is not None and sd is not None:
            if sd == 0:
                self.shape = mean * 10000
                self.rate = 10000
            else:
                self.shape = (mean / sd) ** 2
                self.rate = mean / (sd ** 2)
        else:
            raise ValueError("Provide either (mean, sd) or (shape, rate)")
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.gamma(self.shape, 1.0 / self.rate, size=n)
    
    @property
    def mean(self):
        return self.shape / self.rate
    
    def __repr__(self):
        return f"Gamma(shape={self.shape:.2f}, rate={self.rate:.4f}, mean={self.mean:.2f})"


class Normal(Distribution):
    """Normal (Gaussian) distribution.
    
    Parameters
    ----------
    mean : float
        Mean.
    sd : float
        Standard deviation.
    """
    
    def __init__(self, mean: float = 0, sd: float = 1):
        self.mean_val = float(mean)
        self.sd = float(sd)
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.normal(self.mean_val, self.sd, size=n)
    
    @property
    def mean(self):
        return self.mean_val
    
    def __repr__(self):
        return f"Normal(μ={self.mean_val:.4f}, σ={self.sd:.4f})"


class LogNormal(Distribution):
    """Log-normal distribution.
    
    Parameters
    ----------
    meanlog : float, optional
        Mean on log scale.
    sdlog : float, optional
        SD on log scale.
    mean : float, optional
        Mean on natural scale.
    sd : float, optional
        SD on natural scale.
        
    Notes
    -----
    Commonly used for relative risks, hazard ratios, and odds ratios.
    """
    
    def __init__(self, meanlog: Optional[float] = None, sdlog: Optional[float] = None,
                 mean: Optional[float] = None, sd: Optional[float] = None):
        if meanlog is not None and sdlog is not None:
            self.meanlog = float(meanlog)
            self.sdlog = float(sdlog)
        elif mean is not None and sd is not None:
            var = sd ** 2
            self.meanlog = np.log(mean ** 2 / np.sqrt(var + mean ** 2))
            self.sdlog = np.sqrt(np.log(1 + var / mean ** 2))
        else:
            raise ValueError("Provide either (meanlog, sdlog) or (mean, sd)")
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.lognormal(self.meanlog, self.sdlog, size=n)
    
    @property
    def mean(self):
        return np.exp(self.meanlog + self.sdlog ** 2 / 2)
    
    def __repr__(self):
        return f"LogNormal(μ_log={self.meanlog:.4f}, σ_log={self.sdlog:.4f})"


class Uniform(Distribution):
    """Uniform distribution.
    
    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound.
    """
    
    def __init__(self, low: float = 0, high: float = 1):
        self.low = float(low)
        self.high = float(high)
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=n)
    
    @property
    def mean(self):
        return (self.low + self.high) / 2
    
    def __repr__(self):
        return f"Uniform({self.low:.4f}, {self.high:.4f})"


class Triangular(Distribution):
    """Triangular distribution.
    
    Parameters
    ----------
    low : float
        Lower bound.
    mode : float
        Mode (peak).
    high : float
        Upper bound.
    """
    
    def __init__(self, low: float, mode: float, high: float):
        self.low = float(low)
        self.mode = float(mode)
        self.high = float(high)
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.triangular(self.low, self.mode, self.high, size=n)
    
    @property
    def mean(self):
        return (self.low + self.mode + self.high) / 3
    
    def __repr__(self):
        return f"Triangular({self.low:.2f}, {self.mode:.2f}, {self.high:.2f})"


class Dirichlet(Distribution):
    """Dirichlet distribution.
    
    Parameters
    ----------
    alpha : array-like
        Concentration parameters.
        
    Notes
    -----
    Used for sampling multiple transition probabilities from a single row
    of a transition matrix (conjugate prior for multinomial).
    """
    
    def __init__(self, alpha):
        self.alpha = np.asarray(alpha, dtype=float)
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.random.dirichlet(self.alpha, size=n)
    
    @property
    def mean(self):
        return self.alpha / self.alpha.sum()
    
    def __repr__(self):
        return f"Dirichlet(α={self.alpha})"


class Fixed(Distribution):
    """Fixed (degenerate) distribution — always returns the same value.
    
    Parameters
    ----------
    value : float
        The fixed value.
    """
    
    def __init__(self, value: float):
        self.value = float(value)
    
    def sample(self, n: int = 1) -> np.ndarray:
        return np.full(n, self.value)
    
    @property
    def mean(self):
        return self.value
    
    def __repr__(self):
        return f"Fixed({self.value})"
