""" THIS IS NOT USED!
- MERT INDIBI, 20 August 2023"""

from abc import ABC, abstractmethod, abstractproperty
import time
from typing import Any
import numpy as np


class TwoBlockADMMBase(ABC):
    """Abstract base class for ADMM optimization algorithms."""

    @abstractmethod
    def __init__(self, args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def __call__(self):
        """Iterate over the algorithm until convergence or reaching maximum iterations."""
        pass

    @abstractmethod
    def check_convergence(self):
        pass

    @abstractmethod
    def plot_algorithm_run(self):
        """Plot the objective function, primal and dual residuals; and the step size over algorithm run."""
        pass

    @abstractmethod
    def iterate(self):
        """Iterate the algorithm one step and update the variables."""
        pass

    @abstractproperty
    def it(self):
        """Current iteration number."""
        pass

    @abstractproperty
    def r(self):
        """List of primal residuals for each iteration of the ADMM algorithm."""
        pass

    @abstractproperty
    def s(self):
        """List of dual residuals of each iteration of the ADMM algorithm."""
        pass

    @abstractproperty
    def objective(self):
        """List of objective values of each iteration of the ADMM algorithm."""
        pass
    
    @abstractproperty
    def max_it(self):
        """Maximum number of iterations."""
        pass

    @abstractproperty
    def rhos(self):
        """List of ADMM step sizes in each iteration of the ADMM algorithm."""
        pass

    @abstractproperty
    def elapsed_time(self):
        """Wall clock time spent on algorithm solution"""
        pass

    @abstractproperty
    def verbosity(self):
        """Verbosity level of the algorithm."""
        pass

    @abstractproperty
    def err_tol(self):
        pass