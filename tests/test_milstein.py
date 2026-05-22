import numpy as np
import pytest
import sys
from pathlib import Path
from pytest import approx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import milstein_gbm, MU_GBM, SIGMA_GBM, S0_GBM, MATURITY_GBM

def test_milstein_gbm():
    n = 2 ** 6
    t = np.linspace(0, MATURITY_GBM, n + 1)
    dt = t[1:] - t[:-1]
    dw = np.random.normal(loc=0, scale=np.sqrt(dt), size=n)
    w = np.r_[0, np.cumsum(dw)]
    s_true = S0_GBM * np.exp((MU_GBM - SIGMA_GBM ** 2 / 2) * t + SIGMA_GBM * w)
    s_true_maturity = s_true[-1]
    s_maturity = milstein_gbm(S0_GBM, dt, dw)
    assert  s_maturity == approx(s_true_maturity, abs=1e-3)
    
def test_milstein_cir():
    assert True

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
