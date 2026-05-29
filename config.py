from dataclasses import dataclass


@dataclass(frozen=True)
class GBMConfig:
    mu: float = 0.05
    sigma: float = 0.2
    maturity: float = 1.0
    strike: float = 1.0
    s0: float = 1.0


@dataclass(frozen=True)
class CIRConfig:
    kappa: float = 2.0
    theta: float = 0.125
    sigma: float = 0.5
    v0: float = 0.125
    maturity: float = 1.0


GBM = GBMConfig()
CIR = CIRConfig()