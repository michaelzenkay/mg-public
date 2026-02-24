"""Task specification dataclasses for classification, regression, and risk heads."""

from dataclasses import dataclass


@dataclass
class CatSpec:
    name: str          # model head name
    key: str           # column name in ds2024.cat_keys
    num_classes: int
    weight: float = 1.0


@dataclass
class RegSpec:
    name: str          # head name in the model
    key: str           # column name in ds2024.reg_keys, e.g. "mAs" or "KVP"
    weight: float = 1.0
    metric: str = "mse"   # or "mae"


@dataclass
class RiskSpec:
    name: str
    horizons: int = 5
    weight: float = 5.0
    key: str = "risk"
