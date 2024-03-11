from enum import Enum


class SetType(Enum):
    """Data set type."""

    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class WeightsInitType(Enum):
    """Weights initialization type enum."""

    NORMAL = 1
    UNIFORM = 2


class PreprocessingType(Enum):
    """Preprocessing type enum."""

    NORMALIZATION = 1
    STANDARDIZATION = 2


class LoggingParamType(Enum):
    """Logging parameter type enum."""

    LOSS = 1
    METRIC = 2
