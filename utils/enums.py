from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'validation', 'test'))

WeightsInitType = IntEnum('WeightsInitType', ('normal', 'uniform'))

PreprocessingType = IntEnum('PreprocessingType', ('normalization', 'standardization'))

LoggingParamType = IntEnum('LoggingParamType', ('loss', 'metric'))
