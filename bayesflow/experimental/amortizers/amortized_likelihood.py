
from .amortizer import Amortizer


class AmortizedLikelihood(Amortizer):
    INFERRED_VARIABLE = "observables"
    OBSERVED_VARIABLE = "parameters"

