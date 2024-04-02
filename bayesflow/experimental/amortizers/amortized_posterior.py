
from .amortizer import Amortizer


class AmortizedPosterior(Amortizer):
    INFERRED_VARIABLE = "parameters"
    OBSERVED_VARIABLE = "observables"
