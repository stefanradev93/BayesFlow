
import pytest

from bayesflow.experimental import


@pytest.fixture(scope="module", params=[True, False], name=)
def context_prior(request):
    use_context = request.param
    if not use_context:
        return None

    @


@pytest.fixture(scope="module")
def parameter_prior():
    pass

@pytest.fixture(scope="module")
def simulator():
    pass


@pytest.fixture(scope="module", params=)
def generative_model(context_prior, parameter_prior, simulator):
    return GenerativeModel()
