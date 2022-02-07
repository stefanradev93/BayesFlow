import numpy as np


from bayesflow.exceptions import ConfigurationError


class ContextGenerator:
    """ Basic interface for a simulation module responsible for generating variables over which
    we want to amortize during simulation-based training, but do not want to perform inference on.

    The interface distinguishes between two types of context: batchable and non-batchable.

    - Batchable context variables differ for each simulation in each training batch
    - Non-batchable context varibales stay the same for each simulation in a batch, but differ across batches

    Examples for batchable context variables include experimental design variables, design matrices, etc.
    Examples for non-batchable context variables include the number of observations in an experiment, positional
    encodings, time indices, etc.

    While the latter can also be considered batchable in principle, batching them would require non-Tensor
    (i.e., non-rectangular) data structures, which usually means inefficient computations.  
    """

    def __init__(self, batchable_context_fun : callable = None, non_batchable_context_fun: callable = None,
                 use_non_batchable_for_batchable: bool = False):
        """
        Instantiates a context generator responsible for random generation of variables which vary from data set
        to data set but cannot be considered data or parameters, e.g., time indices, number of observations, etc.
        A batchable, non-batchable, or both context functions should be provided to the constructor. An optional
        argument dictates whether the outputs of the non-batchable context function should be used as inputs to
        batchable context.
        
        Parameters
        ----------
        batchable_context_fun             : callable
            A function with optional control arguments responsible for generating per-simulation set context variables
        non_batchable_context_fun         : callable 
            A function with optional control arguments responsible for generating per-batch-of-simulations context variables.
        use_non_batchable_for_batchable   : bool, optional, default: False
            Determines whether to use output of non_batchable_context_fun as input to batchable_context_fun. Only relevant
            when both context types are provided.
        """
        
        self.batchable_context_fun = batchable_context_fun
        self.non_batchable_context_fun = non_batchable_context_fun
        self.use_non_batchable_for_batchable = use_non_batchable_for_batchable

    def __call__(self, batch_size, *args, **kwargs):
        """ Wraps the method generate_context, which returns a dictionary with 
        batchable and non batchable context.
        
        Optional positional and keyword arguments are passed to the internal 
        context-generating functions or ignored if the latter are None.

         Parameters
        ----------

        batch_size : int
            The batch_size argument used for batchable context.

        Returns
        -------

        context_dict : dictionary
            A dictionary with context variables with the following keys:
            `batchable_context` : value
            `non_batchable_context` : value
        
        Note, that the values of the context variables will be None, if the
        corresponding context-generating functions have not been provided when
        initializing this object.
        """

        return self.generate_context(batch_size, *args, **kwargs)

    def batchable_context(self, batch_size, *args, **kwargs):
        """ Generates 'batch_size' context variables given optional arguments. 
        Return type is a list of context variables.
        """
        if self.batchable_context_fun is not None:
            context = [self.batchable_context_fun(*args, **kwargs) for _ in range(batch_size)]
            return context
        return None

    def non_batchable_context(self, *args, **kwargs):
        """ Generates a context variable shared across simulations in a given batch, given optional arguments.
        """
        if self.non_batchable_context_fun is not None:
            return self.non_batchable_context_fun(*args, **kwargs)
        return None

    def generate_context(self, batch_size, *args, **kwargs):
        """ Creates a dictionary with batchable and non batchable context.

         Parameters
        ----------

        batch_size : int
            The batch_size argument used for batchable context.

        Returns
        -------

        context_dict : dictionary
            A dictionary with context variables with the following keys:
            `batchable_context` : value
            `non_batchable_context` : value
        
        Note, that the values of the context variables will be None, if the
        corresponding context-generating functions have not been provided when
        initializing this object.
        """

        out_dict = {}
        out_dict['non_batchable_context'] = self.non_batchable_context()
        if self.use_non_batchable_for_batchable:
            out_dict['batchable_context'] = self.batchable_context(batch_size, out_dict['non_batchable_context'], *args, **kwargs)
        else:
            out_dict['batchable_context'] = self.batchable_context(batch_size, *args, **kwargs)
        return out_dict
        

class Prior:
    """ Basic interface for a simulation module responsible for generating random draws from a 
    prior distribution.

    The prior functions should return a np.array of simulation parameters which will be internally used
    by the GenerativeModel interface for simulations.
   
    An optional context generator (i.e., an instance of ContextGenerator) or a user-defined callable object 
    implementing the following two methods can be provided:
    - context_generator.batchable_context(batch_size)
    - context_generator.non_batchable_context()
    """

    def __init__(self, prior_fun : callable, context_generator : callable = None):
        """
        Instantiates a prior generator which will draw random parameter configurations from a user-informed prior
        distribution. No improper priors are allowed, as these may render the generative scope of a model undefined.
        
        Parameters
        ----------
        prior_fun           : callable
            A function (callbale object) with optional control arguments responsible for generating per-simulation parameters.
        context generator   : callable (default None, recommended instance of ContextGenerator)
            An optional function (ideally an instance of ContextGenerator) for generating prior context variables.
        """
        self.prior = prior_fun
        self.context_gen = context_generator
        
    def __call__(self, batch_size, *args, **kwargs):
        """Generates 'batch_size' draws from the prior given optional context generator.
        """

        # Prepare placeholder output dictionary
        out_dict = {
            'prior_draws': None,
            'batchable_context': None,
            'non_batchable_context': None
        }

        # Populate dictionary with context or leave at None
        if self.context_gen is not None:
            context_dict = self.context_gen.generate_context(batch_size, *args, **kwargs)
            out_dict['non_batchable_context'] = context_dict['non_batchable_context']
            out_dict['batchable_context'] = context_dict['batchable_context']

        # Generate prior draws according to context:
        # No context type
        if out_dict['batchable_context'] is None and out_dict['non_batchable_context'] is None:
            out_dict['prior_draws'] = np.array([self.prior(*args, **kwargs) for _ in range(batch_size)])
        
        # Only batchable context
        elif out_dict['non_batchable_context'] is None:
            out_dict['prior_draws'] = np.array([self.prior(out_dict['batchable_context'][b], *args, **kwargs) 
            for b in range(batch_size)])
            
        # Only non-batchable context
        elif out_dict['batchable_context'] is None:
            out_dict['prior_draws'] = np.array([self.prior(out_dict['non_batchable_context'], *args, **kwargs) 
            for _ in range(batch_size)])

        # Both batchable and non_batchable context
        else:
            out_dict['prior_draws'] = np.array([
                self.prior(out_dict['batchable_context'][b], 
                           out_dict['non_batchable_context'], *args, **kwargs) 
                for b in range(batch_size)])

        return out_dict

    def density(self, prior_draws):
        raise NotImplementedError('Prior density computation is under construction!')
            

class Simulator:
    """ Basic interface for a simulation module responsible for generating randomized simulations given a prior
    parameter distribution and optional context variables, given a user-provided simulation function.

    The user-provided simulator functions should return a np.array of synthetic data which will be used internally
    by the GenerativeModel interface for simulations.
   
    An optional context generator (i.e., an instance of ContextGenerator) or a user-defined callable object 
    implementing the following two methods can be provided:
    - context_generator.batchable_context(batch_size)
    - context_generator.non_batchable_context()
    """

    def __init__(self, batch_simulator_fun=None, simulator_fun=None, context_generator=None):
        """ Instantiates a data generator which will perform randomized simulations given a set of parameters and optional context.
        Either a batch_simulator_fun or simulator_fun, but not both, should be provided to instantiate a Simulator object.

        If a batch_simulator_fun is provided, the interface will assume that the function operates on batches of parameter
        vectors and context variables and will pass the latter directly to the function.

        If a simulator_fun is provided, the interface will assume thatthe function operates on single parameter vectors and
        context variables and will wrap the simulator internally to allow batched functionality.
        
        Parameters
        ----------
        batch_simulator_fun  : callable
            A function (callbale object) with optional control arguments responsible for generating a batch of simulations
            given a batch of parameters and optional context variables.
        simulator_fun       : callable
            A function (callable object) with optional control arguments responsible for generating a simulaiton given
            a single parameter vector and optional variables.
        context generator   : callable (default None, recommended instance of ContextGenerator)
            An optional function (ideally an instance of ContextGenerator) for generating prior context variables.
        """
        if (batch_simulator_fun is None) is (simulator_fun is None):
            raise ConfigurationError('Either batch_simulator_fun or simulator_fun should be provided, but not both!')
        
        self.is_batched = True if batch_simulator_fun is not None else False
        
        if self.is_batched:
            self.simulator = batch_simulator_fun
        else:
            self.simulator = simulator_fun
        self.context_gen = context_generator
        
    def __call__(self, params, *args, **kwargs):
        """ Generates simulated data given param draws and optional context variable sgenerated internally
        
        Parameters
        ----------
        params  :  np.ndarray of shape (n_sim, ...) - the parameter draws obtained from the prior.
        """
        
        # Always assume first dimension is batch dimension
        batch_size = params.shape[0]
        
        # Prepare placeholder dictionary
        out_dict = {
            'sim_data': None,
            'batchable_context': None,
            'non_batchable_context': None
        }
        
        # Populate dictionary with context or leave at None
        if self.context_gen is not None:
            context_dict = self.context_gen.generate_context(batch_size, *args, **kwargs)
            out_dict['non_batchable_context'] = context_dict['non_batchable_context']
            out_dict['batchable_context'] = context_dict['batchable_context']
        
        if self.is_batched:
            return self._simulate_batched(params, out_dict, *args, **kwargs)
        return self._simulate_non_batched(params, out_dict, *args, **kwargs)
        
    def _simulate_batched(self, params, out_dict, *args, **kwargs):
        """ Assumes a batched simulator accepting batched contexts and priors.
        """
        
        # No context type
        if out_dict['batchable_context'] is None and out_dict['non_batchable_context'] is None:
            out_dict['sim_data'] = self.simulator(params, *args, **kwargs)
            
        # Only batchable context
        elif out_dict['non_batchable_context'] is None:
            out_dict['sim_data'] = self.simulator(params, 
                                                  out_dict['batchable_context'], *args, **kwargs)

        # Only non-batchable context
        elif out_dict['batchable_context'] is None:
            out_dict['sim_data'] = self.simulator(params, 
                                                  out_dict['non_batchable_context'], *args, **kwargs)
        
        # Both batchable and non-batchable context
        else:
            out_dict['sim_data'] = self.simulator(params, 
                                                  out_dict['batchable_context'], 
                                                  out_dict['non_batchable_context'], *args, **kwargs)

        return out_dict
    
    def _simulate_non_batched(self, params, out_dict, *args, **kwargs):
        """ Assumes a non-batched simulator accepting batched contexts and priors.
        """
        
        # Extract batch size
        batch_size = params.shape[0]
        
        # No context type
        if out_dict['batchable_context'] is None and out_dict['non_batchable_context'] is None:
            out_dict['sim_data'] = np.array([self.simulator(params[b],  *args, **kwargs) for b in range(batch_size)])
            
        # Only batchable context
        elif out_dict['non_batchable_context'] is None:
            out_dict['sim_data'] = np.array([self.simulator(params[b], 
                                                            out_dict['batchable_context'][b], 
                                                            *args, **kwargs) 
                                             for b in range(batch_size)])
            
        # Only non-batchable context
        elif out_dict['batchable_context'] is None:
            out_dict['sim_data'] = np.array([self.simulator(params[b], 
                                                            out_dict['non_batchable_context'], 
                                                            *args, **kwargs) 
                                             for b in range(batch_size)])
            
        # Both batchable and non_batchable context
        else:
            out_dict['sim_data'] = np.array([self.simulator(params[b], 
                                                            out_dict['batchable_context'][b], 
                                                            out_dict['non_batchable_context'], 
                                                            *args, **kwargs) 
                                             for b in range(batch_size)])

        return out_dict
                

class GenerativeModel:
    """
    Basic interface for a generative model in a simulation-based context.
    Generally, a generative model consists of two mandatory components:
    
    - Prior : A randomized function returning random parameter draws from a prior distribution;
    - Simulator : A function which transforms the parameters into observables in a non-deterministic manner.
    """
    
    def __init__(self, prior: callable, simulator: callable, skip_test: bool = False, simulator_is_batched: bool = None):
        """
        Instantiates a generative model responsible for drawing generating params, data, and optional context.
        
        Parameters
        ----------
        prior                : callable or bayesflow.forward_inference.Prior instance
            A function returning random draws from the prior parameter distribution. Should encode
            prior knowledge about plausible parameter ranges;
        simulator            : callable or bayesflow.forward_inference.Simulator instance
            A function accepting parameter draws, optional context, and optional arguments as input
            and returning obseravble data;
        skip_test            : bool (default - False)
            If True, a forward inference pass will be performed.
        simulator_is_batched : bool (default - None), only relevant and mandatory if providing a custom simulator without
            the Simulator wrapper. 

        Important
        ----------
        If you are not using the provided Prior and Simulator wrappers for your prior and data generator,
        only functions returning a np.ndarray in the correct format will be accepted, since these will be
        wrapped internally. In addition, you need to indicate whether your simulator operates on batched of
        parameters or on single parameter vectors via tha `simulator_is_batched` argument.
        """
        
        if type(prior) is not Prior:
            self.prior = Prior(prior_fun=prior)
        else:
            self.prior = prior

        if type(simulator) is not Simulator:
            self.simulator = self._config_custom_simulator(simulator, simulator_is_batched)
        else:
            self.simulator = simulator
        self.simulator_is_bactched = self.simulator.is_batched
        
        if not skip_test:
            self._test()

    def __call__(self, batch_size, *args, **kwargs):
        """ Carries out forward inference 'batch_size' times.
        """

        # Forward inference
        prior_out = self.prior(batch_size, *args, **kwargs)
        sim_out = self.simulator(prior_out['prior_draws'], *args, **kwargs)

        # Prepare placeholder
        out_dict = {
            'prior_non_batchable_context': prior_out['non_batchable_context'],
            'prior_batchable_context': prior_out['batchable_context'],
            'prior_draws':  prior_out['prior_draws'],
            'sim_non_batchable_context': sim_out['non_batchable_context'],
            'sim_batchable_context': sim_out['batchable_context'],
            'sim_data': sim_out['sim_data'],
        }

        return out_dict

    def _config_custom_simulator(self, sim_fun, is_batched):
        """ Only called if user has provided a custom simulator not using the Simulator wrapper.
        """

        if is_batched is None:
            raise ConfigurationError('Since you are not using the Simulator wrapper, please set ' +
                                     'simulator_is_batched to True if your simulator operates on batches, ' +
                                     'otherwise set it to False.')
        elif is_batched:
            return Simulator(batch_simulator_fun=sim_fun)
        else:
            return Simulator(simulator_fun=sim_fun)
    
    def _test(self):
        """ Performs a sanity check on forward inference and some verbose information.
        """

        # Use minimal n_sim > 1
        _n_sim = 2
        out = self(_n_sim)

        # Print batch results
        try:
            p_shape_str = "(batch_size = {}, -{}".format(out["prior_draws"].shape[0], out["prior_draws"].shape[1:])
            p_shape_str = p_shape_str.replace('-(', '').replace(',)', ')')
            d_shape_str = "(batch_size = {}, -{}".format(out["sim_data"].shape[0], out["sim_data"].shape[1:])
            d_shape_str = d_shape_str.replace('-(', '').replace(',)', ')')
            print(f'Shape of parameter batch after {_n_sim} pilot simulations: {p_shape_str}')
            print(f'Shape of simulation batch after {_n_sim} pilot simulations: {d_shape_str}')
            for k, v in out.items():
                if 'context' in k:
                    name = k.replace('_', ' ').replace('sim', 'simulation').replace('non ', 'non-')
                    if v is None:
                        print(f'No {name} provided.')
                    else:
                        try:
                            print(f'Shape of {name}: {v.shape}')
                        except Exception as e:
                            print(f'Could not determine shape of {name}. Type appears to be non-array: {type(v)},\
                                    so make sure your input configurator takes cares of that!')
        except Exception as err:
            print('Could not run forward inference with specified generative model...Please re-examine model components!')
            print(err)