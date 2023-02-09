import unittest

import numpy as np

from bayesflow.simulation import ContextGenerator, Prior, Simulator


class TestContext(unittest.TestCase):
    """Tests the ``ContextGenerator`` interface."""

    def test_context_none_separate(self):
        gen = ContextGenerator()
        _batch_size = 3

        # Test separate generation
        out_b = gen.batchable_context(_batch_size)
        out_nb = gen.non_batchable_context()
        self.assertIsNone(out_b)
        self.assertIsNone(out_nb)

    def test_context_none_joint(self):
        gen = ContextGenerator()
        _batch_size = 3

        # Test joint generation
        expected_result = {"batchable_context": None, "non_batchable_context": None}
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size + 1), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

    def test_context_bool_flag(self):
        gen = ContextGenerator()
        self.assertFalse(gen.use_non_batchable_for_batchable)
        gen = ContextGenerator(use_non_batchable_for_batchable=True)
        self.assertTrue(gen.use_non_batchable_for_batchable)

    def test_batchable_context_separate(self):
        const_ret = 42
        batchable_fun = lambda: const_ret
        gen = ContextGenerator(batchable_context_fun=batchable_fun)
        _batch_size = 3

        # Test separate generation
        expected_output = [const_ret] * _batch_size
        out_b = gen.batchable_context(_batch_size)
        out_nb = gen.non_batchable_context()
        self.assertEqual(out_b, expected_output)
        self.assertIsNone(out_nb)

    def test_batchable_context_joint(self):
        const_ret = 42
        batchable_fun = lambda: const_ret
        gen = ContextGenerator(batchable_context_fun=batchable_fun)
        _batch_size = 3

        # Test joint generation
        expected_result = {"batchable_context": [batchable_fun()] * _batch_size, "non_batchable_context": None}
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

    def test_batchable_context_length_correct(self):
        const_ret = 42
        batchable_fun = lambda: const_ret
        gen = ContextGenerator(batchable_context_fun=batchable_fun)
        _batch_size = 3

        # Ensure list lengths are as expected
        out_b3 = gen.batchable_context(_batch_size)
        out_b4 = gen.batchable_context(_batch_size + 1)
        self.assertEqual(len(out_b3) + 1, len(out_b4))

    def test_non_batchable_context_separate(self):
        const_ret = 42
        non_batchable_fun = lambda: const_ret
        gen = ContextGenerator(non_batchable_context_fun=non_batchable_fun)
        _batch_size = 3

        # Test separate generation
        expected_output = non_batchable_fun()
        out_b = gen.batchable_context(_batch_size)
        out_nb = gen.non_batchable_context()
        self.assertIsNone(out_b)
        self.assertEqual(out_nb, expected_output)

    def test_non_batchable_context_joint(self):
        const_ret = 42
        non_batchable_fun = lambda: const_ret
        gen = ContextGenerator(non_batchable_context_fun=non_batchable_fun)
        _batch_size = 3

        # Test joint generation
        expected_result = {"batchable_context": None, "non_batchable_context": non_batchable_fun()}
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

    def test_both_contexts_separate(self):
        const_ret = 42
        batchable_fun = lambda: const_ret
        non_batchable_fun = lambda: const_ret

        gen = ContextGenerator(non_batchable_context_fun=non_batchable_fun, batchable_context_fun=batchable_fun)
        _batch_size = 3

        # Test separate generation
        expected_output_b = [batchable_fun()] * _batch_size
        expected_output_nb = non_batchable_fun()
        out_b = gen.batchable_context(_batch_size)
        out_nb = gen.non_batchable_context()
        self.assertEqual(out_b, expected_output_b)
        self.assertEqual(out_nb, expected_output_nb)

    def test_both_contexts_joint(self):
        const_ret = 42
        batchable_fun = lambda: const_ret
        non_batchable_fun = lambda: const_ret

        gen = ContextGenerator(non_batchable_context_fun=non_batchable_fun, batchable_context_fun=batchable_fun)
        _batch_size = 3

        # Test joint generation
        expected_result = {
            "batchable_context": [batchable_fun()] * _batch_size,
            "non_batchable_context": non_batchable_fun(),
        }
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

    def test_non_batchable_used_for_batchable(self):

        # Ensure everything non-batchable feeds into batchable
        _batch_size = 4
        const_ret = 42
        batchable_fun = lambda b: const_ret + b
        non_batchable_fun = lambda: const_ret % 3

        gen = ContextGenerator(
            non_batchable_context_fun=non_batchable_fun,
            batchable_context_fun=batchable_fun,
            use_non_batchable_for_batchable=True,
        )
        expected_result = {
            "batchable_context": [const_ret + non_batchable_fun()] * _batch_size,
            "non_batchable_context": non_batchable_fun(),
        }
        self.assertTrue(gen.use_non_batchable_for_batchable)
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))


class TestPrior(unittest.TestCase):
    """Tests the ``Prior`` interface."""

    @classmethod
    def setUpClass(cls):
        cls._batch_size = 4
        cls._dim = 3

    def test_prior_no_context(self):

        p = Prior(prior_fun=lambda: np.zeros(self._dim))

        # Prepare placeholder output dictionary
        expected_draws = (np.zeros((self._batch_size, self._dim)),)

        out = p(self._batch_size)

        self.assertIsNone(out["batchable_context"])
        self.assertIsNone(out["non_batchable_context"])
        self.assertTrue(np.all(expected_draws == out["prior_draws"]))

    def test_prior_with_batchable_context(self):

        const_ret = 2
        batchable_fun = lambda: const_ret
        gen = ContextGenerator(batchable_context_fun=batchable_fun)
        p = Prior(prior_fun=lambda c: np.ones(self._dim) * c, context_generator=gen)

        # Prepare placeholder output dictionary
        expected_context = [const_ret] * self._batch_size
        expected_draws = (batchable_fun() * np.ones((self._batch_size, self._dim)),)

        out = p(self._batch_size)

        self.assertIsNone(out["non_batchable_context"])
        self.assertEqual(out["batchable_context"], expected_context)
        self.assertTrue(np.all(expected_draws == out["prior_draws"]))

    def test_prior_with_non_batchable_context(self):

        const_ret = 42
        non_batchable_fun = lambda: const_ret
        gen = ContextGenerator(non_batchable_context_fun=non_batchable_fun)
        p = Prior(prior_fun=lambda c: np.zeros(self._dim) * c, context_generator=gen)

        # Prepare placeholder output dictionary
        expected_context = non_batchable_fun()
        expected_draws = np.zeros((self._batch_size, self._dim))

        out = p(self._batch_size)

        self.assertEqual(out["non_batchable_context"], expected_context)
        self.assertIsNone(out["batchable_context"])
        self.assertTrue(np.all(expected_draws == out["prior_draws"]))

    def test_prior_with_both_contexts(self):

        const_ret_b = 2
        const_ret_nb = 3
        batchable_fun = lambda: const_ret_b
        non_batchable_fun = lambda: const_ret_nb
        gen = ContextGenerator(non_batchable_context_fun=non_batchable_fun, batchable_context_fun=batchable_fun)

        p = Prior(prior_fun=lambda bc, nbc: np.ones(self._dim) * bc + nbc, context_generator=gen)

        # Prepare placeholder output dictionary
        expected_context_b = [const_ret_b] * self._batch_size
        expected_context_nb = non_batchable_fun()
        expected_draws = batchable_fun() * np.ones((self._batch_size, self._dim)) + non_batchable_fun()

        out = p(self._batch_size)

        self.assertEqual(out["non_batchable_context"], expected_context_nb)
        self.assertEqual(out["batchable_context"], expected_context_b)
        self.assertTrue(np.all(expected_draws == out["prior_draws"]))


class TestSimulator(unittest.TestCase):
    """Tests the ``Simulator`` interface."""

    @classmethod
    def setUpClass(cls):
        cls._batch_size = 4
        cls._p_dim = 3
        cls._x_dim = 5

    def test_sim_no_context_batched(self):

        s_fun = lambda p_draws: np.zeros((p_draws.shape[0], p_draws.shape[1], self._x_dim))
        sim = Simulator(s_fun)

        # Prepare placeholder output dictionary
        np.random.seed(42)
        expected_draws = np.zeros((self._batch_size, self._p_dim, self._x_dim))

        out = sim(np.random.randn(self._batch_size, self._p_dim))

        self.assertIsNone(out["batchable_context"])
        self.assertIsNone(out["non_batchable_context"])
        self.assertTrue(np.all(expected_draws == out["sim_data"]))

    def test_sim_no_context_non_batched(self):
        pass
