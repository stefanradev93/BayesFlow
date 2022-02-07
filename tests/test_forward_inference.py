import unittest

import numpy as np

from bayesflow.forward_inference import ContextGenerator, Prior, Simulator, GenerativeModel


class TestContextgenerator(unittest.TestCase):
    """
    This unit test ensures the inetrgity of the ContextGenerator interface.
    """

    def test_context_none(self):
        gen = ContextGenerator()
        _batch_size = 3

        # Test separate generation
        out_b = gen.batchable_context(_batch_size)
        out_nb = gen.non_batchable_context()
        self.assertIsNone(out_b)
        self.assertIsNone(out_nb)

        # Test joint generation
        expected = {'batchable_context': None, 'non_batchable_context': None}
        self.assertEqual(gen.generate_context(_batch_size), expected)
        self.assertEqual(gen(_batch_size + 1), expected)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

    def test_batchable_context(self):
        pass



