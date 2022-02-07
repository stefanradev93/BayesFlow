import unittest

import numpy as np

from bayesflow.forward_inference import ContextGenerator, Prior, Simulator, GenerativeModel


class TestPrior(unittest.TestCase):
    """
    Tests the ContextGenerator interface.
    """

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
        expected_result = {'batchable_context': None, 'non_batchable_context': None}
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
        batchable_fun = lambda :  const_ret
        gen = ContextGenerator(batchable_context_fun=batchable_fun)
        _batch_size = 3

        # Test separate generation
        expected_output = [const_ret]*_batch_size
        out_b = gen.batchable_context(_batch_size)
        out_nb = gen.non_batchable_context()
        self.assertEqual(out_b, expected_output)
        self.assertIsNone(out_nb)
    
    def test_batchable_context_joint(self):
        const_ret = 42
        batchable_fun = lambda :  const_ret
        gen = ContextGenerator(batchable_context_fun=batchable_fun)
        _batch_size = 3

        # Test joint generation
        expected_result = {
            'batchable_context':[batchable_fun()]*_batch_size,
            'non_batchable_context': None
        }
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

    def test_batchable_context_length_correct(self):
        const_ret = 42
        batchable_fun = lambda :  const_ret
        gen = ContextGenerator(batchable_context_fun=batchable_fun)
        _batch_size = 3

        # Ensure list lengths are as expected
        out_b3 = gen.batchable_context(_batch_size)
        out_b4 = gen.batchable_context(_batch_size+1)
        self.assertEqual(len(out_b3)+1, len(out_b4))


    def test_non_batchable_context(self):
        
        const_ret = 42
        non_batchable_fun = lambda :  const_ret

        gen = ContextGenerator(non_batchable_context_fun=non_batchable_fun)
        _batch_size = 3

        # Test separate generation
        expected_output = const_ret 
        out_b = gen.batchable_context(_batch_size)
        out_nb = gen.non_batchable_context()
        self.assertIsNone(out_b)
        self.assertEqual(out_nb, expected_output)

        # Test joint generation
        expected_result = {
            'batchable_context': None,
            'non_batchable_context': expected_output 
        }
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

        # Ensure everything works even optional parameters
        non_batchable_fun = lambda :  const_ret
        gen = ContextGenerator(
            non_batchable_context_fun=non_batchable_fun,
            use_non_batchable_for_batchable=True
        )
        self.assertTrue(gen.use_non_batchable_for_batchable)
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

    def test_both_contexts(self):
        
        const_ret = 42
        batchable_fun = lambda :  const_ret
        non_batchable_fun = lambda :  const_ret

        gen = ContextGenerator(
            non_batchable_context_fun=non_batchable_fun,
            batchable_context_fun=batchable_fun
        )
        _batch_size = 3

        # Test separate generation
        expected_output_b = [const_ret]*_batch_size
        expected_output_nb = const_ret 
        out_b = gen.batchable_context(_batch_size)
        out_nb = gen.non_batchable_context()
        self.assertEqual(out_b, expected_output_b)
        self.assertEqual(out_nb, expected_output_nb)

        # Test joint generation
        expected_result = {
            'batchable_context': expected_output_b,
            'non_batchable_context': expected_output_nb 
        }
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

        # Ensure everything non-batchable feeds into batchable
        const_ret = 42
        batchable_fun = lambda b:  const_ret + b
        non_batchable_fun = lambda :  const_ret
        expected_output_b = [const_ret + non_batchable_fun()]*_batch_size
        expected_output_nb = const_ret 

        gen = ContextGenerator(
            non_batchable_context_fun=non_batchable_fun,
            batchable_context_fun=batchable_fun,
            use_non_batchable_for_batchable=True
        )
        expected_result = {
            'batchable_context': expected_output_b,
            'non_batchable_context': expected_output_nb 
        }
        self.assertTrue(gen.use_non_batchable_for_batchable)
        self.assertEqual(gen.generate_context(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), expected_result)
        self.assertEqual(gen(_batch_size), gen.generate_context(_batch_size))

class TestPrior(unittest.TestCase):
    """
    Tests the ContextGenerator interface.
    """

    def test_prior_no_context(self):
        pass

    def test_prior_with_batchable_context(self):
        pass

    def test_prior_with_non_batchable_context(self):
        pass

    def test_prior_with_botch_contexts(self):
        pass