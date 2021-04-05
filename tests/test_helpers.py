import unittest

from bayesflow.default_settings import DEFAULT_SETTING_INVARIANT_NET, DEFAULT_SETTING_INVERTIBLE_NET, MetaDictSetting
from bayesflow.exceptions import ConfigurationError
from bayesflow.helpers import build_meta_dict, merge_left_into_right


class TestMergeLeftIntoRight(unittest.TestCase):
    """
    Test bayesflow.helpers.merge_left_into_right
    """
    def test_merge_left_into_right_emptyL_emptyR(self):
        left_dict = {}
        right_dict = {}
        correct_result = {}
        self.assertEqual(correct_result, merge_left_into_right(left_dict=left_dict, right_dict=right_dict))

    def test_merge_left_into_right_emptyL(self):
        left_dict = {}
        right_dict = {'X': 100, 'Y': 200, 'Z': 300}
        correct_result = {'X': 100, 'Y': 200, 'Z': 300}
        self.assertEqual(correct_result, merge_left_into_right(left_dict=left_dict, right_dict=right_dict))

    def test_merge_left_into_right_emptyR(self):
        left_dict = {'A': 1, 'B': 2, 'C': 3}
        right_dict = {}
        correct_result = {'A': 1, 'B': 2, 'C': 3}
        self.assertEqual(correct_result, merge_left_into_right(left_dict=left_dict, right_dict=right_dict))

    def test_merge_left_into_right_distinct_LR(self):
        left_dict = {'A': 1, 'B': 2, 'C': 3}
        right_dict = {'X': 100, 'Y': 200, 'Z': 300}
        correct_result = {'A': 1, 'B': 2, 'C': 3,
                          'X': 100, 'Y': 200, 'Z': 300}
        self.assertEqual(correct_result, merge_left_into_right(left_dict=left_dict, right_dict=right_dict))

    def test_merge_left_into_right_unique_element_in_L(self):
        left_dict = {'NEW': 999}
        right_dict = {'X': 100, 'Y': 200, 'Z': 300}
        correct_result = {'NEW': 999, 'X': 100, 'Y': 200, 'Z': 300}
        self.assertEqual(correct_result, merge_left_into_right(left_dict=left_dict, right_dict=right_dict))

    def test_merge_left_into_right_non_distinct_LR(self):
        left_dict = {'A': 1, 'B': 2, 'C': 3}
        right_dict = {'A': -999, 'C': -999,
                      'X': 100, 'Y': 200, 'Z': 300}
        correct_result = {'A': 1, 'B': 2, 'C': 3,
                          'X': 100, 'Y': 200, 'Z': 300}
        self.assertEqual(correct_result, merge_left_into_right(left_dict=left_dict, right_dict=right_dict))

    def test_merge_left_into_right_nested_LR_empty_R_nested_dict(self):
        left_dict = {'A': 1, 'B': 2, 'C': 3,
                     'D':
                         {'J': -100, 'K': -200, 'L': -300}}
        right_dict = {'A': -999, 'C': -999, 'D': {},
                      'X': 100, 'Y': 200, 'Z': 300}
        correct_result = {'A': 1, 'B': 2, 'C': 3,
                          'D':
                              {'J': -100, 'K': -200, 'L': -300},
                          'X': 100, 'Y': 200, 'Z': 300}
        self.assertEqual(correct_result, merge_left_into_right(left_dict=left_dict, right_dict=right_dict))

    def test_merge_left_into_right_nested_LR_nonempty_nested_R(self):
        left_dict = {'A': 1, 'B': 2, 'C': 3,
                     'D':
                         {'J': -100, 'K': -200, 'L': -300}}
        right_dict = {'A': -999, 'C': -999,
                      'D':
                          {'J': -999, 'K': -999, 'L': -999},
                      'X': 100, 'Y': 200, 'Z': 300}
        correct_result = {'A': 1, 'B': 2, 'C': 3,
                          'D':
                              {'J': -100, 'K': -200, 'L': -300},
                          'X': 100, 'Y': 200, 'Z': 300}
        self.assertEqual(correct_result, merge_left_into_right(left_dict=left_dict, right_dict=right_dict))


class TestBuildMetaDict(unittest.TestCase):
    """
    Tests for bayesflow.helpers.build_meta_dict
    """
    def test_build_meta_dict_empty_violate_mandatory(self):
        my_dict = {}
        default_setting = MetaDictSetting(
            meta_dict={
                'A': 1,
                'B': {
                    'c': [2, 3],
                    'd': 'xyz'
                },
                'alpha': 1,
                'permute': True
            },
            mandatory_fields=['A']
        )
        with self.assertRaises(ConfigurationError):
            build_meta_dict(my_dict, default_setting)

    def test_build_meta_dict_empty_fulfil_mandatory(self):
        my_dict = {}
        default_setting = MetaDictSetting(
            meta_dict={
                'A': 1,
                'B': {
                    'c': [2, 3],
                    'd': 'xyz'
                },
                'alpha': 1,
                'permute': True
            },
            mandatory_fields=[]
        )
        self.assertEqual(build_meta_dict(my_dict, default_setting), default_setting.meta_dict)

    def test_build_meta_dict_nonempty_fulfil_mandatory(self):
        my_dict = {'A': 100}
        default_setting = MetaDictSetting(
            meta_dict={
                'A': 1,
                'B': {
                    'c': [2, 3],
                    'd': 'xyz'
                },
                'alpha': 1,
                'permute': True
            },
            mandatory_fields=['A']
        )
        correct_result = {
                'A': 100,
                'B': {
                    'c': [2, 3],
                    'd': 'xyz'
                },
                'alpha': 1,
                'permute': True
            }
        self.assertEqual(build_meta_dict(my_dict, default_setting), correct_result)

    def test_build_meta_dict_nonempty_fulfil_mandatory_nested_partial_override(self):
        my_dict = {'A': 100,
                   'B': {
                       'c': [100, 3]
                   },
                   }
        default_setting = MetaDictSetting(
            meta_dict={
                'A': 1,
                'B': {
                    'c': [2, 3],
                    'd': 'xyz'
                },
                'alpha': 1,
                'permute': True
            },
            mandatory_fields=['A']
        )
        correct_result = {
                'A': 100,
                'B': {
                    'c': [100, 3],
                    'd': 'xyz'
                },
                'alpha': 1,
                'permute': True
            }
        self.assertEqual(build_meta_dict(my_dict, default_setting), correct_result)

    def test_build_meta_dict_nonempty_fulfil_mandatory_nested_full_override(self):
        my_dict = {'A': 100,
                   'B': {
                       'c': [100, 100],
                       'd': 'ZZZ'
                   },
                   }
        default_setting = MetaDictSetting(
            meta_dict={
                'A': 1,
                'B': {
                    'c': [2, 3],
                    'd': 'xyz'
                },
                'alpha': 1,
                'permute': True
            },
            mandatory_fields=['A']
        )
        correct_result = {
                'A': 100,
                'B': {
                    'c': [100, 100],
                    'd': 'ZZZ'
                },
                'alpha': 1,
                'permute': True
            }
        self.assertEqual(build_meta_dict(my_dict, default_setting), correct_result)

    def test_build_meta_dict_invariant_empty_my_dict(self):
        my_dict = {}
        default_setting = DEFAULT_SETTING_INVARIANT_NET
        correct_result = DEFAULT_SETTING_INVARIANT_NET.meta_dict
        
        self.assertEqual(build_meta_dict(my_dict, default_setting), correct_result)

    def test_build_meta_dict_invariant_override_default(self):
        my_dict = {
            'n_dense_s1': 100,
            'n_equiv': 200,
            'dense_s1_args': {'units': 300},
        }
        default_setting = DEFAULT_SETTING_INVARIANT_NET
        correct_result = {
                'n_dense_s1': 100,
                'n_dense_s2': 2,
                'n_dense_s3': 2,
                'n_equiv':    200,
                'dense_s1_args': {'activation': 'relu', 'units': 300},
                'dense_s2_args': {'activation': 'relu', 'units': 64},
                'dense_s3_args': {'activation': 'relu', 'units': 32}
            }

        self.assertEqual(build_meta_dict(my_dict, default_setting), correct_result)

    def test_build_meta_dict_invertible_violate_mandatory(self):
        my_dict = {'n_coupling_layers': 100}
        default_setting = DEFAULT_SETTING_INVERTIBLE_NET

        with self.assertRaises(ConfigurationError):
            build_meta_dict(my_dict, default_setting)

    def test_build_meta_dict_invertible_minimal(self):
        my_dict = {'n_params': 100}
        default_setting = DEFAULT_SETTING_INVERTIBLE_NET
        correct_result = {
            'n_params': 100,
            'n_coupling_layers': 4,
            's_args': {
                'units': [128, 128],
                'activation': 'elu',
                'initializer': 'glorot_uniform',
            },
            't_args': {
                'units': [128, 128],
                'activation': 'elu',
                'initializer': 'glorot_uniform',
            },
            'alpha': 1.9,
            'permute': True
        }
        self.assertEqual(build_meta_dict(my_dict, default_setting), correct_result)

    def test_build_meta_dict_invertible_override_default(self):
        my_dict = {'n_params': 100}
        default_setting = DEFAULT_SETTING_INVERTIBLE_NET
        correct_result = DEFAULT_SETTING_INVERTIBLE_NET.meta_dict

        self.assertEqual(build_meta_dict(my_dict, default_setting), correct_result)
