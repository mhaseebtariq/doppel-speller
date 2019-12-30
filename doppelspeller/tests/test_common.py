from unittest import TestCase

import pandas as pd

import doppelspeller.constants as c
from doppelspeller.common import transform_title, get_words_counter, idf


class TestCommon(TestCase):
    def setUp(self):
        ground_truth = [['first', 'second', 'first', 'third', 'first'], ['first', 'first'], ['fifth']]
        ground_truth_df = pd.DataFrame(index=range(len(ground_truth)))
        ground_truth_df.loc[:, c.COLUMN_WORDS] = ground_truth
        self.ground_truth = ground_truth_df

    def test_transform_title(self):
        title = '''LKJblksd skjasl dfkjf &* 8*&&&8 GGdjsdkj--sdsd-"sdi..//' d'  k   bkjh77_asda33'''
        transformed = transform_title(title, True)
        return self.assertEqual(transformed, 'lkjblksd skjasl dfkjf 88 ggdjsdkj sdsd sdi d k bkjh77asda33')

    def test_get_ground_truth_words_counter(self):
        response = dict(get_words_counter(self.ground_truth))
        self.assertDictEqual(response, {'first': 2, 'second': 1, 'third': 1, 'fifth': 1})

    def test_idf(self):
        words_counter = get_words_counter(self.ground_truth)
        response = idf('first', words_counter, len(self.ground_truth))
        self.assertEqual(round(response, 5), 0.40547)
