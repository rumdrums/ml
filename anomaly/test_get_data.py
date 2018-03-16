import unittest

import collections
from get_data import *
import pandas as pd
from pandas.util.testing import assert_frame_equal

mock_record = collections.namedtuple('mock_record', 'one two three')

mock_fields = {
  'one': 'int',
  'two': 'string',
  'three': 'int'
}

class TestDataSet(unittest.TestCase):

  def test_array(self):
    mock_records = [
      mock_record(one=0, two='cat1', three=0),
      mock_record(one=1, two='cat2', three=100),
      mock_record(one=2, two='cat1', three=200),
      mock_record(one=3, two='cat3', three=300),
      mock_record(one=4, two='cat2', three=400),
      mock_record(one=5, two='cat4', three=500),
    ]

    a = DataSet(mock_records, mock_fields)
    b = a.get_array()

    c = pd.DataFrame({
      'cat1': [1, 0, 1, 0, 0, 0],
      'cat2': [0, 1, 0, 0, 1, 0],
      'cat3': [0, 0, 0, 1, 0, 0],
      'cat4': [0, 0, 0, 0, 0, 1],
      'one': [0, 1, 2, 3, 4, 5],
      'three': [0, 100, 200, 300, 400, 500]
    })

    assert_frame_equal(b.sort_index(axis=1),c.sort_index(axis=1), check_dtype=False)
