import pandas as pd
import numpy as np
from datetime import datetime

import pytest

import batch

def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),        
    ]
    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    input_df = pd.DataFrame(data, columns=columns)
    actual_result = batch.prepare_data(input_df, ['PUlocationID', 'DOlocationID'])

    expected_data = [
        ("-1", "-1", dt(1, 2), dt(1, 10), 8.0),
        ("1", "1", dt(1, 2), dt(1, 10), 8.0),
    ]
    expected_result = pd.DataFrame(expected_data, columns=columns + ['duration'])

    # 2 asserts here: 1 for all non float columns and 1 for float columns only (OMG)!!!!!!
    assert actual_result[columns].reset_index(drop=True).equals(expected_result[columns].reset_index(drop=True))
    np.testing.assert_array_almost_equal(actual_result['duration'].values, expected_result['duration'].values)
    