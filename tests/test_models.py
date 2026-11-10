"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean, daily_max, daily_min
from inflammation.models import patient_normalise

@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None),
        ([[-5, 10,20], [4,4,4], [float('nan'),3, 6]], [[0,0.5,1], [1,1,1], [0,0.5,1]], ValueError),
        ([[1e7, 1e8, 1e9], [ 0, 0, 1e99]], [[0.01,0.1,1], [0,0,1]], None)
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            patient_normalise(np.array(test))
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test that mean function works for an array of zeros."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test), expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1, 2, 9, 4], [3, 4, 6, 12], [5, 6, 3, 1] ], [5, 6, 9, 12]),
        ([ [1, 2], [3, 4], [5, 6] ], [5, 6]),
        ([ [4, 9, 13, 28], [23, 6, 1,15], [18, 4, 51, 7]], [23, 9, 51, 28])
    ])
def test_daily_max_integers(test, expected):
    """Test that mean function works for an array of positive integers."""
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test), expected)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1, 2, 9, 4], [3, 4, 6, 12], [5, 6, 3, 1] ], [1, 2, 3, 1]),
        ([ [1, 2], [3, 4], [5, 6] ], [1, 2]),
        ([ [4, 9, 13, 28], [23, 6, 1,15], [18, 4, 51, 7]], [4, 4, 1, 7])
    ])    
def test_daily_min_integers(test, expected):
    """Test that mean function works for an array of positive integers."""

    npt.assert_array_equal(daily_min(test), expected)
    
def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])

