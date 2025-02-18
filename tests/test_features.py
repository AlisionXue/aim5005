from aim5005.features import MinMaxScaler, StandardScaler
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
    
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line
    
    def test_min_max_scaler_zero_variance_column(self):
        """
        Test MinMaxScaler when one column has zero variance (all values are the same).
        Expected: that column should be mapped entirely to 0.
        """
        data = [[1, 2], [1, 4], [1, 6]]
        expected = np.array([[0., 0.], [0., 0.5], [0., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), f"Expected {expected}, but got {result}"


    def test_standard_scaler_zero_variance_column(self):
        """
        Test StandardScaler when one column has zero variance.
        The result for that column should be all zeros, i.e. (x - mean)/1.0 = 0.
        """
        data = [
            [2, 3, 4],
            [2, 5, 6],
            [2, 7, 8]
        ]  # Column 0 is always 2
        
        # Columns 1 and 2 will be scaled normally; column 0 should end up all zeros.
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        
        # Check that column 0 is all zeros
        zero_col = result[:, 0]
        assert np.allclose(zero_col, 0), f"Zero-variance column should transform to all zeros, got {zero_col}"
        
        # Also check the shape of the resulting array
        assert result.shape == (3, 3), f"Result shape should be (3,3), got {result.shape}"


    def test_standard_scaler_large_range_data(self):
        """
        Test StandardScaler with data covering a large numeric range, to check numerical stability.
        """
        data = [
            [1000, -1000],
            [2000, -2000],
            [3000, -3000]
        ]
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        
        # Manual calculations:
        # mean = [2000, -2000]
        # std = [816.4965809..., 816.4965809...]
        # After scaling, the data should have mean ~0 and std ~1.
        means_after = np.mean(result, axis=0)
        std_after = np.std(result, axis=0, ddof=0)
        
        # Check that the mean is near 0 and std is near 1
        assert np.allclose(means_after, 0, atol=1e-7), f"After scaling, mean should be 0. Got {means_after}"
        assert np.allclose(std_after, 1, atol=1e-7), f"After scaling, std should be 1. Got {std_after}"



if __name__ == '__main__':
    unittest.main()
