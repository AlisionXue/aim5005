## AIM 5005 - Feature Engineering Assignment
Author: Chang Xue
Email: cxue@mail.yu.edu

Assignment Description
This project focuses on creating custom implementations of machine learning features, inspired by the API. The main objectives of this assignment include:sklearn

Implementing and Testing Feature Engineering Components: The task was to create alternative implementations of common utilities such as the , , and . These implementations mimic the API and include necessary methods such as , , , and class-specific attributes like for .sklearnStandardScalerMinMaxScalerLabelEncodersklearnfittransformfit_transformclasses_LabelEncoder

Bug Fixes: A bug was identified in the original implementation, and it was required to fix this bug to ensure proper scaling behavior.MinMaxScaler

Custom Tests: Unit tests were written to ensure the correctness of the feature engineering components and to validate that the , , and classes perform as expected.LabelEncoderStandardScalerMinMaxScaler

Modifications
1. StandardScaler Implementation
Implemented a custom to standardize data by removing the mean and scaling to unit variance.StandardScaler
The custom implementation mimics the API with methods such as , , and .sklearnfittransformfit_transform
2. MinMaxScaler Bug Fix
Fixed an issue in the implementation to ensure proper normalization of features between a given range (default is 0 to 1).MinMaxScaler
3. LabelEncoder Implementation
Created a custom class to encode categorical labels into numerical values.LabelEncoder
Implemented necessary methods such as , , , and the attribute to store the classes encountered during fitting.fittransformfit_transformclasses_
4. Test Cases
Added test cases for to validate correct encoding and decoding.LabelEncoder
Test cases for and were modified to test the new and fixed implementations.StandardScalerMinMaxScaler
Installation and Usage
git clone git@github.com:Parlynx1/AIM5005-Package-Feature-Engineering.git
cd AIM5005-Package-Feature-Engineering
make install
You can ensure the tests pass by running (which defaults to makemake tests)
