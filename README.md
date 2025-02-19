# AIM5005 Package Feature Engineering

GitHub Repository: [AIM5005-Package-Feature-Engineering](https://github.com/AlisionXue/aim5005)

## Author
Chang Xue  
Email: [cxue@mail.yu.edu](mailto:cxue@mail.yu.edu)

## Assignment Description
This project focuses on creating custom implementations of machine learning features, inspired by the `sklearn` API.  
The main objectives of this assignment include:

### Implementing and Testing Feature Engineering Components
- Created alternative implementations of common `sklearn` utilities such as:
  - `StandardScaler`
  - `MinMaxScaler`
  - `LabelEncoder`
- Implementations mimic the `sklearn` API and include necessary methods such as:
  - `fit()`
  - `transform()`
  - `fit_transform()`
  - `classes_` for `LabelEncoder`

### Bug Fixes
- A bug was identified in the original `MinMaxScaler` implementation, and a fix was applied to ensure proper scaling behavior.

### Custom Tests
- Unit tests were written to ensure the correctness of the feature engineering components.
- Validated that the `LabelEncoder`, `StandardScaler`, and `MinMaxScaler` classes perform as expected.

## Modifications

### 1. StandardScaler Implementation
- Implemented a custom `StandardScaler` to standardize data by removing the mean and scaling to unit variance.
- The custom implementation mimics the `sklearn` API with methods such as:
  - `fit()`
  - `transform()`
  - `fit_transform()`

### 2. MinMaxScaler Bug Fix
- Fixed an issue in the `MinMaxScaler` implementation to ensure proper normalization of features within a given range (default: 0 to 1).

### 3. LabelEncoder Implementation
- Created a custom class to encode categorical labels into numerical values.
- Implemented necessary methods such as:
  - `fit()`
  - `transform()`
  - `fit_transform()`
- Included an attribute `classes_` to store the unique class labels encountered during fitting.

### 4. Test Cases
- Added unit tests to validate correct encoding and decoding for `LabelEncoder`.
- Created test cases for `StandardScaler` and `MinMaxScaler` to verify correctness.
- Updated documentation to reflect the new implementations.

## Installation and Usage
Clone the repository:
```bash
git clone https://github.com/AlisionXue/aim5005.git
cd aim5005
