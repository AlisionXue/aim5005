# AIM5005 Package ： Feature Engineering

GitHub Repository: [AIM5005-Package-Feature-Engineering](https://github.com/AlisionXue/aim5005)

## Author
Chang Xue  
Email: [cxue@mail.yu.edu](mailto:cxue@mail.yu.edu)

## Assignment Description
This project focuses on implementing alternative versions of machine learning utilities, similar to the `sklearn` API. The main tasks included:

### Feature Engineering Components Implemented
- Developed alternative implementations for the following `sklearn` utilities:
  - `StandardScaler`
  - `MinMaxScaler`
  - `LabelEncoder`
- Implementations follow the `sklearn` API structure and include:
  - `fit()`
  - `transform()`
  - `fit_transform()`
  - `classes_` attribute for `LabelEncoder`

### Bug Fixes
- Identified and fixed a bug in the `MinMaxScaler` implementation that caused incorrect scaling behavior.

### Custom Unit Tests
- Created unit tests to validate the correctness of implemented components.
- Ensured that `LabelEncoder`, `StandardScaler`, and `MinMaxScaler` perform as expected.

## Modifications

### 1. StandardScaler Implementation
- Implemented `StandardScaler` to standardize data by subtracting the mean and scaling to unit variance.
- API methods included:
  - `fit()`
  - `transform()`
  - `fit_transform()`

### 2. MinMaxScaler Bug Fix
- Fixed the `MinMaxScaler` implementation to correctly normalize features within a specified range (default: 0 to 1).

### 3. LabelEncoder Implementation
- Developed `LabelEncoder` to convert categorical labels into numerical values.
- Implemented key methods:
  - `fit()`
  - `transform()`
  - `fit_transform()`
- Introduced `classes_` attribute to store unique class labels encountered during fitting.

### 4. Unit Tests
- Created unit tests to ensure `LabelEncoder` correctly encodes and decodes categorical labels.
- Validated `StandardScaler` and `MinMaxScaler` functionality with test cases.
- Updated documentation to reflect new implementations.

## Installation and Usage
Clone the repository:
```bash
git clone https://github.com/AlisionXue/aim5005.git
cd aim5005
```

Install dependencies:
```bash
pip install -e .
```

Run tests:
```bash
python -m pytest tests/test_features.py
```

## Contribution
- Contributions were made in compliance with the project guidelines.
- All modifications adhere to the given `sklearn` API structure.

## License
This project is for educational purposes and follows the guidelines provided in the assignment instructions.

