# Gaussian Fit Parameter Time Series Analyzer

A module for analyzing time series data of Gaussian fit parameters, designed to be compatible with the FIT Detector Toolkit.

## Description

This module provides tools for analyzing time series data of Gaussian fit parameters, which is essential for understanding detector performance and parameter evolution over time.

## Installation

### Development Installation

To install the module in development mode:

```bash
pip install -e .
```

### Production Installation

```bash
pip install gaussian-fit-parameter-tsa
```

## Usage

### As a FIT Detector Toolkit Module

This module is designed to be launched through the FIT Detector Toolkit launcher. When launched, it will execute the main analysis workflow.

### Direct Usage

You can also run the module directly:

```bash
python -m gaussian_fit_parameter_tsa.main
```

Or using the command-line script:

```bash
gaussian-fit-parameter-tsa
```

## Development

### Setting up Development Environment

1. Clone the repository
2. Install in development mode with dev dependencies:
   ```bash
   pip install -e .[dev]
   ```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black gaussian_fit_parameter_tsa/
```

### Type Checking

```bash
mypy gaussian_fit_parameter_tsa/
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- Pandas >= 1.3.0

## License

MIT License

## Contributing

This module is part of the FIT Detector Toolkit. Please refer to the main toolkit documentation for contribution guidelines.

## Support

For issues and questions, please use the GitHub Issues page of this repository.
