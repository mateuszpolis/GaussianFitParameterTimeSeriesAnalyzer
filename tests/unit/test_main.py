"""Tests for the main module."""

import pytest

from gaussian_fit_parameter_tsa.main import main


def test_main_function_exists() -> None:
    """Test that the main function exists and is callable."""
    assert callable(main)


def test_main_function_runs() -> None:
    """Test that the main function runs without error."""
    # This should not raise an exception
    main()


def test_main_function_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that the main function produces expected output."""
    main()
    captured = capsys.readouterr()
    assert "Gaussian Fit Parameter Time Series Analyzer" in captured.out
    assert "Launching..." in captured.out
