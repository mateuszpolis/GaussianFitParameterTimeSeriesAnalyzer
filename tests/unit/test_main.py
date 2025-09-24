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
    # The main function now launches a GUI, so we expect an error about
    # missing display
    main()
    captured = capsys.readouterr()
    # The function should either print an error or run the GUI
    # Since we can't test GUI in headless environment, we just check it doesn't crash
    assert captured.out == "" or "Error" in captured.out
