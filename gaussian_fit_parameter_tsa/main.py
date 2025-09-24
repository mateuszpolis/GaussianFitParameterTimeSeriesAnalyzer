"""Main entry point for the Gaussian Fit Parameter Time Series Analyzer."""

import sys

from gaussian_fit_parameter_tsa.gui import main as gui_main


def main() -> None:
    """Launch the Gaussian Fit Parameter Time Series Analyzer GUI.

    This function will be called when the user clicks "Launch".
    """
    try:
        gui_main()
    except ImportError as e:
        print("Error: Required GUI dependencies not found.")
        print("Please install with: pip install -e .[gui]")
        print(f"Original error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
