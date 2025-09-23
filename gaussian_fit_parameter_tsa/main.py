"""Main entry point for the Gaussian Fit Parameter Time Series Analyzer."""

from pathlib import Path

from gaussian_fit_parameter_tsa.csv_loader import CSVDataLoader


def main() -> None:
    """Launch the Gaussian Fit Parameter Time Series Analyzer.

    This function will be called when the user clicks "Launch".
    """
    print("Gaussian Fit Parameter Time Series Analyzer - Launching...")
    print("CSV Loader with timestamp extraction ready!")

    # Initialize the CSV data loader
    loader = CSVDataLoader()  # noqa: F841

    # Example usage
    print("\nCSV Loader Features:")
    print("- Robust CSV file reading with multiple separator support")
    print("- Automatic timestamp extraction from filenames")
    print("- Channel data processing (pairs of columns are summed)")
    print("- Support for various timestamp formats:")
    print("  * YYYY-MM-DD")
    print("  * YYYY-MM-DDThhmm")
    print("  * YYYY-MM-DD-hhmm")
    print("  * YYYY-MM-DDTHH:MM:SS")
    print("  * YYYYMMDD")

    # Check if there are any CSV files in the current directory
    current_dir = Path.cwd()
    csv_files = [f for f in current_dir.glob("*.csv")]

    if csv_files:
        print(f"\nFound {len(csv_files)} CSV files in current directory:")
        for csv_file in csv_files[:5]:  # Show first 5 files
            print(f"  - {csv_file.name}")
        if len(csv_files) > 5:
            print(f"  ... and {len(csv_files) - 5} more files")
    else:
        print("\nNo CSV files found in current directory.")
        print("Place some CSV files here to test the loader functionality.")

    print("\nModule is ready for development!")


if __name__ == "__main__":
    main()
