"""CSV loader module for Gaussian fit parameter time series analysis."""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_timestamp_from_filename(path: str) -> Optional[datetime]:
    """Extract timestamp from filename using various patterns.

    Try to extract timestamp from the filename using patterns like:
    YYYY-MM-DD, YYYY-MM-DDThhmm, YYYY-MM-DD-hhmm, and a generic ISO pattern.
    Examples: "2022-09-19-0T-time-PMA0.csv" should parse date 2022-09-19
    (00:00 time if no time present).

    Args:
        path: File path to extract timestamp from

    Returns:
        datetime object if timestamp found, None otherwise
    """
    filename = os.path.basename(path)

    # Pattern 1: Generic ISO pattern (YYYY-MM-DDTHH:MM:SS or similar)
    # Most specific first
    generic_iso_pattern = r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):?(\d{2}):?(\d{2})?"
    match = re.search(generic_iso_pattern, filename)
    if match:
        year, month, day, hour, minute = map(int, match.groups()[:5])
        second = int(match.group(6)) if match.group(6) else 0
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            pass

    # Pattern 2: YYYY-MM-DDThhmm (ISO format with time)
    iso_pattern = r"(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})"
    match = re.search(iso_pattern, filename)
    if match:
        year, month, day, hour, minute = map(int, match.groups())
        try:
            return datetime(year, month, day, hour, minute)
        except ValueError:
            pass

    # Pattern 3: YYYY-MM-DD-hhmm (date with dash-separated time)
    dash_time_pattern = r"(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})"
    match = re.search(dash_time_pattern, filename)
    if match:
        year, month, day, hour, minute = map(int, match.groups())
        try:
            return datetime(year, month, day, hour, minute)
        except ValueError:
            pass

    # Pattern 4: YYYY-MM-DD (date only) - least specific last
    date_pattern = r"(\d{4})-(\d{2})-(\d{2})"
    match = re.search(date_pattern, filename)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    # Pattern 5: YYYYMMDD format
    compact_date_pattern = r"(\d{4})(\d{2})(\d{2})"
    match = re.search(compact_date_pattern, filename)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    return None


class CSVDataLoader:
    """Service for reading and processing CSV files with channel data."""

    def __init__(self) -> None:
        """Initialize the CSV data loader service."""
        self.loaded_files: Dict[str, pd.DataFrame] = {}
        self.processed_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.file_timestamps: Dict[str, datetime] = {}

    def read_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Read a CSV file and return the data as a pandas DataFrame.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame containing the data or None if failed
        """
        try:
            # Try different separators
            separators = [",", ";", ":", "\t"]
            df = None

            for sep in separators:
                try:
                    df = pd.read_csv(file_path, sep=sep, header=0, index_col=0)
                    if len(df.columns) > 0:  # Valid data found
                        break
                except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
                    continue

            if df is None or len(df.columns) == 0:
                print(f"Could not read CSV file {file_path} with any separator")
                return None

            # Clean column names (remove whitespace)
            df.columns = df.columns.str.strip()

            # Convert index to numeric (bin values)
            df.index = pd.to_numeric(df.index, errors="coerce")

            # Drop any rows with NaN index (invalid bin values)
            df = df.dropna()

            # Convert data to numeric, replacing non-numeric values with 0
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            return df

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    def process_channel_data(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Process channel data by grouping paired columns.

        Each channel is split into 2 columns and needs to be summed up.
        Expects an even number of channel columns after the index column.

        Args:
            file_path: Path to the processed file

        Returns:
            Dictionary mapping channel names to data arrays
        """
        if file_path not in self.loaded_files:
            return None

        df = self.loaded_files[file_path]

        # Check if we have an even number of columns
        if len(df.columns) % 2 != 0:
            print(f"Warning: Expected even number of columns, got {len(df.columns)}")
            # Still process, but treat odd columns as single channels

        # Process channels by pairs
        processed_channels: Dict[str, np.ndarray] = {}

        # Process columns in pairs
        for i in range(0, len(df.columns), 2):
            if i + 1 < len(df.columns):
                # We have a pair of columns
                col1, col2 = df.columns[i], df.columns[i + 1]
                channel_name = f"Channel_{i//2 + 1}"

                # Sum the two columns
                channel_data = (df[col1] + df[col2]).values
                processed_channels[channel_name] = channel_data
            else:
                # Odd column, treat as single channel
                col = df.columns[i]
                channel_name = f"Channel_{i//2 + 1}"
                channel_data = df[col].values
                processed_channels[channel_name] = channel_data

        return {
            "channels": processed_channels,
            "bin_values": df.index.values,
            "num_bins": len(df.index),
            "num_channels": len(processed_channels),
        }

    def load_file(self, file_path: str) -> bool:
        """Load a file into memory.

        Args:
            file_path: Path to the file to load

        Returns:
            True if file was loaded successfully, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return False

        # Extract timestamp from filename
        timestamp = parse_timestamp_from_filename(file_path)
        if timestamp:
            self.file_timestamps[file_path] = timestamp

        # Read the CSV file
        df = self.read_csv_file(file_path)
        if df is None:
            return False

        # Store the raw data
        self.loaded_files[file_path] = df

        # Process the data
        processed = self.process_channel_data(file_path)
        if processed:
            self.processed_data[file_path] = processed

        return True

    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """Get information about a loaded file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information or None if not loaded
        """
        if file_path not in self.loaded_files:
            return None

        df = self.loaded_files[file_path]
        processed = self.processed_data.get(file_path, {})
        timestamp = self.file_timestamps.get(file_path)

        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "timestamp": timestamp.isoformat() if timestamp else None,
            "num_channels": processed.get("num_channels", 0),
            "num_bins": processed.get("num_bins", 0),
            "bin_range": {
                "min": float(df.index.min()) if len(df.index) > 0 else 0,
                "max": float(df.index.max()) if len(df.index) > 0 else 0,
            },
            "channels": list(processed.get("channels", {}).keys()),
        }

    def get_channel_data(
        self, file_path: str, channel_name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get data for a specific channel.

        Args:
            file_path: Path to the file
            channel_name: Name of the channel

        Returns:
            Tuple of (bin_values, channel_values) or None if not found
        """
        if file_path not in self.processed_data:
            return None

        processed = self.processed_data[file_path]
        channels = processed.get("channels", {})

        if channel_name not in channels:
            return None

        return processed["bin_values"], channels[channel_name]

    def get_all_channel_data(
        self, file_path: str
    ) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Get data for all channels in a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary mapping channel names to (bin_values, channel_values) tuples
        """
        if file_path not in self.processed_data:
            return None

        processed = self.processed_data[file_path]
        channels = processed.get("channels", {})

        result = {}
        for channel_name, channel_values in channels.items():
            result[channel_name] = (processed["bin_values"], channel_values)

        return result

    def get_timestamp(self, file_path: str) -> Optional[datetime]:
        """Get the timestamp for a loaded file.

        Args:
            file_path: Path to the file

        Returns:
            datetime object if timestamp was extracted, None otherwise
        """
        return self.file_timestamps.get(file_path)

    def scan_folder(self, folder_path: str) -> List[str]:
        """Scan a folder for supported CSV files.

        Args:
            folder_path: Path to the folder to scan

        Returns:
            List of file paths for supported files
        """
        supported_extensions = {".csv", ".txt", ".dat"}
        found_files = []

        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()

                    if file_ext in supported_extensions:
                        found_files.append(file_path)
        except Exception as e:
            print(f"Error scanning folder {folder_path}: {e}")

        return found_files

    def unload_file(self, file_path: str) -> bool:
        """Unload a file from memory.

        Args:
            file_path: Path to the file to unload

        Returns:
            True if file was unloaded successfully, False otherwise
        """
        if file_path in self.loaded_files:
            del self.loaded_files[file_path]

        if file_path in self.processed_data:
            del self.processed_data[file_path]

        if file_path in self.file_timestamps:
            del self.file_timestamps[file_path]

        return True

    def unload_all_files(self) -> None:
        """Unload all files from memory."""
        self.loaded_files.clear()
        self.processed_data.clear()
        self.file_timestamps.clear()

    def get_loaded_files(self) -> List[str]:
        """Get list of currently loaded file paths.

        Returns:
            List of file paths
        """
        return list(self.loaded_files.keys())

    def is_file_loaded(self, file_path: str) -> bool:
        """Check if a file is currently loaded.

        Args:
            file_path: Path to the file

        Returns:
            True if file is loaded, False otherwise
        """
        return file_path in self.loaded_files
