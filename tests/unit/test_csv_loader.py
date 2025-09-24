"""Tests for the CSV loader module."""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from gaussian_fit_parameter_tsa.csv_loader import (
    CSVDataLoader,
    parse_timestamp_from_filename,
)


class TestTimestampParsing:
    """Test timestamp parsing functionality."""

    def test_parse_date_only(self) -> None:
        """Test parsing date-only format."""
        filename = "2022-09-19-data.csv"
        result = parse_timestamp_from_filename(filename)
        assert result == datetime(2022, 9, 19)

    def test_parse_iso_with_time(self) -> None:
        """Test parsing ISO format with time."""
        filename = "2022-09-19T1430-data.csv"
        result = parse_timestamp_from_filename(filename)
        assert result == datetime(2022, 9, 19, 14, 30)

    def test_parse_dash_time(self) -> None:
        """Test parsing dash-separated time format."""
        filename = "2022-09-19-1430-data.csv"
        result = parse_timestamp_from_filename(filename)
        assert result == datetime(2022, 9, 19, 14, 30)

    def test_parse_generic_iso(self) -> None:
        """Test parsing generic ISO format."""
        filename = "2022-09-19T14:30:45-data.csv"
        result = parse_timestamp_from_filename(filename)
        assert result == datetime(2022, 9, 19, 14, 30, 45)

    def test_parse_compact_date(self) -> None:
        """Test parsing compact date format."""
        filename = "20220919-data.csv"
        result = parse_timestamp_from_filename(filename)
        assert result == datetime(2022, 9, 19)

    def test_parse_no_timestamp(self) -> None:
        """Test parsing file with no timestamp."""
        filename = "data-file.csv"
        result = parse_timestamp_from_filename(filename)
        assert result is None

    def test_parse_invalid_date(self) -> None:
        """Test parsing invalid date."""
        filename = "2022-13-45-data.csv"  # Invalid month and day
        result = parse_timestamp_from_filename(filename)
        assert result is None

    def test_parse_complex_filename(self) -> None:
        """Test parsing complex filename with timestamp."""
        filename = "2022-09-19-0T-time-PMA0.csv"
        result = parse_timestamp_from_filename(filename)
        assert result == datetime(2022, 9, 19)


class TestCSVDataLoader:
    """Test CSV data loader functionality."""

    @pytest.fixture  # type: ignore[misc]
    def loader(self) -> Generator[CSVDataLoader, None, None]:
        """Create a CSV data loader instance."""
        yield CSVDataLoader()

    @pytest.fixture  # type: ignore[misc]
    def sample_csv_data(self) -> Generator[pd.DataFrame, None, None]:
        """Create sample CSV data for testing."""
        data = {
            "Ch01": [1, 2, 3, 4, 5],
            "Ch02": [2, 3, 4, 5, 6],
            "Ch03": [3, 4, 5, 6, 7],
            "Ch04": [4, 5, 6, 7, 8],
        }
        df = pd.DataFrame(data, index=[0.1, 0.2, 0.3, 0.4, 0.5])
        yield df

    def test_loader_initialization(self, loader: CSVDataLoader) -> None:
        """Test loader initialization."""
        assert len(loader.loaded_files) == 0
        assert len(loader.processed_data) == 0
        assert len(loader.file_timestamps) == 0

    def test_load_nonexistent_file(self, loader: CSVDataLoader) -> None:
        """Test loading a non-existent file."""
        result = loader.load_file("nonexistent.csv")
        assert result is False

    def test_load_csv_file(
        self, loader: CSVDataLoader, sample_csv_data: pd.DataFrame
    ) -> None:
        """Test loading a CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_csv_data.to_csv(f.name, sep=":")
            temp_path = f.name

        try:
            result = loader.load_file(temp_path)
            assert result is True
            assert temp_path in loader.loaded_files
            assert temp_path in loader.processed_data
        finally:
            os.unlink(temp_path)

    def test_process_channel_data(
        self, loader: CSVDataLoader, sample_csv_data: pd.DataFrame
    ) -> None:
        """Test processing channel data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_csv_data.to_csv(f.name, sep=":")
            temp_path = f.name

        try:
            loader.load_file(temp_path)
            processed = loader.process_channel_data(temp_path)

            assert processed is not None
            assert "channels" in processed
            assert "bin_values" in processed
            assert "num_bins" in processed
            assert "num_channels" in processed

            # Check that channels are processed directly
            channels = processed["channels"]
            assert len(channels) == 4  # 4 separate channels

            # Check that each channel is processed directly
            assert "Ch01" in channels
            assert "Ch02" in channels
            assert "Ch03" in channels
            assert "Ch04" in channels

            # Check that data is not summed
            ch01_data = channels["Ch01"]
            np.testing.assert_array_equal(ch01_data, sample_csv_data["Ch01"].values)

            ch02_data = channels["Ch02"]
            np.testing.assert_array_equal(ch02_data, sample_csv_data["Ch02"].values)

        finally:
            os.unlink(temp_path)

    def test_get_file_info(
        self, loader: CSVDataLoader, sample_csv_data: pd.DataFrame
    ) -> None:
        """Test getting file information."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="2022-09-19-data.csv", delete=False
        ) as f:
            sample_csv_data.to_csv(f.name, sep=":")
            temp_path = f.name

        try:
            loader.load_file(temp_path)
            info = loader.get_file_info(temp_path)

            assert info is not None
            assert info["file_path"] == temp_path
            assert info["file_name"] == os.path.basename(temp_path)
            assert info["timestamp"] == "2022-09-19T00:00:00"
            assert info["num_channels"] == 4
            assert info["num_bins"] == 5
            assert "bin_range" in info
            assert "channels" in info
            assert len(info["channels"]) == 4

        finally:
            os.unlink(temp_path)

    def test_get_channel_data(
        self, loader: CSVDataLoader, sample_csv_data: pd.DataFrame
    ) -> None:
        """Test getting specific channel data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_csv_data.to_csv(f.name, sep=":")
            temp_path = f.name

        try:
            loader.load_file(temp_path)
            bin_values, channel_data = loader.get_channel_data(temp_path, "Ch01")

            assert bin_values is not None
            assert channel_data is not None
            assert len(bin_values) == 5
            assert len(channel_data) == 5

            # Check that the data is correct (direct channel data)
            expected = sample_csv_data["Ch01"]
            np.testing.assert_array_equal(channel_data, expected.values)

        finally:
            os.unlink(temp_path)

    def test_get_all_channel_data(
        self, loader: CSVDataLoader, sample_csv_data: pd.DataFrame
    ) -> None:
        """Test getting all channel data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_csv_data.to_csv(f.name, sep=":")
            temp_path = f.name

        try:
            loader.load_file(temp_path)
            all_data = loader.get_all_channel_data(temp_path)

            assert all_data is not None
            assert len(all_data) == 4
            assert "Ch01" in all_data
            assert "Ch02" in all_data
            assert "Ch03" in all_data
            assert "Ch04" in all_data

            # Check Ch01 data
            bin_vals, ch01_data = all_data["Ch01"]
            expected_ch01 = sample_csv_data["Ch01"]
            np.testing.assert_array_equal(ch01_data, expected_ch01.values)

        finally:
            os.unlink(temp_path)

    def test_get_timestamp(
        self, loader: CSVDataLoader, sample_csv_data: pd.DataFrame
    ) -> None:
        """Test getting file timestamp."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="2022-09-19T1430-data.csv", delete=False
        ) as f:
            sample_csv_data.to_csv(f.name, sep=":")
            temp_path = f.name

        try:
            loader.load_file(temp_path)
            timestamp = loader.get_timestamp(temp_path)

            assert timestamp is not None
            assert timestamp == datetime(2022, 9, 19, 14, 30)

        finally:
            os.unlink(temp_path)

    def test_scan_folder(self, loader: CSVDataLoader) -> None:
        """Test scanning folder for CSV files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            csv_file1 = os.path.join(temp_dir, "data1.csv")
            csv_file2 = os.path.join(temp_dir, "data2.txt")  # .txt is supported
            txt_file = os.path.join(temp_dir, "readme.txt")  # .txt is supported
            dat_file = os.path.join(temp_dir, "data.dat")  # .dat is supported
            other_file = os.path.join(temp_dir, "data.xyz")  # not supported

            # Create empty files
            Path(csv_file1).touch()
            Path(csv_file2).touch()
            Path(txt_file).touch()
            Path(dat_file).touch()
            Path(other_file).touch()

            found_files = loader.scan_folder(temp_dir)

            assert len(found_files) == 4  # csv, txt, txt, dat
            assert csv_file1 in found_files
            assert csv_file2 in found_files
            assert txt_file in found_files
            assert dat_file in found_files
            assert other_file not in found_files

    def test_unload_file(
        self, loader: CSVDataLoader, sample_csv_data: pd.DataFrame
    ) -> None:
        """Test unloading a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_csv_data.to_csv(f.name, sep=":")
            temp_path = f.name

        try:
            loader.load_file(temp_path)
            assert loader.is_file_loaded(temp_path)

            result = loader.unload_file(temp_path)
            assert result is True
            assert not loader.is_file_loaded(temp_path)

        finally:
            os.unlink(temp_path)

    def test_unload_all_files(
        self, loader: CSVDataLoader, sample_csv_data: pd.DataFrame
    ) -> None:
        """Test unloading all files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_csv_data.to_csv(f.name, sep=":")
            temp_path = f.name

        try:
            loader.load_file(temp_path)
            assert len(loader.get_loaded_files()) == 1

            loader.unload_all_files()
            assert len(loader.get_loaded_files()) == 0

        finally:
            os.unlink(temp_path)

    def test_direct_channel_processing(self, loader: CSVDataLoader) -> None:
        """Test processing channels directly without summing."""
        # Create data where each column is a channel
        data = {
            "Ch01": [1, 2, 3],
            "Ch02": [2, 3, 4],
            "Ch03": [3, 4, 5],
        }
        df = pd.DataFrame(data, index=[0.1, 0.2, 0.3])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, sep=":")
            temp_path = f.name

        try:
            result = loader.load_file(temp_path)
            assert result is True

            processed = loader.process_channel_data(temp_path)
            assert processed is not None
            assert len(processed["channels"]) == 3  # Three separate channels

            # Check that channels are processed directly
            assert "Ch01" in processed["channels"]
            assert "Ch02" in processed["channels"]
            assert "Ch03" in processed["channels"]

            # Check that data is not summed
            ch01_data = processed["channels"]["Ch01"]
            np.testing.assert_array_equal(ch01_data, [1, 2, 3])

        finally:
            os.unlink(temp_path)
