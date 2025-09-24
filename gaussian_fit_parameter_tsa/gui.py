"""GUI application for Gaussian Fit Parameter Time Series Analyzer."""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from gaussian_fit_parameter_tsa.csv_loader import (
    CSVDataLoader,
    parse_timestamp_from_filename,
)
from gaussian_fit_parameter_tsa.model import (
    FitParams,
    fit_gaussian_robust,
    fwhm_from_sigma,
    gaussian_bg,
)


class GaussianFitApp:
    """Main application class for Gaussian Fit Parameter Time Series Analyzer."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the application."""
        self.root = root
        self.root.title("Gaussian Fit Parameter Time Series Analyzer")
        self.root.geometry("1200x800")

        # Data storage
        self.csv_loader = CSVDataLoader()
        self.loaded_files: List[str] = []
        self.current_file: Optional[str] = None
        self.current_channel: Optional[str] = None
        self.fit_results: Dict[
            str, Dict[str, FitParams]
        ] = {}  # file -> channel -> FitParams

        # GUI setup
        self.setup_gui()

        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()

    def setup_gui(self) -> None:
        """Set up the GUI components."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Button(file_frame, text="Load Files", command=self.load_files).grid(
            row=0, column=0, padx=(0, 5)
        )
        ttk.Button(file_frame, text="Load Folder", command=self.load_folder).grid(
            row=0, column=1, padx=(0, 5)
        )

        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(
            file_frame, textvariable=self.file_var, state="readonly", width=50
        )
        self.file_combo.grid(row=0, column=2, padx=(5, 0), sticky="ew")
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_selected)

        file_frame.columnconfigure(2, weight=1)

        # Channel selection frame
        channel_frame = ttk.LabelFrame(
            main_frame, text="Channel Selection", padding="5"
        )
        channel_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(channel_frame, text="Channel:").grid(row=0, column=0, padx=(0, 5))

        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(
            channel_frame, textvariable=self.channel_var, state="readonly", width=20
        )
        self.channel_combo.grid(row=0, column=1, padx=(0, 10), sticky="ew")
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_selected)

        ttk.Button(channel_frame, text="Fit Gaussian", command=self.fit_gaussian).grid(
            row=0, column=2, padx=(0, 5)
        )
        ttk.Button(
            channel_frame, text="Show Time Series", command=self.show_time_series
        ).grid(row=0, column=3)

        # Add zoom controls
        ttk.Button(channel_frame, text="Reset Zoom", command=self.reset_zoom).grid(
            row=0, column=4, padx=(5, 0)
        )
        ttk.Button(channel_frame, text="Zoom to Fit", command=self.zoom_to_fit).grid(
            row=0, column=5, padx=(5, 0)
        )
        ttk.Button(channel_frame, text="Help", command=self.show_help).grid(
            row=0, column=6, padx=(5, 0)
        )

        channel_frame.columnconfigure(1, weight=1)

        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Raw data tab
        self.raw_data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.raw_data_frame, text="Raw Data")

        # Fitted data tab
        self.fitted_data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.fitted_data_frame, text="Fitted Data")

        # Time series tab
        self.time_series_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.time_series_frame, text="Time Series")

        # Parameters tab
        self.parameters_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.parameters_frame, text="Parameters")

        # Setup matplotlib figures
        self.setup_matplotlib()

    def setup_matplotlib(self) -> None:
        """Set up matplotlib figures for each tab."""
        # Raw data figure
        self.raw_fig = Figure(figsize=(8, 6), dpi=100)
        self.raw_ax = self.raw_fig.add_subplot(111)
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, self.raw_data_frame)
        self.raw_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar for raw data
        self.raw_toolbar = NavigationToolbar2Tk(self.raw_canvas, self.raw_data_frame)
        self.raw_toolbar.update()

        # Fitted data figure
        self.fitted_fig = Figure(figsize=(8, 6), dpi=100)
        self.fitted_ax = self.fitted_fig.add_subplot(111)
        self.fitted_canvas = FigureCanvasTkAgg(self.fitted_fig, self.fitted_data_frame)
        self.fitted_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar for fitted data
        self.fitted_toolbar = NavigationToolbar2Tk(
            self.fitted_canvas, self.fitted_data_frame
        )
        self.fitted_toolbar.update()

        # Time series figure
        self.ts_fig = Figure(figsize=(8, 6), dpi=100)
        self.ts_ax = self.ts_fig.add_subplot(111)
        self.ts_canvas = FigureCanvasTkAgg(self.ts_fig, self.time_series_frame)
        self.ts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar for time series
        self.ts_toolbar = NavigationToolbar2Tk(self.ts_canvas, self.time_series_frame)
        self.ts_toolbar.update()

        # Parameters table
        self.setup_parameters_table()

    def setup_parameters_table(self) -> None:
        """Set up the parameters table."""
        # Create treeview for parameters
        columns = ("Parameter", "Value", "Uncertainty", "Unit")
        self.params_tree = ttk.Treeview(
            self.parameters_frame, columns=columns, show="headings", height=10
        )

        for col in columns:
            self.params_tree.heading(col, text=col)
            self.params_tree.column(col, width=120)

        # Add scrollbar
        params_scrollbar = ttk.Scrollbar(
            self.parameters_frame, orient=tk.VERTICAL, command=self.params_tree.yview
        )
        self.params_tree.configure(yscrollcommand=params_scrollbar.set)

        self.params_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_keyboard_shortcuts(self) -> None:
        """Set up keyboard shortcuts for the application."""
        # Bind keyboard shortcuts
        self.root.bind("<Control-r>", lambda e: self.reset_zoom())
        self.root.bind("<Control-f>", lambda e: self.zoom_to_fit())
        self.root.bind("<Control-g>", lambda e: self.fit_gaussian())
        self.root.bind("<Control-t>", lambda e: self.show_time_series())
        self.root.bind("<Control-o>", lambda e: self.load_files())
        self.root.bind("<Control-Shift-O>", lambda e: self.load_folder())

        # Make sure the root window can receive focus for keyboard shortcuts
        self.root.focus_set()

    def load_files(self) -> None:
        """Load CSV files."""
        file_paths = filedialog.askopenfilenames(
            title="Select CSV files",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("Data files", "*.dat"),
                ("All files", "*.*"),
            ],
        )

        if file_paths:
            self.load_files_from_paths(list(file_paths))

    def load_folder(self) -> None:
        """Load all CSV files from a folder."""
        folder_path = filedialog.askdirectory(
            title="Select folder containing CSV files"
        )

        if folder_path:
            file_paths = self.csv_loader.scan_folder(folder_path)
            if file_paths:
                self.load_files_from_paths(file_paths)
            else:
                try:
                    messagebox.showwarning(
                        "No Files", "No supported files found in the selected folder."
                    )
                except tk.TclError:
                    print("Warning: No supported files found in the selected folder.")

    def load_files_from_paths(self, file_paths: List[str]) -> None:
        """Load files from the given paths."""
        loaded_count = 0

        for file_path in file_paths:
            if self.csv_loader.load_file(file_path):
                self.loaded_files.append(file_path)
                loaded_count += 1
            else:
                try:
                    messagebox.showerror(
                        "Error", f"Failed to load file: {os.path.basename(file_path)}"
                    )
                except tk.TclError:
                    print(f"Error: Failed to load file: {os.path.basename(file_path)}")

        if loaded_count > 0:
            self.update_file_combo()
            try:
                messagebox.showinfo(
                    "Success", f"Loaded {loaded_count} file(s) successfully."
                )
            except tk.TclError:
                print(f"Success: Loaded {loaded_count} file(s) successfully.")
        else:
            try:
                messagebox.showerror("Error", "No files were loaded successfully.")
            except tk.TclError:
                print("Error: No files were loaded successfully.")

    def update_file_combo(self) -> None:
        """Update the file combo box with loaded files."""
        file_names = [os.path.basename(f) for f in self.loaded_files]
        self.file_combo["values"] = file_names

        if file_names and not self.current_file:
            self.file_combo.current(0)
            self.on_file_selected(None)

    def on_file_selected(self, event: Optional[tk.Event]) -> None:
        """Handle file selection."""
        if not self.file_combo.get():
            return

        selected_name = self.file_combo.get()
        self.current_file = next(
            f for f in self.loaded_files if os.path.basename(f) == selected_name
        )

        # Update channel combo
        self.update_channel_combo()

        # Plot raw data
        self.plot_raw_data()

    def update_channel_combo(self) -> None:
        """Update the channel combo box."""
        if not self.current_file:
            return

        file_info = self.csv_loader.get_file_info(self.current_file)
        if file_info and "channels" in file_info:
            channels = file_info["channels"]
            self.channel_combo["values"] = channels

            if channels and not self.current_channel:
                self.channel_combo.current(0)
                self.on_channel_selected(None)

    def on_channel_selected(self, event: Optional[tk.Event]) -> None:
        """Handle channel selection."""
        self.current_channel = self.channel_combo.get()
        if self.current_channel:
            self.plot_raw_data()

    def plot_raw_data(self) -> None:
        """Plot raw data for the selected file and channel."""
        if not self.current_file or not self.current_channel:
            return

        # Get channel data
        channel_data = self.csv_loader.get_channel_data(
            self.current_file, self.current_channel
        )
        if not channel_data:
            return

        bin_values, histogram_values = channel_data

        # Clear and plot
        self.raw_ax.clear()
        self.raw_ax.scatter(bin_values, histogram_values, alpha=0.6, s=20)
        self.raw_ax.set_xlabel("Bin Value")
        self.raw_ax.set_ylabel("Count")
        self.raw_ax.set_title(
            f"Raw Data - {os.path.basename(self.current_file)} - {self.current_channel}"
        )
        self.raw_ax.grid(True, alpha=0.3)

        self.raw_canvas.draw()

    def fit_gaussian(self) -> None:
        """Fit Gaussian to the selected channel data."""
        if not self.current_file or not self.current_channel:
            try:
                messagebox.showwarning(
                    "No Selection", "Please select a file and channel first."
                )
            except tk.TclError:
                # Handle case where tkinter is not properly initialized
                print("Warning: Please select a file and channel first.")
            return

        # Get channel data
        channel_data = self.csv_loader.get_channel_data(
            self.current_file, self.current_channel
        )
        if not channel_data:
            try:
                messagebox.showerror(
                    "Error", "No data available for the selected channel."
                )
            except tk.TclError:
                print("Error: No data available for the selected channel.")
            return

        bin_values, histogram_values = channel_data

        try:
            # Use the robust fitting function
            fit_result = fit_gaussian_robust(bin_values, histogram_values)

            if not fit_result.success:
                try:
                    messagebox.showerror(
                        "Fit Error", "Gaussian fit failed. Check your data quality."
                    )
                except tk.TclError:
                    print("Fit Error: Gaussian fit failed. Check your data quality.")
                return

            # Store fit results
            if self.current_file not in self.fit_results:
                self.fit_results[self.current_file] = {}

            self.fit_results[self.current_file][self.current_channel] = fit_result

            # Plot fitted data
            popt = np.array(
                [fit_result.A, fit_result.mu, fit_result.sigma, fit_result.B]
            )
            self.plot_fitted_data(bin_values, histogram_values, popt)

            # Update parameters table
            self.update_parameters_table()

            # Show success message with fit quality
            success_msg = (
                f"Gaussian fit completed successfully!\n"
                f"R² = {fit_result.r_squared:.4f}\n"
                f"RMSE = {fit_result.rmse:.2f}"
            )
            try:
                messagebox.showinfo("Success", success_msg)
            except tk.TclError:
                print(f"Success: {success_msg}")

        except Exception as e:
            try:
                messagebox.showerror("Fit Error", f"Failed to fit Gaussian: {str(e)}")
            except tk.TclError:
                print(f"Fit Error: Failed to fit Gaussian: {str(e)}")

    def plot_fitted_data(
        self, bin_values: np.ndarray, histogram_values: np.ndarray, popt: np.ndarray
    ) -> None:
        """Plot the fitted Gaussian data."""
        A, mu, sigma, B = popt

        # Clear and plot
        self.fitted_ax.clear()

        # Plot raw data
        self.fitted_ax.scatter(
            bin_values, histogram_values, alpha=0.6, s=20, label="Data", color="blue"
        )

        # Plot fitted curve
        x_fit = np.linspace(np.min(bin_values), np.max(bin_values), 1000)
        y_fit = gaussian_bg(x_fit, A, mu, sigma, B)
        self.fitted_ax.plot(x_fit, y_fit, "r-", linewidth=2, label="Gaussian Fit")

        # Add vertical lines for center and FWHM
        fwhm = fwhm_from_sigma(sigma)
        self.fitted_ax.axvline(
            mu, color="green", linestyle="--", alpha=0.7, label=f"Center (μ={mu:.3f})"
        )
        self.fitted_ax.axvline(mu - fwhm / 2, color="orange", linestyle=":", alpha=0.7)
        self.fitted_ax.axvline(
            mu + fwhm / 2,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label=f"FWHM={fwhm:.3f}",
        )

        self.fitted_ax.set_xlabel("Bin Value")
        self.fitted_ax.set_ylabel("Count")
        file_name = (
            os.path.basename(self.current_file) if self.current_file else "Unknown"
        )
        self.fitted_ax.set_title(f"Gaussian Fit - {file_name} - {self.current_channel}")
        self.fitted_ax.legend()
        self.fitted_ax.grid(True, alpha=0.3)

        self.fitted_canvas.draw()

    def update_parameters_table(self) -> None:
        """Update the parameters table with current fit results."""
        # Clear existing items
        for item in self.params_tree.get_children():
            self.params_tree.delete(item)

        if not self.current_file or not self.current_channel:
            return

        if (
            self.current_file in self.fit_results
            and self.current_channel in self.fit_results[self.current_file]
        ):
            params = self.fit_results[self.current_file][self.current_channel]

            # Add parameters with uncertainties
            parameters = [
                ("Amplitude (A)", f"{params.A:.2f}", f"±{params.A_err:.2f}", ""),
                ("Center (μ)", f"{params.mu:.4f}", f"±{params.mu_err:.4f}", ""),
                (
                    "Std. deviation (σ)",
                    f"{params.sigma:.4f}",
                    f"±{params.sigma_err:.4f}",
                    "",
                ),
                ("Background (B)", f"{params.B:.2f}", f"±{params.B_err:.2f}", ""),
                ("FWHM", f"{params.fwhm:.4f}", "", ""),
                ("Peak area", f"{params.area:.2f}", "", ""),
                ("R²", f"{params.r_squared:.4f}", "", ""),
                ("RMSE", f"{params.rmse:.2f}", "", ""),
            ]

            for param, value, uncertainty, unit in parameters:
                self.params_tree.insert(
                    "", "end", values=(param, value, uncertainty, unit)
                )

    def show_time_series(self) -> None:
        """Show time series plot for the selected channel across all files."""
        if not self.current_channel:
            try:
                messagebox.showwarning("No Selection", "Please select a channel first.")
            except tk.TclError:
                print("Warning: Please select a channel first.")
            return

        # Collect data for time series
        times = []
        parameters: Dict[str, List[float]] = {
            "A": [],
            "mu": [],
            "sigma": [],
            "B": [],
            "fwhm": [],
            "area": [],
        }

        for file_path in self.loaded_files:
            if (
                file_path in self.fit_results
                and self.current_channel in self.fit_results[file_path]
            ):
                # Get timestamp
                timestamp = parse_timestamp_from_filename(file_path)
                if timestamp:
                    times.append(timestamp)

                    # Get parameters
                    params = self.fit_results[file_path][self.current_channel]
                    parameters["A"].append(params.A)
                    parameters["mu"].append(params.mu)
                    parameters["sigma"].append(params.sigma)
                    parameters["B"].append(params.B)
                    parameters["fwhm"].append(params.fwhm)
                    parameters["area"].append(params.area)

        if not times:
            try:
                messagebox.showwarning(
                    "No Data", "No fitted data available for time series analysis."
                )
            except tk.TclError:
                print("Warning: No fitted data available for time series analysis.")
            return

        # Sort by time
        sorted_indices = np.argsort(times)
        times = [times[i] for i in sorted_indices]
        for key in parameters:
            parameters[key] = [parameters[key][i] for i in sorted_indices]

        # Plot time series
        self.ts_ax.clear()

        # Create subplots for different parameters
        self.ts_fig.clear()

        # Create 2x3 subplot layout
        axes = []
        for i in range(6):
            ax = self.ts_fig.add_subplot(2, 3, i + 1)
            axes.append(ax)

        self.ts_fig.suptitle(f"Time Series Analysis - {self.current_channel}")

        param_names = ["A", "mu", "sigma", "B", "fwhm", "area"]
        param_labels = [
            "Amplitude",
            "Center (μ)",
            "Std. Dev. (σ)",
            "Background",
            "FWHM",
            "Area",
        ]

        for i, (param, label) in enumerate(zip(param_names, param_labels)):
            axes[i].plot(times, parameters[param], "o-", markersize=4)
            axes[i].set_title(label)
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Value")
            axes[i].grid(True, alpha=0.3)

            # Rotate x-axis labels
            axes[i].tick_params(axis="x", rotation=45)

        self.ts_fig.tight_layout()

        # Update the time series canvas
        self.ts_canvas.draw()

    def reset_zoom(self) -> None:
        """Reset zoom on all plots."""
        # Reset raw data plot
        if hasattr(self, "raw_ax"):
            self.raw_ax.relim()
            self.raw_ax.autoscale()
            self.raw_canvas.draw()

        # Reset fitted data plot
        if hasattr(self, "fitted_ax"):
            self.fitted_ax.relim()
            self.fitted_ax.autoscale()
            self.fitted_canvas.draw()

        # Reset time series plot
        if hasattr(self, "ts_fig"):
            for ax in self.ts_fig.get_axes():
                ax.relim()
                ax.autoscale()
            self.ts_canvas.draw()

    def zoom_to_fit(self) -> None:
        """Zoom to fit the data in all plots."""
        # Get current tab
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:  # Raw data tab
            if hasattr(self, "raw_ax") and self.raw_ax.has_data():
                self.raw_ax.relim()
                self.raw_ax.autoscale()
                self.raw_canvas.draw()
        elif current_tab == 1:  # Fitted data tab
            if hasattr(self, "fitted_ax") and self.fitted_ax.has_data():
                self.fitted_ax.relim()
                self.fitted_ax.autoscale()
                self.fitted_canvas.draw()
        elif current_tab == 2:  # Time series tab
            if hasattr(self, "ts_fig"):
                for ax in self.ts_fig.get_axes():
                    if ax.has_data():
                        ax.relim()
                        ax.autoscale()
                self.ts_canvas.draw()

    def show_help(self) -> None:
        """Show help dialog with keyboard shortcuts and usage instructions."""
        help_text = """Gaussian Fit Parameter Time Series Analyzer - Help

KEYBOARD SHORTCUTS:
• Ctrl+O: Load files
• Ctrl+Shift+O: Load folder
• Ctrl+G: Fit Gaussian
• Ctrl+T: Show Time Series
• Ctrl+R: Reset zoom on all plots
• Ctrl+F: Zoom to fit current plot

USAGE:
1. Load CSV files using 'Load Files' or 'Load Folder' buttons
2. Select a file from the dropdown
3. Select a channel from the channel dropdown
4. Click 'Fit Gaussian' to perform the fit
5. Use 'Show Time Series' to see parameter evolution over time
6. Use the navigation toolbars to zoom and pan on plots
7. Use 'Reset Zoom' to return to original view
8. Use 'Zoom to Fit' to fit data in current plot

PLOT FEATURES:
• All plots have built-in zoom and pan functionality
• Use mouse wheel to zoom in/out
• Click and drag to pan
• Use navigation toolbar buttons for more control
• Right-click for context menu options

For more information, see the documentation."""

        try:
            messagebox.showinfo("Help", help_text)
        except tk.TclError:
            print(help_text)


def main() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    GaussianFitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
