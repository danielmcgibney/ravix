from importlib import resources
import pandas as pd
import warnings
import os

def get_data(filename=None):
    """
    Load data from the package's data directory.
    """

    # -------------------------------
    # LIST FILES
    # -------------------------------
    if filename is None:
        try:
            data_dir = resources.files('ravix').joinpath('data')

            if data_dir.is_dir():
                # Use Traversable API
                files = [
                    f.name for f in data_dir.iterdir()
                    if f.is_file()
                    and not f.name.startswith('.')
                    and f.name != '__init__.py'
                    and not f.name.endswith('.pyc')
                ]

                files = sorted(files)

                if files:
                    print("Available data files in ravix:")
                    for f in files:
                        print(f"  - {f}")
                else:
                    print("No data files found in ravix package.")

                return None
            else:
                print("No data directory found in ravix package.")
                return []

        except Exception as e:
            warnings.warn(f"Could not list package data files: {e}", UserWarning)
            return []

    # -------------------------------
    # LOAD FROM PACKAGE
    # -------------------------------
    try:
        csv_path = resources.files('ravix').joinpath('data', filename)

        if csv_path.is_file():
            with csv_path.open('rb') as f:
                return pd.read_csv(f)

    except Exception:
        pass

    # -------------------------------
    # FALLBACK TO FILESYSTEM
    # -------------------------------
    if os.path.exists(filename):
        warnings.warn(
            f"Loading '{filename}' from filesystem. "
            f"get_data() is intended for package data files. "
            f"For external files, please use pd.read_csv() directly.",
            UserWarning,
            stacklevel=2
        )
        return pd.read_csv(filename)

    # -------------------------------
    # FILE NOT FOUND
    # -------------------------------
    available_files = get_data()  # List files for message

    if available_files:
        raise FileNotFoundError(
            f"'{filename}' not found in package data directory or filesystem.\n"
            f"See above for available package data files."
        )
    else:
        raise FileNotFoundError(
            f"'{filename}' not found in package data directory or filesystem. "
            f"Use get_data() for package data files or pd.read_csv() for external files."
        )
