#!/usr/bin/env python3
"""
NASA Turbofan Engine Degradation Dataset Downloader.

This script downloads the C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
dataset from Kaggle (primary) or NASA's Prognostic Data Repository (backup).
The dataset contains run-to-failure simulated data for turbofan engine degradation.

Dataset Information:
    - Source: NASA Prognostic Center of Excellence (via Kaggle)
    - Primary URL: https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation
    - Backup URL: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    - Contains: FD001, FD002, FD003, FD004 datasets
    - Each dataset has train, test, and RUL files

Usage:
    python data/download_data.py
    python data/download_data.py --dataset FD001
    python data/download_data.py --output-dir /path/to/data

Example:
    >>> from data.download_data import TurbofanDataDownloader
    >>> downloader = TurbofanDataDownloader()
    >>> downloader.download_and_extract()
"""

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Custom exception for download-related errors."""

    pass


class ExtractionError(Exception):
    """Custom exception for extraction-related errors."""

    pass


class ProgressReporter:
    """Report download progress to console."""

    def __init__(self, filename: str) -> None:
        """
        Initialize progress reporter.

        Args:
            filename: Name of file being downloaded
        """
        self.filename = filename
        self.last_reported = 0

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        """
        Report progress callback for urlretrieve.

        Args:
            block_num: Current block number
            block_size: Size of each block in bytes
            total_size: Total file size in bytes (-1 if unknown)
        """
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)

            # Report every 10%
            if int(percent / 10) > self.last_reported:
                self.last_reported = int(percent / 10)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                logger.info(
                    f"Downloading {self.filename}: {percent:.1f}% "
                    f"({mb_downloaded:.1f}/{mb_total:.1f} MB)"
                )


class TurbofanDataDownloader:
    """
    Download and extract NASA Turbofan Engine Degradation dataset.

    This class handles downloading the C-MAPSS dataset from NASA's data repository,
    extracting the ZIP file, and organizing the data files.

    Attributes:
        output_dir: Directory where data will be saved
        dataset_url: URL to download the dataset from

    Example:
        >>> downloader = TurbofanDataDownloader(output_dir=Path("./data/raw"))
        >>> downloader.download_and_extract()
        >>> print(downloader.get_available_files())
    """

    # Primary dataset URL (Kaggle - more reliable)
    DATASET_URL = (
        "https://www.kaggle.com/api/v1/datasets/download/bishals098/nasa-turbofan-engine-degradation-simulation"
    )

    # Alternative mirror/backup URLs (NASA sources)
    BACKUP_URLS = [
        "https://ti.arc.nasa.gov/c/6/",
        "https://data.nasa.gov/download/cqx2-p9e6/application%2Fzip",
    ]

    # Expected files after extraction
    EXPECTED_FILES = [
        "train_FD001.txt",
        "train_FD002.txt",
        "train_FD003.txt",
        "train_FD004.txt",
        "test_FD001.txt",
        "test_FD002.txt",
        "test_FD003.txt",
        "test_FD004.txt",
        "RUL_FD001.txt",
        "RUL_FD002.txt",
        "RUL_FD003.txt",
        "RUL_FD004.txt",
    ]

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        dataset_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the downloader.

        Args:
            output_dir: Directory to save downloaded data. Defaults to data/raw
                       relative to project root.
            dataset_url: Custom URL to download from. Defaults to NASA repository.
        """
        if output_dir is None:
            # Default to project's data/raw directory
            project_root = Path(__file__).parent.parent
            self.output_dir = project_root / "data" / "raw"
        else:
            self.output_dir = Path(output_dir)

        self.dataset_url = dataset_url or self.DATASET_URL
        self.zip_path = self.output_dir / "CMAPSSData.zip"

    def download_and_extract(self, force: bool = False) -> Path:
        """
        Download and extract the dataset.

        Args:
            force: If True, re-download even if files exist

        Returns:
            Path to the directory containing extracted files

        Raises:
            DownloadError: If download fails
            ExtractionError: If extraction fails
        """
        # Check if already downloaded
        if self._check_existing_files() and not force:
            logger.info("Dataset already exists. Use force=True to re-download.")
            return self.output_dir

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Download
        self._download_file()

        # Extract
        self._extract_zip()

        # Verify
        self._verify_extraction()

        # Cleanup
        self._cleanup()

        logger.info("Dataset download and extraction complete!")
        return self.output_dir

    def _check_existing_files(self) -> bool:
        """
        Check if dataset files already exist.

        Returns:
            True if all expected files exist, False otherwise
        """
        if not self.output_dir.exists():
            return False

        existing_files = set(f.name for f in self.output_dir.iterdir())
        expected_files = set(self.EXPECTED_FILES)

        if expected_files.issubset(existing_files):
            logger.info("All dataset files already present")
            return True

        missing = expected_files - existing_files
        logger.info(f"Missing files: {missing}")
        return False

    def _download_file(self) -> None:
        """
        Download the dataset ZIP file.

        Raises:
            DownloadError: If download fails from all URLs
        """
        urls_to_try = [self.dataset_url] + self.BACKUP_URLS

        for url in urls_to_try:
            try:
                logger.info(f"Attempting download from: {url}")
                progress = ProgressReporter("CMAPSSData.zip")
                urlretrieve(url, self.zip_path, reporthook=progress)
                logger.info(f"Successfully downloaded to: {self.zip_path}")
                return

            except HTTPError as e:
                logger.warning(f"HTTP Error {e.code}: {e.reason}")
            except URLError as e:
                logger.warning(f"URL Error: {e.reason}")
            except Exception as e:
                logger.warning(f"Download error: {e}")

        raise DownloadError(
            "Failed to download dataset from all available URLs. "
            "Please download manually from: "
            "https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation "
            "or https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/"
        )

    def _extract_zip(self) -> None:
        """
        Extract the downloaded ZIP file.

        Raises:
            ExtractionError: If extraction fails
        """
        if not self.zip_path.exists():
            raise ExtractionError(f"ZIP file not found: {self.zip_path}")

        try:
            logger.info(f"Extracting {self.zip_path}...")

            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                # List contents
                file_list = zip_ref.namelist()
                logger.info(f"Archive contains {len(file_list)} files")

                # Extract all files
                for file_info in zip_ref.infolist():
                    # Skip directories
                    if file_info.is_dir():
                        continue

                    # Get just the filename (ignore subdirectories in ZIP)
                    filename = Path(file_info.filename).name

                    # Skip non-txt files
                    if not filename.endswith(".txt"):
                        continue

                    # Extract to output directory
                    source = zip_ref.open(file_info)
                    target = self.output_dir / filename

                    with open(target, "wb") as f:
                        shutil.copyfileobj(source, f)

                    logger.debug(f"Extracted: {filename}")

            logger.info("Extraction complete")

        except zipfile.BadZipFile as e:
            raise ExtractionError(f"Invalid ZIP file: {e}")
        except Exception as e:
            raise ExtractionError(f"Extraction failed: {e}")

    def _verify_extraction(self) -> None:
        """
        Verify that all expected files were extracted.

        Raises:
            ExtractionError: If verification fails
        """
        missing_files = []

        for filename in self.EXPECTED_FILES:
            file_path = self.output_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
            else:
                # Check file is not empty
                if file_path.stat().st_size == 0:
                    missing_files.append(f"{filename} (empty)")

        if missing_files:
            raise ExtractionError(
                f"Missing or empty files after extraction: {missing_files}"
            )

        logger.info("All expected files verified")

        # Log file sizes
        for filename in self.EXPECTED_FILES:
            file_path = self.output_dir / filename
            size_kb = file_path.stat().st_size / 1024
            logger.info(f"  {filename}: {size_kb:.1f} KB")

    def _cleanup(self) -> None:
        """Remove temporary files."""
        if self.zip_path.exists():
            self.zip_path.unlink()
            logger.info("Cleaned up ZIP file")

    def get_available_files(self) -> dict:
        """
        Get information about available dataset files.

        Returns:
            Dictionary with dataset information
        """
        files = {}

        for filename in self.EXPECTED_FILES:
            file_path = self.output_dir / filename
            if file_path.exists():
                files[filename] = {
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "size_kb": file_path.stat().st_size / 1024,
                }

        return files


def main() -> int:
    """
    Main entry point for the download script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Download NASA Turbofan Engine Degradation Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_data.py
    python download_data.py --output-dir ./my_data
    python download_data.py --force
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the dataset (default: data/raw)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        downloader = TurbofanDataDownloader(output_dir=args.output_dir)
        output_path = downloader.download_and_extract(force=args.force)

        print(f"\nDataset downloaded to: {output_path}")
        print("\nAvailable files:")
        for filename, info in downloader.get_available_files().items():
            print(f"  - {filename}: {info['size_kb']:.1f} KB")

        return 0

    except DownloadError as e:
        logger.error(f"Download failed: {e}")
        return 1

    except ExtractionError as e:
        logger.error(f"Extraction failed: {e}")
        return 1

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
