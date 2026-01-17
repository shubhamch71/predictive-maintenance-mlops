"""
Version information for Predictive Maintenance MLOps Platform.

This module provides version metadata for the package.
The version follows Semantic Versioning (https://semver.org/).

Version format: MAJOR.MINOR.PATCH
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible functionality additions
- PATCH: Backwards-compatible bug fixes
"""

__version__ = "1.0.0"
__version_info__ = tuple(int(x) for x in __version__.split("."))

# Package metadata
__title__ = "predictive-maintenance-mlops"
__description__ = "Production-grade MLOps platform for predictive maintenance"
__author__ = "ML Platform Team"
__author_email__ = "ml-platform@example.com"
__license__ = "MIT"
__url__ = "https://github.com/your-org/predictive-maintenance-mlops"

# Build metadata (populated during CI/CD)
__build_date__ = ""
__git_commit__ = ""
__git_branch__ = ""


def get_version() -> str:
    """Return the package version string."""
    return __version__


def get_version_info() -> dict:
    """Return complete version information as a dictionary."""
    return {
        "version": __version__,
        "title": __title__,
        "description": __description__,
        "author": __author__,
        "license": __license__,
        "url": __url__,
        "build_date": __build_date__,
        "git_commit": __git_commit__,
        "git_branch": __git_branch__,
    }
