"""UI utilities for segment-wise annotation management."""

from .app import AnnotationApp, build_parser, main
from .browser import launch_browser_and_collect
from .constants import ANNOTATION_COLUMNS, DEFAULT_ROOT
from .session import AnnotationSession

__all__ = [
    "AnnotationApp",
    "AnnotationSession",
    "ANNOTATION_COLUMNS",
    "DEFAULT_ROOT",
    "build_parser",
    "main",
    "launch_browser_and_collect",
]
