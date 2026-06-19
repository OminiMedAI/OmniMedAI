"""Pathology feature extractors with lazy optional dependencies."""

__all__ = ['CellProfilerExtractor', 'TITANExtractor']


def __getattr__(name):
    if name == "CellProfilerExtractor":
        from .cellprofiler_extractor import CellProfilerExtractor

        return CellProfilerExtractor
    if name == "TITANExtractor":
        from .titan_extractor import TITANExtractor

        return TITANExtractor
    raise AttributeError(name)
