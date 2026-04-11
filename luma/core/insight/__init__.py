"""
luma.core.insight — Pattern Recognition & Insight Engine

Public exports for the insight module.
"""

from luma.core.insight.schemas import (
    PatternResult,
    TrendResult,
    Insight,
    InsightReport,
)
from luma.core.insight.pattern_detector import PatternDetector
from luma.core.insight.trend_analyzer import TrendAnalyzer
from luma.core.insight.insight_generator import InsightGenerator
from luma.core.insight.insight_engine import InsightEngine

__all__ = [
    "InsightEngine",
    "PatternDetector",
    "TrendAnalyzer",
    "InsightGenerator",
    "PatternResult",
    "TrendResult",
    "Insight",
    "InsightReport",
]
