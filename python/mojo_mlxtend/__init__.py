"""Mojo-accelerated frequent-pattern mining with an mlxtend-compatible API."""

from .frequent_patterns import apriori, association_rules, fpgrowth, fpmax

__all__ = ["apriori", "association_rules", "fpgrowth", "fpmax"]
__version__ = "0.1.0"
