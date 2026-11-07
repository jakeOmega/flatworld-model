"""flatworld -- a from-scratch climate model for the flat-plane Pantheon setting.

See DESIGN.md for the architecture and the list of fixed givens. The prior climate chain it
replaces is retired in ../CHANGELOG_climate.md.
"""
from . import constants, geometry  # noqa: F401

__all__ = ["constants", "geometry"]
