"""flatworld.stage_c_map -- compatibility ALIAS for :mod:`flatworld.climate_map` (ROADMAP GEO-22).

The stage was renamed ``stage_c_map`` -> ``climate_map``: "Stage C" named a lettered A->B->C sequence
whose other members were fused into the AD core (``stage_ad_core``), and the operation is a *dimension*
change ((x,z) -> (x,e)), not downscaling. This shim makes ``flatworld.stage_c_map`` the SAME module
object as ``flatworld.climate_map`` (not a re-export), so attribute access, monkeypatching, and
``is``-identity all behave identically through either name. New code should import
``flatworld.climate_map``.
"""
import sys

from . import climate_map as _climate_map

sys.modules[__name__] = _climate_map
