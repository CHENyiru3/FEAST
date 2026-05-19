"""Public de novo simulation API."""

from __future__ import annotations

from .builder import (
    BlueprintBuilder as SimulationBlueprintBuilder,
    ParameterCloudBuilder as SimulationParameterBuilder,
    generate_virtual_slice_from_design as _generate_from_design_impl,
    plot_blueprint,
)
from .conditional import (
    ConditionalReferenceConfig as ReferenceFitConfig,
    ConditionalReferenceModel as SimulationReference,
    VirtualSliceGenerationConfig as SimulationConfig,
    fit_virtual_slice_reference as _fit_reference_impl,
    generate_virtual_slice as _simulate_from_reference_impl,
)
from .core import SliceBlueprint as SimulationBlueprint, load_blueprint
from .pattern import (
    SpatialPatternBuilder as SimulationPatternBuilder,
    compose_gene_pattern as _compose_pattern_impl,
    evaluate_spatial_motif as _evaluate_motif_impl,
    plot_gene_pattern as _plot_pattern_impl,
    plot_pattern_panel,
)
from .stack import simulate_stack as _simulate_stack_impl


def simulate_from_design(*args, **kwargs):
    return _generate_from_design_impl(*args, **kwargs)


def fit_reference(*args, **kwargs):
    return _fit_reference_impl(*args, **kwargs)


def simulate_from_reference(*args, **kwargs):
    return _simulate_from_reference_impl(*args, **kwargs)


def simulate_stack(*args, **kwargs):
    return _simulate_stack_impl(*args, **kwargs)


def evaluate_motif(*args, **kwargs):
    return _evaluate_motif_impl(*args, **kwargs)


def compose_pattern(*args, **kwargs):
    return _compose_pattern_impl(*args, **kwargs)


def plot_pattern(*args, **kwargs):
    return _plot_pattern_impl(*args, **kwargs)


_PUBLIC_NAMES = [
    "SimulationBlueprint",
    "SimulationBlueprintBuilder",
    "SimulationParameterBuilder",
    "SimulationPatternBuilder",
    "ReferenceFitConfig",
    "SimulationReference",
    "SimulationConfig",
    "compose_pattern",
    "evaluate_motif",
    "fit_reference",
    "simulate_from_reference",
    "simulate_stack",
    "simulate_from_design",
    "load_blueprint",
    "plot_pattern",
    "plot_pattern_panel",
    "plot_blueprint",
]

__all__ = list(_PUBLIC_NAMES)


def __dir__():
    return sorted(__all__)


for _name in [
    "simulate_from_design",
    "fit_reference",
    "simulate_from_reference",
    "simulate_stack",
    "evaluate_motif",
    "compose_pattern",
    "plot_pattern",
]:
    globals()[_name].__name__ = _name
    globals()[_name].__qualname__ = _name

for _obj, _name in [
    (SimulationBlueprint, "SimulationBlueprint"),
    (SimulationBlueprintBuilder, "SimulationBlueprintBuilder"),
    (SimulationParameterBuilder, "SimulationParameterBuilder"),
    (SimulationPatternBuilder, "SimulationPatternBuilder"),
    (ReferenceFitConfig, "ReferenceFitConfig"),
    (SimulationReference, "SimulationReference"),
    (SimulationConfig, "SimulationConfig"),
]:
    _obj.__name__ = _name
    _obj.__qualname__ = _name

del _name, _obj
