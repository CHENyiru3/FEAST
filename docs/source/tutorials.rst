Tutorials
=========

The repository includes a Jupyter notebook demonstrating core workflows.

- **Single-slice simulation** (:file:`example_single_sim.ipynb`): Simulate
  sequencing-based (Visium) and imaging-based (MERFISH) ST data, with and
  without controlled expression alterations.

Single-slice simulation
-----------------------

Basic simulation from a reference AnnData:

.. code-block:: python

    from FEAST import simulator
    import scanpy as sc

    adata = sc.read_h5ad("your_spatial_data.h5ad")
    simulated = simulator.simulate_single_slice(adata=adata, verbose=True)

Simulation with expression alteration:

.. code-block:: python

    from FEAST.modeling.marginal_alteration import AlterationConfig

    config = AlterationConfig.mean_only(fold_change=0.95)
    altered = simulator.simulate_single_slice(
        adata=adata,
        alteration_config=config,
        use_heuristic_search=True,
    )

Alignment simulation
--------------------

Generate paired datasets with known geometric transformations for benchmarking
alignment algorithms:

.. code-block:: python

    from FEAST import alignment

    original, rotated = alignment.simulate_alignment_rotation(
        adata=adata,
        rotation_angle=30.0,
        data_type="imaging",
    )

Deconvolution simulation
------------------------

Create ground-truth data with known cell-type mixtures:

.. code-block:: python

    from FEAST import deconvolution

    benchmark = deconvolution.create_deconvolution_benchmark_data(
        adata=single_cell_adata,
        downsampling_factor=0.25,
        grid_type="hexagonal",
        cell_type_key="cell_type",
    )

De novo virtual slice generation
--------------------------------

Build virtual slices from blueprints, parameter clouds, and spatial patterns:

.. code-block:: python

    from FEAST import de_novo

    genes = ["GeneA", "GeneB", "GeneC"]

    blueprint = (
        de_novo.SimulationBlueprintBuilder.rectangular_grid(4, 4)
        .set_domains(["cortex"] * 8 + ["medulla"] * 8)
        .build()
    )

    parameter_cloud = (
        de_novo.SimulationParameterBuilder.from_gene_names(genes)
        .set_all(mean=3.0, variance=5.0, zero_prop=0.2)
        .build()
    )

    patterns = (
        de_novo.SimulationPatternBuilder.from_gene_names(genes)
        .gradient("GeneA", axis="x")
        .hotspot("GeneB", center=[0.5, 0.5], radius=0.25)
        .build()
    )

    virtual_slice = de_novo.simulate_from_design(
        blueprint, parameter_cloud,
        pattern_spec=patterns, random_seed=7,
    )
