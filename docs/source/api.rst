API Reference
=============

Public API
----------

.. automodule:: FEAST
   :members: simulate, generate, generate_from, fit, decode, Alteration
   :imported-members: false

Classes
-------

.. autoclass:: FEAST.GeneParameterSimulator
   :members:

.. autoclass:: FEAST.SliceBlueprint
   :members:

.. autoclass:: FEAST.ReferenceFitConfig
   :members:

.. autoclass:: FEAST.SimulationConfig
   :members:

Utility functions
-----------------

.. autofunction:: FEAST.stats_to_theta
.. autofunction:: FEAST.theta_to_stats

Spatial transforms
------------------

.. automodule:: FEAST.spatial_transform
   :members:

Alignment subpackage
--------------------

.. automodule:: FEAST.alignment
   :members:

.. automodule:: FEAST.alignment.alignment_simulator
   :members:

.. automodule:: FEAST.alignment.spatial_align_alter
   :members:

Deconvolution subpackage
------------------------

.. automodule:: FEAST.deconvolution
   :members:

.. automodule:: FEAST.deconvolution.deconvolution_simulator
   :members:

.. automodule:: FEAST.deconvolution.generate_deconvolution
   :members:

De novo generation subpackage
-----------------------------

.. automodule:: FEAST.de_novo
   :members:

.. automodule:: FEAST.de_novo.builder
   :members:

.. automodule:: FEAST.de_novo.conditional
   :members:

.. automodule:: FEAST.de_novo.core
   :members:

.. automodule:: FEAST.de_novo.pattern
   :members:

.. automodule:: FEAST.de_novo.quantile_field
   :members:

.. automodule:: FEAST.de_novo.stack
   :members:

Internal modules
----------------

.. automodule:: FEAST.FEAST_core.simulator
   :members:

.. automodule:: FEAST.FEAST_core.parameter_cloud
   :members:

.. automodule:: FEAST.FEAST_core.count_decoding
   :members:

.. automodule:: FEAST.FEAST_core.theta_transform
   :members:

.. automodule:: FEAST.modeling.StudentT_mixture_model
   :members:

.. automodule:: FEAST.modeling.Beta_mixture_model
   :members:

.. automodule:: FEAST.modeling.marginal_alteration
   :members:
