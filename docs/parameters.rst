Parameter reference
===================

These are the released CLI defaults, before project/user settings or explicit
arguments are applied. The napari plugin has its own controls and defaults; see
:doc:`napari_guide`. See :doc:`cli_guide` for configuration precedence.

In JSON, put each setting in its named section and use underscores, for example
``"matching": {"top_k": 50}``. CLI flags use hyphens. Boolean flags accept both
forms, such as ``--gpu`` and ``--no-gpu``,
``--use-ransac`` and ``--no-use-ransac``. Optional numeric settings can be
set to ``null`` in JSON; the CLI numeric flags do not accept the word ``null``.

Inputs and output size
----------------------

.. list-table:: Top-level registration configuration
   :header-rows: 1
   :widths: 28 24 48

   * - JSON key
     - CLI flag / default
     - Meaning
   * - ``mode``
     - ``--mode`` / ``auto``
     - ``auto``, ``image``, ``mask``, or ``features``. Input paths determine execution.
   * - ``fixed``, ``moving``
     - ``--fixed``, ``--moving`` / ``null``
     - Reference and source intensity images.
   * - ``fixed_mask``, ``moving_mask``
     - ``--fixed-mask``, ``--moving-mask`` / ``null``
     - Existing instance label images.
   * - ``fixed_features``, ``moving_features``
     - ``--fixed-features``, ``--moving-features`` / ``null``
     - Existing feature CSVs; take precedence over features extracted from masks.
   * - ``matches``
     - ``--matches`` / ``null``
     - Existing match CSV, using feature row indices ``idx1`` and ``idx2``.
   * - ``fixed_shape``
     - JSON only on ``run`` / ``null``
     - ``[height, width]`` in pixels. A Fixed mask takes precedence; otherwise this value precedes inference from centroid extents.

The standalone ``match`` and ``warp`` commands accept
``--fixed-shape HEIGHT WIDTH``; it is required for ``warp``. ``run`` does
not accept that flag. Pass ``--config FILE`` to load registration-only JSON.

Segmentation and feature filtering
----------------------------------

These settings belong to ``segmentation``.

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - JSON key / CLI flag
     - Default
     - Meaning
   * - ``channel_axis`` / ``--channel-axis``
     - ``auto``
     - ``auto``, ``first``, ``last``, or ``none``; see the image-layout limits in :doc:`cli_guide`.
   * - ``registration_channel`` / ``--registration-channel``
     - ``-1``
     - Zero-based segmentation channel for 2D multichannel inputs; ``-1`` selects the last channel.
   * - ``gpu`` / ``--gpu``
     - ``false``
     - Use the GPU for Cellpose segmentation.
   * - ``min_area`` / ``--min-area``
     - ``null``
     - Retain cells with area greater than or equal to this number of pixels.
   * - ``max_area`` / ``--max-area``
     - ``null``
     - Retain cells with area less than or equal to this number of pixels.

Area filtering applies during feature extraction; it does not remove labels
from the saved mask. It is not reapplied to a supplied feature CSV. The
standalone ``segment`` command accepts channel and GPU settings; standalone
``features`` accepts the area bounds. Standalone ``match`` extracts mask
features with default area bounds.

The CLI uses Cellpose-SAM (``cpsam``) with diameter ``None``, flow threshold
``0.4``, cell probability threshold ``0.0``, and minimum size ``15``.
These model settings are not exposed as CLI flags or accepted keys in the
registration JSON. Feature extraction uses three nearest neighbors for its
topology descriptors.

Matching
--------

These settings belong to ``matching`` and are accepted by ``run`` and
``match``.

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - JSON key / CLI flag
     - Default
     - Meaning
   * - ``top_k`` / ``--top-k``
     - ``50``
     - Target number of representative matches; see the coverage and clustering exceptions below.
   * - ``feature_weight`` / ``--feature-weight``
     - ``1.0``
     - Weight of standardized morphological feature distances.
   * - ``topology_weight`` / ``--topology-weight``
     - ``0.35``
     - Weight of standardized local-neighborhood feature distances.
   * - ``position_weight`` / ``--position-weight``
     - ``4.0``
     - Weight of normalized spatial distances after coarse alignment.
   * - ``distance_threshold`` / ``--distance-threshold``
     - ``2.0``
     - Matching-score cutoff; this is not a pixel distance. ``null`` disables this cutoff for fine matching, subject to the coverage behavior below.
   * - ``spatial_window_size`` / ``--spatial-window-size``
     - ``100.0``
     - Spatial candidate window in pixels. ``null`` selects an internally derived guided window; it does not make fine matching unrestricted.
   * - ``use_spatial_clusters`` / ``--use-spatial-clusters``
     - ``false``
     - Replace the final matches with matching performed separately within spatial clusters.
   * - ``n_clusters`` / ``--n-clusters``
     - ``9``
     - Number of clusters when spatial clustering is enabled.

The default matcher balances landmarks across a 2-by-2 grid of the Fixed field
of view. Its first coverage pass may retain one available pair per region even
when the matching score exceeds ``distance_threshold``. This can also exceed
``top_k`` when that value is smaller than four. With spatial clustering enabled,
``top_k`` is applied within each cluster, so the total can exceed it. Neither
the score nor the count is an automatic quality guarantee.

The coarse matching stage is attempted when each side contains at least 20
cells. It uses morphology-guided candidates and a rigid coarse model even if the
final method is similarity or affine. In ``run``,
``ransac_residual_threshold`` and ``ransac_max_trials`` also influence this
coarse fit: its residual threshold is ``max(5, 2 * threshold)`` pixels and its
trial count is clamped to 200–2000. Disabling final RANSAC does not disable this
coarse estimation.

Accepted but inactive options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following ``matching`` keys and corresponding flags are accepted and saved,
but are not used by the released CLI registration service:

.. list-table::
   :header-rows: 1
   :widths: 38 42 20

   * - JSON key
     - CLI flag
     - Default
   * - ``use_topology_filtering``
     - ``--use-topology-filtering``
     - ``false``
   * - ``k_pos_nei``
     - ``--k-pos-nei``
     - ``5``
   * - ``k_neighbor``
     - ``--k-neighbor``
     - ``5``
   * - ``tau_pos``
     - ``--tau-pos``
     - ``0.5``
   * - ``tau_nei``
     - ``--tau-nei``
     - ``0.3``
   * - ``tau_map``
     - ``--tau-map``
     - ``0.3``

These inactive controls are separate from ``topology_weight``, which is active.

Transform estimation
--------------------

These settings belong to ``transform``.

.. list-table::
   :header-rows: 1
   :widths: 32 16 52

   * - JSON key / CLI flag
     - Default
     - Meaning
   * - ``method`` / ``--method``
     - ``rigid``
     - ``rigid``: rotation and translation; ``similarity``: also uniform scale; ``affine``: also directional scaling and shear.
   * - ``use_ransac`` / ``--use-ransac``
     - ``false``
     - Robust final fitting for ``rigid`` only. Ignored for similarity and affine.
   * - ``ransac_max_trials`` / ``--ransac-max-trials``
     - ``1000``
     - Maximum final rigid RANSAC trials.
   * - ``ransac_residual_threshold`` / ``--ransac-residual-threshold``
     - ``2.0``
     - Final rigid RANSAC inlier threshold, in Fixed-image pixels.
   * - ``residual_prune_quantile`` / ``--residual-prune-quantile``
     - ``null``
     - Refit after dropping residuals above a quantile, for example ``0.9``. Active only for ``0 < q < 1``, when at least three pairs remain and at least one pair is removed.
   * - ``initial_coarse_transform`` / ``--initial-coarse-transform``
     - ``null``
     - Path to a JSON 3-by-3 Moving-to-Fixed matrix used to initialize matching in ``run``.

Rigid and similarity fits require at least two match pairs; affine fits require
at least three. Degenerate, duplicated, or poorly distributed points can still
give an unreliable fit. Prefer well-spread landmarks over the minimum count.

An initial-transform JSON may contain the raw nested 3-by-3 array or an object
with a ``matrix``, ``moving_to_fixed``, ``affine``, or ``transform`` key.
The exported ``transform.moving_to_fixed.json`` is suitable. All matrices use
pixel ``(x, y)`` coordinates, not row-column order or physical units.

.. note::

   ``match`` accepts the transform flags listed above but does not apply them
   in the release. Standalone ``transform`` applies the final-fit controls but
   ignores ``--initial-coarse-transform``; matching has already been completed.
   Use ``run`` when initializing matching from an existing matrix.

Output controls
---------------

These settings belong to ``output``. Only ``output_dir`` has a CLI flag;
set the save switches in a configuration file.

.. list-table::
   :header-rows: 1
   :widths: 32 25 43

   * - JSON key
     - Default
     - Meaning
   * - ``output_dir``
     - ``outputs/topoalign-run``
     - Destination directory; overridden by ``--output-dir``.
   * - ``save_overlay``
     - ``true``
     - Save an overlay when a resampled source and compatible Fixed reference are available.
   * - ``save_registered_moving``
     - ``true``
     - Save the resampled Moving image or mask.
   * - ``save_features``
     - ``true``
     - Save the Fixed and Moving feature tables.
   * - ``save_matches``
     - ``true``
     - Save/register the initial match artifact. The final transform stage still writes ``matches.csv`` even when this is ``false``.

A resampling source still triggers warping and ``valid_overlap_mask.tif`` export
when ``save_registered_moving`` is false. Existing output directories are not
cleared. See :doc:`outputs` for the exact files and export conditions.

Tuning workflow
---------------

#. Check the Fixed/Moving direction, image axes, segmentation channel, and masks.
   Correct segmentation errors before tuning matching.
#. Start with a rigid fit and inspect both landmark coverage and the overlay.
#. If too few candidates are found, check field-of-view overlap and cell counts.
   Adjust the spatial window or supply an initial transform when appropriate.
   Increasing ``top_k`` cannot create missing candidates.
#. For outliers, try final rigid RANSAC or a residual-pruning quantile. Recheck
   the spatial coverage after pruning.
#. Use similarity when uniform scale differences are expected; use affine only
   when the geometry justifies its additional degrees of freedom.
#. Change one group of settings at a time and keep separate output directories.

Residual tolerances depend on pixel size, cell dimensions, and the downstream
analysis. A low fitted residual alone does not demonstrate correct registration.
