Outputs and quality checks
==========================

This page describes the CLI exports. The napari plugin has different save
controls and filenames; see :doc:`napari_guide`.

Full-pipeline outputs
---------------------

A successful image-based ``topoalign run`` with default save settings produces:

.. code-block:: text

   <output_dir>/
   |-- segmentation_fixed/
   |   |-- mask.tif
   |   `-- segmentation.json
   |-- segmentation_moving/
   |   |-- mask.tif
   |   `-- segmentation.json
   |-- config.resolved.json
   |-- fixed_features.csv
   |-- moving_features.csv
   |-- matches.csv
   |-- transform.moving_to_fixed.json
   |-- registered_moving.tif
   |-- valid_overlap_mask.tif
   |-- overlay.tif
   |-- diagnostics.json
   `-- result.json

This is a conditional set of files, not a required checklist for every workflow.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - File
     - Contents and export condition
   * - ``segmentation_fixed/``, ``segmentation_moving/``
     - Created only for a side segmented during this run. Existing input masks are not copied here.
   * - ``config.resolved.json``
     - Effective registration configuration, written at the start. Its presence does not prove success.
   * - ``fixed_features.csv``, ``moving_features.csv``
     - Feature tables used for registration, if ``save_features`` is true.
   * - ``matches.csv``
     - Final match table with ``residual_px``. Written after transform fitting, including when ``save_matches`` is false.
   * - ``transform.moving_to_fixed.json``
     - Method, direction, coordinate space, and 3-by-3 transform matrix.
   * - ``registered_moving.tif``
     - Moving intensity data, or the Moving mask when no Moving image was supplied. Requires a resampling source and ``save_registered_moving=true``.
   * - ``valid_overlap_mask.tif``
     - Binary geometric coverage in the Fixed output grid. Written whenever warping runs, independent of the image save switch.
   * - ``overlay.tif``
     - RGB comparison of Fixed (magenta) and registered Moving (green), when enabled and their display projections have matching shapes.
   * - ``diagnostics.json``
     - Match count, transform, mean and maximum residual, elapsed time, warnings, and errors.
   * - ``result.json``
     - Completed-run manifest containing stage timings, artifact paths, and diagnostics.

A feature-only run with no Moving image or mask has no warp, overlap-mask, or
overlay output. When reusing features and also supplying a Moving image, supply
a Fixed image/mask for the overlay or set ``output.save_overlay=false``.
To set the output grid without a Fixed mask, use ``fixed_shape`` in the
registration JSON.

.. warning::

   Files can remain after an unsuccessful run. The release does not write a
   final failure manifest when an exception interrupts the pipeline, and an old
   ``result.json`` can remain if an output directory is reused. Use a new
   directory for each run and check the command's exit status and terminal log.

The manifest has ``product``, ``status``, ``run_id``, ``stages``,
``artifacts``, ``diagnostics``, ``warnings``, and ``errors`` fields.
Artifact paths may be relative to the working directory from which the command
ran. Do not automatically interpret them as relative to ``result.json``.
The segmentation directories are not individually indexed in the artifact map.
When ``save_matches=false``, the final match CSV can exist without a
``matches`` entry in that map.

Stage-command outputs
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Command
     - Files written in its output directory
   * - ``segment``
     - ``mask.tif``, ``segmentation.json``
   * - ``features``
     - ``features.csv``, ``features.json``
   * - ``match``
     - ``matches.csv``, ``matching_diagnostics.json``
   * - ``transform``
     - ``transform.moving_to_fixed.json``, ``matches.with_residuals.csv``
   * - ``warp``
     - ``registered_moving.tif``, ``valid_overlap_mask.tif``
   * - ``inspect``
     - None; prints inspection information.

``segment`` metadata records the source path, mode, mask shapes, channel
settings, and GPU flag. For a Z-stack, ``mask_shape`` describes the 3D result,
while the saved ``mask.tif`` is the 2D projection described by
``projected_mask_shape``. The CLI does not save the 3D label volume here.
``features.json`` records the source mask, projected shape, cell count, and
column names. ``matching_diagnostics.json`` contains the match count and
reference shape; it has no post-fit residuals.

Feature table schema
--------------------

The tables exported by ``features`` and by mask/image-based ``run`` have one
row per retained instance. Area filtering happens before neighborhood descriptors
are calculated. Features are computed from label geometry, not image intensity.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Columns
     - Meaning
   * - ``cell_id``
     - Original instance label. It need not equal the feature row number.
   * - ``centroid_x``, ``centroid_y``
     - Cell centroid in pixel column/row coordinates.
   * - ``area``, ``perimeter``
     - Region area in pixels and boundary-length estimate in pixels.
   * - ``eccentricity``, ``solidity``, ``major_axis_length``, ``minor_axis_length``, ``orientation``
     - Region properties from scikit-image; axis lengths are in pixels and orientation is in radians.
   * - ``roundness``
     - ``4 * pi * area / perimeter**2``, with safe handling of zero perimeter.
   * - ``aspect_ratio``, ``elongation``, ``equivalent_diameter``
     - Major/minor axis ratio, ``1 - minor/major`` clipped to [0, 1], and diameter of an equal-area circle.
   * - ``pos_x_norm``, ``pos_y_norm``
     - Centroid divided by the mask width and height, respectively.
   * - ``axis_vec_x``, ``axis_vec_y``
     - Cosine and sine of the stored orientation.
   * - ``nn_dist_1``, ``nn_dist_2``, ``nn_dist_3``
     - Neighbor distances normalized by the median nearest-neighbor distance, with that scale bounded below by one pixel.
   * - ``local_density``
     - The same global distance scale divided by mean neighbor distance.
   * - ``neighbor_area_ratio_mean``, ``neighbor_roundness_mean``
     - Mean neighboring area relative to the current cell area, and mean neighboring roundness.

The automatic matcher requires the ten morphology columns ``area``,
``perimeter``, ``roundness``, ``eccentricity``, ``solidity``,
``major_axis_length``, ``minor_axis_length``, ``aspect_ratio``,
``elongation``, and ``equivalent_diameter``, plus the six neighborhood
columns in the last three rows of the table, centroid coordinates, and normalized
positions under the default settings. Preserve the complete generated table
when exchanging data between stages. Disabling a feature weight does not remove
all column checks.

If feature CSVs are supplied to ``run``, the exported tables retain the supplied
columns; missing morphology is not reconstructed. Coordinate-only tables are
usable when matches are supplied explicitly.

Match table schema
------------------

``idx1`` and ``idx2`` index the Fixed and Moving feature rows, starting at zero.
``cell_id_1`` and ``cell_id_2`` report the corresponding labels.
``distance`` is the weighted matching score; smaller means a better match
under the chosen scoring settings. It is not the fitted geometric error or a
probability. Columns ending in ``_1`` and ``_2`` report the paired feature
values. Cluster-based matching can also include ``cluster_id``.

After transform fitting, ``residual_px`` is the Euclidean distance between the
transformed Moving centroid and its Fixed partner, in Fixed pixels. Residual
pruning can remove rows and refit the transform. RANSAC fitting alone does not
remove all non-inlier rows from the exported table or mark them with an inlier
column; inspect residuals as well.

Transform convention
--------------------

The matrix maps Moving points to Fixed points:

.. code-block:: text

   [x_fixed, y_fixed, 1]^T = matrix @ [x_moving, y_moving, 1]^T

Here ``x`` is the image column and ``y`` is the image row. The full pipeline
records ``"direction": "moving_to_fixed"`` and
``"coordinate_space": "pixel_xy"``. The standalone ``transform`` export
omits ``coordinate_space`` but uses the same convention. Pass this file directly
to ``topoalign warp``; the resampler handles the inverse mapping internally.
These files describe global rigid, similarity, or affine geometry, not a
deformation field.

Image types and display
-----------------------

``run`` loads intensity images as ``float32``, so its registered intensity
output is also normally ``float32``. Loaded/generated masks are ``int32``.
Standalone ``warp`` preserves the input dtype, rounding and clipping integer
results after interpolation. Both paths write ordinary TIFF files; they do not
preserve a slide pyramid or reproduce all input metadata.

The overlap mask is a ``uint8`` array with values 0 and 1. It represents the
transformed rectangular Moving pixel domain, not cell overlap, tissue overlap,
or registration accuracy.

The overlay independently scales each display image between its 1st and 99th
intensity percentiles and saves ``uint8`` RGB. Bright agreement appears white.
Its projection helper uses the last channel for channel-last images with up to
four channels, or a maximum projection over axis 0 for other three-dimensional
arrays. It does not necessarily display the selected segmentation channel.
Inspect relevant channels independently before accepting a result.

Assess a result
---------------

#. Confirm that the command completed and that the manifest belongs to this run.
#. Inspect masks for missed, merged, or spurious cells.
#. Check that matched cells correspond visually and span the overlapping tissue,
   rather than concentrating in one small region.
#. Review global alignment, tissue edges, and individual cells in the overlay and
   registered image.
#. Review the distribution of ``residual_px`` together with spatial coverage.
   The full-run diagnostics export the mean and maximum, not P95. Compute other
   summaries from the final CSV if needed.
#. Check whether the geometric field-of-view overlap is adequate for the
   downstream analysis.

``diagnostics.match_count`` in a full run is recorded **before** residual
pruning. Count the final match CSV rows to obtain the retained count. Empty
warning/error lists mean no messages were recorded; they are not a scientific
quality assessment. Fit residuals are measured on the landmarks used to estimate
the transform, so independent landmarks are preferable when measuring accuracy.
