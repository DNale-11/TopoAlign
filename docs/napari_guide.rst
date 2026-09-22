napari user guide
=================

The ``napari-cell-registration`` plugin provides interactive segmentation,
mask editing, registration, and whole-slide image (WSI) inspection. This guide
describes the plugin included in TopoAlign 0.1.0. The plugin and the main
``topoalign`` CLI have different registration workflows and defaults.

Install and open the plugin
---------------------------

From the repository root, in your activated Python environment:

.. code-block:: console

   python -m pip install -e "./napari-cell-registration"
   napari

The plugin requires Python 3.10 or later, napari, a working Qt backend, and
Cellpose 4.x. See :doc:`installation` for environment setup. Install the plugin
in the same environment from which you start napari, then restart napari after
installation.

Under **Plugins → Cell Registration**, open the required widget:

* **Cell Segmentation (Cellpose)**: segment one or more image layers.
* **Save Mask Layers**: export existing Labels layers.
* **Manual Segmentation**: create masks with painting and shape tools.
* **Registration Workflow**: register two rounds using their masks.
* **WSI Segmentation**: read whole-slide images and create masks with WSI metadata.

Prepare image and mask layers
-----------------------------

The plugin uses round numbers:

* **Round 1** is the fixed image and reference coordinate system.
* **Round 2** is the moving image to align with Round 1.

Use two-dimensional spatial images. Supported multichannel layouts are
``(C, Y, X)`` and ``(Y, X, C)``; select the channel layout explicitly when automatic
detection is ambiguous. A stack of Z slices is not a multichannel 2D image.
Masks must be two-dimensional instance labels: background is ``0`` and each
cell has a distinct positive integer ID. Load existing masks as **Labels**
layers, with the same pixel grid as their corresponding images.

The workflow reads image and mask arrays. Moving a layer visually with napari
display transforms does not replace registration or resample the underlying
array. For ordinary images, prepare aligned image/mask grids before running.

In version 0.1.0, the normal image-warp routines identify channels by array
dimensions and expect at most 10 channels. The channel-layout selector does
not override this internal warp heuristic. Use the CLI with an explicit
channel axis for images with more channels.

Segment cells
-------------

#. Load the images into napari and open **Cell Segmentation (Cellpose)**.
#. Select one or more **Image layers**. All selected layers use the same settings.
#. Set **Channel layout** and **DAPI channel**. The default DAPI channel is
   ``0``; inspect the image to confirm the nuclear channel.
#. Keep ``model=cpsam`` for the Cellpose-SAM workflow. Enable ``gpu`` only when
   the environment has a compatible PyTorch GPU installation.
#. Click **Segment cells**. The widget adds an ``<image>_mask`` Labels layer
   for each image.
#. Inspect representative crowded, sparse, dim, and boundary regions before
   registration. Correct merged or fragmented cells with napari's Labels tools.

The segmentation defaults are ``diameter=15``, ``flow_threshold=-2.0``,
``cellprob_threshold=1.0``, ``min_size=5``, and ``gpu=false``. These are plugin
defaults, not universal settings for every stain or magnification. The model
selector also lists legacy Cellpose names; availability depends on the
installed Cellpose version and weights.

Large images
~~~~~~~~~~~~

``mode`` controls how the selected nuclear channel is processed:

``auto - chunk only if large``
   Use chunks when the image reaches ``large_image_threshold_mp`` (64 MP by
   default).

``full image - ignore chunk settings``
   Process the entire image in one inference call.

``chunked - use label stitching``
   Always process overlapping tiles. Defaults are ``chunk_size=2048`` and
   ``chunk_overlap=128`` pixels. Keep ``stitch_labels=true`` to reconcile
   overlapping instance labels.

Chunking reduces inference memory, but the ordinary segmentation widget still
holds a full image and full output mask. Reduce chunk size if inference runs
out of memory, and inspect cell boundaries near tile seams.

Save and edit masks
-------------------

``save_masks`` is off by default. Enable it before segmentation to write
``<image>_mask.tif`` into ``output_dir``, or use **Save Mask Layers** after
reviewing the masks:

#. Choose a **Mask layer**, or **All label layers**.
#. Set **Save to** to a separate output directory.
#. Leave **Relabel binary masks** enabled if a binary mask needs connected
   foreground components converted to instance IDs.
#. Click **Save mask layers**.

Binary relabeling cannot separate touching cells. Existing instance masks are
otherwise preserved. This exporter uses uint16 or uint32 according to the
largest label ID. Reusing an output directory can overwrite files with the
same names.

To draw a mask from scratch, open **Manual Segmentation**:

#. Select a 2D image layer and click **New Label Layer**. For a channel-first
   stack, first select or create a single 2D channel layer; this widget assumes
   a 3D input has channel-last layout.
#. Use **Paint**, **Erase**, and **Fill**, with **Brush Size** and
   **Current Label ID**. Click **+ Next Cell** before painting another instance.
#. For shapes, use **Rectangle** or **Polygon**, then click
   **Convert Shapes → Labels** to rasterize them into successive label IDs.
#. Review the Labels layer, then export it with **Save Mask** or
   **Save Mask Layers**.

**New Label Layer** replaces an existing manual layer with the same name.
**Finalize Mask** labels connected foreground components after discarding the
original IDs; touching painted cells can therefore merge. Skip this action
when you need to preserve separately assigned IDs. The manual widget's
**Save Mask** writes uint16; use **Save Mask Layers** if IDs exceed 65,535.
For corrections to an existing segmentation, use napari's own Labels editing
tools rather than creating a new empty manual layer.

Register ordinary images
------------------------

#. Open **Registration Workflow**.
#. Set ``image_round1`` and ``mask_round1`` to the fixed image and its mask.
   Set the Round 2 fields to the moving image and its mask.
#. Set ``image_channel_axis`` to the image layout. To transform all channels,
   choose a multichannel image layer; selecting one 2D channel transforms only
   that layer.
#. Choose ``registration_mode=normal`` for ordinary images, including large
   TIFF images without WSI metadata.
#. Start with the default matching settings and run the widget.
#. Inspect the registered image and mask, matched points, and match lines.
   Examine the center and edges, and check that matches cover the shared tissue.
#. Enable ``save_results`` and set ``output_dir`` for a run whose artifacts
   you want to retain, subject to the release limitations below.

The main controls are:

.. list-table:: Registration controls
   :header-rows: 1
   :widths: 30 15 55

   * - Control
     - Default
     - Meaning
   * - ``top_k``
     - 320
     - Requested matching selection size; the final count can be smaller.
   * - ``max_match_distance_px``
     - 100
     - Spatial matching window. The normal workflow can retry with a wider window.
   * - ``position_weight``
     - 1.0
     - Contribution of position to matching.
   * - ``residual_prune_quantile``
     - 0.0
     - Additional residual pruning; zero disables this option.
   * - ``use_ransac_transform``
     - true
     - Use robust transform fitting; defaults are 1,000 trials and a 2 px residual threshold.
   * - ``min_area`` / ``max_area``
     - 0 / 0
     - Cell-area filters in mask pixels; zero disables the corresponding bound.
   * - ``use_gpu``
     - true
     - Request supported GPU acceleration, with CPU fallback if unavailable.
   * - ``save_results``
     - false
     - Write output files in addition to adding viewer layers.

Normal mode estimates a rigid baseline and uses thin-plate spline (TPS)
refinement when at least 50 final matches remain. With fewer matches, it uses
a rigid fallback. TPS can fit landmarks closely while producing implausible
deformation elsewhere; a small fitting residual alone does not validate the
registration. This widget has no rigid-only mode selector; use the
:doc:`cli_guide` when you require an explicitly rigid result.

``registered_display`` controls the viewer presentation: **colored channel
stack**, **composite RGB preview**, or **composite + colored stack**. Preview
colors and display downsampling are for inspection, not intensity measurements.

Normal-mode output and release limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With saving enabled, normal mode writes ``registered_image.tif``,
``registered_mask.tif``, ``features_round1.csv``, ``features_round2.csv``,
``matches.csv``, ``registered_features_round2.csv``,
``registered_centroids_round2.csv``, and ``transform_info.txt``.
It does not produce the CLI's ``result.json`` manifest.

.. important::

   In version 0.1.0, the exported mask fills background pixels of the warped
   moving mask with fixed-mask labels. For matching-size 2D grayscale images,
   the exported image is the pixelwise maximum of the fixed image and warped
   moving image. These exports therefore include fixed-image content. Use the
   CLI when you need a separate registered moving image or mask for analysis.

Additional limitations in this release:

* With fewer than 50 matches, saving can fail while writing
  ``transform_info.txt`` because it references TPS control points even after
  rigid fallback. Earlier TIFF and CSV files may already have been written.
  Run with saving disabled for viewer inspection, or use CLI rigid registration
  for a complete reproducible export.
* TPS registered point layers and centroid CSVs apply the TPS sampling map
  directly to moving points. That map is fitted in the inverse direction for
  image resampling, so these coordinates should not be used as validated
  moving-to-fixed cell positions. Verify alignment in the warped image;
  for rigid coordinate mapping, use the forward matrix saved by the CLI.
* ``transform_info.txt`` describes the rigid baseline and TPS settings; it
  does not serialize a complete reusable TPS transform.

Whole-slide images
------------------

Install the WSI extra in the plugin environment:

.. code-block:: console

   python -m pip install -e "./napari-cell-registration[wsi]"
   python -c "import openslide; print(openslide.__version__)"

This extra installs ``openslide-python``. A working native OpenSlide runtime
is also required. CellViT++, its ``pathopatch`` dependency, and model weights
are separate requirements; they are not installed by the extra. The CellViT++
path requires CUDA. See :doc:`installation` and :doc:`troubleshooting` before
starting a large run.

Segment and inspect a WSI pair
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Open **WSI Segmentation** and select the fixed and moving WSI files.
   The intended pairing is fixed HE and moving mIF/DAPI.
#. Choose each segmentation backend. Defaults are **CellViT++** for fixed and
   **Cellpose-SAM** for moving. CellViT model choices are **SAM** and **HIPT**.
#. If using CellViT++, enable ``use_gpu`` and specify the correct
   ``resolution``: 0.25 or 0.5 micrometers per pixel. Do not use an arbitrary
   value merely to satisfy the selector.
#. Confirm ``moving_segmentation_channel``. The widget reads QPTIFF channel
   metadata and attempts to select DAPI. This channel selection applies to
   Cellpose; CellViT++ reads the original WSI.
#. Set ``read_level`` for the viewer preview (default 4), and
   ``cellpose_read_level`` for Cellpose inference (default 2). Cellpose masks
   are projected back to the preview grid, so coarse previews can lose small
   objects even when inference uses a finer level.
#. For a region of interest, set ``region_x``, ``region_y``, ``region_width``,
   and ``region_height`` in level-0 pixels. Both positive width and height are
   needed; otherwise the full selected level is read. The same region fields
   are applied to both slides. CellViT++ inference still receives the complete
   WSI path; the displayed masks are restricted to the selected region.
#. Run segmentation and inspect the generated masks. Keep these layers in the
   viewer: their WSI metadata is needed by registration.

``max_cellpose_tiles=80`` prevents unintended long Cellpose runs. If the tile
limit is reached, use a smaller region or a larger ``cellpose_read_level``.
Set the limit to ``0`` only for an intentional unrestricted run. Cellpose
sources above 120 million pixels use streaming inference; smaller sources
are read into memory. ``cellpose_chunk_size=4096`` and
``cellpose_chunk_overlap=128`` control tiles.

``use_cell_shapes=true`` renders CellViT contours; disabling it creates masks
from centroid markers instead of cell boundaries. CellViT inference outputs
and logs are stored under ``fixed_cellvit`` or ``moving_cellvit`` within
``output_dir``. Existing matching CellViT outputs are reused, so use a new
output directory when changing inference settings. **Save Mask Layers** can
export the displayed mask pixels, but a TIFF alone does not preserve the WSI
coordinate metadata needed for a later session.

Register WSI centroids
~~~~~~~~~~~~~~~~~~~~~~

Open **Registration Workflow**, select the corresponding images and generated
mask layers, and explicitly choose ``registration_mode=wsi``. WSI mode
estimates a transform from mask centroids in level-0 coordinates. The default
refinement is **knn local affine**, with ``wsi_knn_k=8``,
``wsi_knn_power=2.0``, and ``wsi_residual_filter=true``.

.. note::

   ``auto`` switches to WSI mode when the fixed mask reaches 64 MP by default,
   regardless of whether WSI metadata exists. Large ordinary TIFF masks can
   therefore fail with a missing-metadata message; select ``normal`` for them.
   Small WSI previews can fall into normal mode; select ``wsi`` explicitly.
   In this release, selecting **4x4 local translation grid** in the WSI widget
   disables KNN refinement and uses the global affine result for centroids;
   it does not apply the advertised local translation refinement in this path.

Enable ``save_results`` to export:

* ``fixed_centroids_level0.csv`` and ``moving_centroids_level0.csv``;
* ``registered_moving_centroids_he_level0.csv``;
* ``wsi_centroid_matches.csv``;
* ``local_deformation_grid_he_level0.csv``;
* ``wsi_transform.json`` and the two normalized centroid metadata JSON files;
* ``centroid_registration_preview.tif``.

The saved forward direction is **mIF level-0 pixels → HE level-0 pixels**.
The deformation grid is defined in HE level-0 coordinates. Keep the transform,
metadata, and matching tables together so origins and downsampling remain
unambiguous.

This widget registers centroids; it does not export a fully resampled
multichannel WSI. ``add_moving_channel_stack`` is a visualization option in
the segmentation widget. Its presence does not mean that the whole slide has
been registered or written to disk.
