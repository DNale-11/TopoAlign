Overview
========

What TopoAlign does
-------------------

TopoAlign estimates alignment from corresponding cells. It can segment images,
extract cell features from instance masks, identify corresponding cells, and
fit a geometric transform. The command-line workflow can start at any of these
stages, so existing segmentation and feature tables can be reused.

The typical CLI workflow is::

   Fixed image   -> Fixed mask   -> Fixed features  --+
                                                     +-> Matches -> Transform
   Moving image  -> Moving mask  -> Moving features --+                |
                                                                      v
                         Fixed coordinates <- Resampled Moving image

The same tissue structures must be identifiable in both inputs. Segmentation
quality, overlapping field of view, and the spatial distribution of correct
matches determine whether a fitted transform is useful. A completed run is not
by itself evidence of a biologically correct alignment.

Choose an interface
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 46 20 16

   * - Interface
     - Use it for
     - Start command
     - API key
   * - CLI
     - Reproducible registration, saved configurations, individual stages,
       and scripted batches
     - ``topoalign --help``
     - Not required
   * - napari
     - Image and label inspection, manual segmentation, interactive
       registration, and dedicated WSI tools
     - ``napari``
     - Not required
   * - Agent
     - Natural-language assistance with the local CLI workflow
     - ``topoalign agent``
     - Required

The CLI and napari share the project name but use separate implementations.
CLI registration offers rigid, similarity, and affine transforms. The napari
Normal workflow can refine a rigid alignment with thin-plate splines (TPS),
while its WSI workflow has additional metadata requirements. Do not assume that
identical-looking parameters or exported files are interchangeable.

Inputs and coordinates
----------------------

**Images.** Common inputs are TIFF, PNG, and JPEG. Use lossless source images
when possible. Check the array shape and dtype with ``topoalign inspect``, then
review the intensity range and channel contents in an image viewer before
segmentation. Automatic interpretation uses array shape rather than a complete
microscopy metadata model.

**Masks.** Supply integer instance labels: ``0`` for background and a distinct
positive label for each cell. A binary foreground mask is not an instance
segmentation. Save masks as TIFF, retaining the integer labels; a colorized
preview or JPEG cannot substitute for a label mask.

**Feature tables.** Reuse TopoAlign-generated CSV files whenever possible.
Default matching uses morphology and neighborhood features as well as
centroids. See :doc:`cli_guide` for the full schema and index conventions.

**Coordinates.** Image shapes are written as ``[height, width]`` or ``[Y, X]``.
Feature and transform coordinates are ``(x, y)`` in pixels: ``x`` is the column
and ``y`` is the row. Fixed and Moving may have different image dimensions.
The registered Moving image uses the Fixed canvas, so content outside that
canvas is clipped.

**Multichannel data.** Select the actual cell or nuclear channel deliberately.
Channel indices are zero-based; ``-1`` selects the last channel. For ``C,Y,X``
images, set ``--channel-axis first``. For ``Y,X,C`` images, set
``--channel-axis last``; see the current limitations in :doc:`troubleshooting`.

**Z-stacks.** Some segmentation paths accept Z-stacks and project their masks
to two dimensions. The CLI registration and transform described here are 2D;
this is not a general volumetric registration pipeline. For a first run, prepare
and inspect a 2D projection yourself.

Choose a transform
------------------

``rigid``
   Rotation and translation. Start here when pixel size and tissue geometry
   are comparable.

``similarity``
   Rotation, translation, and a single scale factor. Use it when an isotropic
   scale difference is plausible.

``affine``
   Rotation, translation, anisotropic scaling, and shear. It needs well-spread,
   reliable correspondences; flexibility can also fit incorrect matches.

These names apply to the CLI. Read :doc:`napari_guide` before choosing Normal,
WSI, or TPS behavior in the plugin.

A practical analysis workflow
-----------------------------

#. Pick a representative overlapping region and a common cell or nuclear channel.
#. Inspect both images, then inspect the segmentation masks before matching.
#. Establish a rigid baseline in a new output directory.
#. Check match locations, residuals, overlap coverage, and tissue alignment.
#. Adjust one parameter group at a time and compare results on the same region.
#. Save the configuration and software version with the selected result.
#. Apply the selected workflow to additional images, using a distinct output
   directory for every Moving input.

Residuals describe agreement at fitted matches, not accuracy everywhere in the
image. When quantitative accuracy matters, evaluate independent landmarks that
were not used to estimate the transform.
