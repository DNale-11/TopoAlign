Troubleshooting
===============

Installation and startup
------------------------

The topoalign command is not found
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Activate the environment where you installed the package, then run:

.. code-block:: console

   python -m pip install -e .
   topoalign --help

From the source checkout, ``python topoalign.py --help`` is an alternative.
Check ``python -c "import sys; print(sys.executable)"`` to identify the
interpreter in use.

Dependency installation downgrades PyTorch or fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The published ``requirements.txt`` constrains PyTorch to
``torch>=2.2,<2.5``. An older README example requested PyTorch 2.11.0,
which conflicts with that constraint. Follow :doc:`installation` in a clean
environment and use a matching PyTorch/torchvision pair. Run
``python -m pip check`` after installation.

The core installation, including Matplotlib, supports mask and feature workflows. Cellpose and
napari need the additional dependencies described in the installation guide.

CUDA is unavailable
~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

If the last value is ``False``, confirm that this is the intended Python
environment and that the installed wheel supports CUDA. Check the NVIDIA
driver against the selected PyTorch wheel. Continue with ``--no-gpu``
until CUDA works in this exact environment.

The first segmentation appears to stall
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cellpose-SAM may need to download its model on first use. Check terminal output
for download activity, network errors, or a model-loading error. Try a small
2D crop before a whole image. CPU inference can take substantially longer
than GPU inference.

Segmentation and inputs
-----------------------

No cells or too many fragments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspect the selected image channel and the label mask. Confirm that the input
is not blank, that cells are visible, and that channels were not mistaken for
Z slices. For channel-first images, pass ``--channel-axis first`` explicitly.

``min_area`` and ``max_area`` filter the objects used for feature extraction;
they do not repair or rewrite the original mask. Use corrected instance masks
or the napari manual-segmentation workflow when needed.

A binary mask is treated as one object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TopoAlign expects instance labels. All pixels with label ``1`` describe one
instance, even if they belong to disconnected regions. Convert the mask to a
valid instance segmentation upstream, checking whether touching cells need
to be separated. Do not use a colorized mask screenshot.

Multichannel data or Z-stacks are interpreted incorrectly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatic shape detection treats a three-dimensional array whose last
dimension is at most four as a channel-last 2D image; otherwise it may treat
it as a Z-stack. This can misinterpret ``C,Y,X`` data.

Specify the channel layout for multichannel images. For channel-last images
with more than four channels, the current full-run shape inference can still
fail even with ``--channel-axis last``. Use a prepared 2D registration channel
or a channel-first array instead. Do not assume a complete 3D volume will be
resampled by the 2D registration pipeline.

Out of memory
~~~~~~~~~~~~~

Try a smaller representative region or precompute masks and reuse them. CPU
mode avoids GPU memory limits but still needs sufficient system memory. The
standard CLI loads images into memory; it is not an OpenSlide streaming
pipeline. See :doc:`napari_guide` for dedicated WSI tools and their limitations.

Matching and transforms
-----------------------

Few matches or no matches
~~~~~~~~~~~~~~~~~~~~~~~~~

#. Confirm that the inputs show overlapping tissue and comparable cell types.
#. Compare segmentation quality, object scale, and cell density on both sides.
#. Check area filters and confirm that the retained feature tables are nonempty.
#. Inspect the spatial window and distance threshold; broaden constraints
   cautiously, changing one value at a time.
#. For externally supplied feature tables, use the schema in :doc:`cli_guide`.
   Centroid-only CSV files do not provide the default matching features.

A more flexible transform cannot compensate for incorrect correspondences.

The transform has the wrong direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CLI transform files map Moving coordinates to Fixed coordinates. Apply the
stored matrix to ``[x_moving, y_moving, 1]``, not to ``[row, column, 1]``.
Image resampling internally uses an inverse mapping; this does not change the
direction recorded in the transform file. If you need Fixed-to-Moving
coordinates, invert the matrix explicitly and label that direction correctly.

The residual is small but the tissue is misaligned
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reported residuals are measured at the correspondences used in fitting.
Repeated tissue patterns or a compact set of incorrect matches can produce
small residuals. Inspect match coverage and tissue structures across the
image, and assess independent landmarks where quantitative accuracy matters.

RANSAC has no effect
~~~~~~~~~~~~~~~~~~~~

In the released CLI service, RANSAC is used only for the ``rigid`` method.
Similarity and affine fitting do not use it even if the option is set.
Some accepted topology-filter controls are also inactive in the CLI matching
service; :doc:`parameters` identifies them.

Configuration and output
------------------------

Unknown configuration keys
~~~~~~~~~~~~~~~~~~~~~~~~~~

``topoalign run --config`` accepts a registration-only JSON object.
The application settings example ``topoalign.config.example.json`` contains
other sections and must not be passed directly to that option. See
:doc:`parameters` for both formats and their precedence.

The selected input mode does not match the executed stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current full-run service chooses stages from supplied image, mask,
feature-table, and match paths. The ``mode`` setting does not independently
enforce or validate those inputs. Remove stale paths from reused configurations
and inspect ``config.resolved.json`` to see the effective inputs.

A feature-only result uses the wrong canvas size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``"fixed_shape": [HEIGHT, WIDTH]`` at the top level of the
registration JSON. The ``run`` command does not have a
``--fixed-shape`` flag. Without an image, mask, or explicit shape, the
canvas is inferred from feature centroids and may exclude image borders. When
reusing features, a Fixed intensity image alone does not determine the canvas;
supply a Fixed mask or an explicit ``fixed_shape``.

An expected image or overlay is missing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A mask-only run writes resampled labels to ``registered_moving.tif`` and can
produce an overlay of label values. A feature-only run with no Moving image or
mask has nothing to resample. An overlay needs a Fixed image or mask and a
compatible registered display shape, as well as enabled output settings.
Check ``result.json`` for the files actually produced;
see :doc:`outputs` instead of assuming every run has the same directory tree.

An earlier result was overwritten
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a unique ``output_dir`` for each run and each batch item. Existing
directories are not protected by the CLI. Do not use one shared directory for
several Moving images.

napari and WSI
--------------

The plugin is missing from the Plugins menu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install ``./napari-cell-registration`` in the same environment used to
launch napari, then restart napari. Check the terminal for import errors. The
menu uses the display name **Cell Registration**, not only the TopoAlign brand.

OpenSlide cannot import or reports a missing DLL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``[wsi]`` extra adds the OpenSlide Python binding. A compatible native
OpenSlide library must also be available. Test
``python -c "import openslide; print(openslide.__version__)"`` in the napari
environment and follow the platform instructions linked in :doc:`installation`.

CellViT does not start
~~~~~~~~~~~~~~~~~~~~~~

CellViT++ requires a separate installation, its dependencies and weights,
a working CUDA environment, and a supported resolution setting. Installing
TopoAlign or the ``[wsi]`` extra does not provision that full environment.
Follow the prerequisites in :doc:`napari_guide`.

WSI masks cannot be registered after reloading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The WSI registration path needs paired WSI centroid metadata on the napari
layers. **Save Mask Layers** saves TIFF labels but not that metadata. Continue
from WSI segmentation to registration in the same napari session, or preserve
and restore the required metadata separately.

Saving a Normal registration fails after writing some files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current rigid fallback can fail during export when TPS control points
were not created. Avoid treating a partly written directory as a complete
result. For an explicitly rigid, reproducible export, use the CLI, or disable
the plugin's result saving and inspect/export the resulting layers manually.
The plugin's output semantics are detailed in :doc:`napari_guide`.

Agent and missing Web instructions
----------------------------------

The Agent cannot connect
~~~~~~~~~~~~~~~~~~~~~~~~

Check the endpoint, model identifier, and credentials together. A listed
model may not be enabled for a particular account. Use the settings and
connection workflow in :doc:`agent_guide`. Regular local CLI and napari work
do not require an API key.

The webapp directory is missing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is not included in the public source revision covered here. Use CLI or
napari; see :doc:`web_guide` for the replacement of older Web instructions.

Report a reproducible issue
---------------------------

When opening a `GitHub issue
<https://github.com/DNale-11/TopoAlign/issues>`_, include:

* Operating system, Python version, ``topoalign --version``, and
  ``git rev-parse HEAD``.
* The interface used and the smallest command or sequence that reproduces it.
* Input shapes, dtypes, channel layout, and whether inputs are images,
  instance masks, or feature tables.
* The error traceback and a redacted resolved registration configuration.
* Relevant match counts, residual summaries, and a small shareable example
  when available.

Remove API keys, private paths, and sensitive image content before posting.
