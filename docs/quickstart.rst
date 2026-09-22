Quick start
===========

This tutorial first checks registration using small synthetic instance masks.
It requires only the :ref:`core CLI installation <install-core>`: no GPU,
segmentation model, API account, or microscopy dataset is needed. The second
part shows how to run your own image pair.

Run a reproducible mask example
-------------------------------

1. Create the example masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, run:

.. code-block:: console

   python docs/examples/create_demo_masks.py

The script writes ``demo/fixed_mask.tif`` and ``demo/moving_mask.tif``.
Each is a 256 by 256 integer label image with 12 distinct elliptical objects.
The Moving objects are shifted six pixels right and four pixels up relative
to Fixed. The script refuses to overwrite existing example files.

For readers using the HTML manual, the script is available as
:download:`create_demo_masks.py <examples/create_demo_masks.py>`.
Save it locally and run ``python create_demo_masks.py`` from a working
directory of your choice.

These shapes are synthetic installation data, not a biological benchmark.

2. Register the masks
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   topoalign run --fixed-mask demo/fixed_mask.tif --moving-mask demo/moving_mask.tif --mode mask --method rigid --output-dir outputs/demo-mask

Use a fresh output directory for each rerun, such as
``outputs/demo-mask-002``. The CLI can overwrite existing files with the same
names.

The command extracts features, matches the objects, estimates a rigid
Moving-to-Fixed transform, and resamples the Moving labels. It does not invoke
Cellpose.

3. Inspect the result
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   topoalign inspect outputs/demo-mask/result.json

The verified example produces 12 matches. The transform in
``transform.moving_to_fixed.json`` is approximately:

.. code-block:: json

   {
     "direction": "moving_to_fixed",
     "coordinate_space": "pixel_xy",
     "matrix": [
       [1.0, 0.0, -6.0],
       [0.0, 1.0, 4.0],
       [0.0, 0.0, 1.0]
     ]
   }

Only selected transform fields are shown. Small floating-point differences
around zero and one are expected.

The translation is ``x_fixed = x_moving - 6``,
``y_fixed = y_moving + 4``. Residuals should be close to zero, and
``registered_moving.tif`` should coincide with the Fixed labels. In a mask-only
run, this file contains registered labels rather than intensity data.

Open Fixed and the registered mask as **Labels** layers in napari to compare
their outlines. The generated ``overlay.tif`` compares the label values for
this mask-only example; it is not an intensity image. See :doc:`outputs` for
conditional output files and display conventions.

Register your own images
------------------------

Install the :ref:`image segmentation dependencies <install-images>` first.
Replace the sample filenames below with your data paths; the repository does
not include a microscopy image pair.

1. Inspect both inputs
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   topoalign inspect fixed.tif
   topoalign inspect moving.tif

Choose images with overlapping tissue and a comparable cell or nuclear
channel. Begin with a representative region that fits comfortably in memory.

For a 2D grayscale pair, the following CPU example is explicit about the
channel layout:

.. code-block:: console

   topoalign run --fixed fixed.tif --moving moving.tif --mode image --channel-axis none --method rigid --no-gpu --output-dir outputs/first-image-run

For a ``C,Y,X`` array, replace ``--channel-axis none`` with
``--channel-axis first --registration-channel 0``, adjusting the channel
index to your data. Both images use the same channel-axis and channel-index
settings in a single run. If their layouts or nuclear-channel positions differ,
prepare compatible images or segment them separately.

For a verified CUDA environment, replace ``--no-gpu`` with ``--gpu``.
The first Cellpose-SAM run may download model weights. CPU segmentation can
be slow; reuse saved masks when trying different matching parameters.

2. Check segmentation before tuning matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open ``segmentation_fixed/mask.tif`` and ``segmentation_moving/mask.tif`` in
the output directory. Check that cells are distinct positive labels and that the
tissue regions of interest are represented on both sides. If segmentation is
poor, fix the channel selection or supply corrected instance masks before
interpreting the registration.

To use corrected masks with the raw images:

.. code-block:: console

   topoalign run --fixed fixed.tif --moving moving.tif --fixed-mask fixed_mask.tif --moving-mask moving_mask.tif --method rigid --no-gpu --output-dir outputs/corrected-mask-run

The supplied masks avoid repeated segmentation, while the raw images allow
intensity resampling and an overlay.

3. Review alignment
~~~~~~~~~~~~~~~~~~~

Inspect ``result.json`` and ``diagnostics.json``, then review
``overlay.tif``, ``registered_moving.tif``, and
``valid_overlap_mask.tif`` when present.

Check several locations across the overlapping field. Look for coherent
alignment of the same structures, well-distributed matches, and residuals
consistent with the precision needed for your analysis. A low residual from
a small cluster of matches does not establish accuracy in the rest of the
image.

Keep the input files, ``config.resolved.json``, and software revision with
the selected result. The :doc:`parameters` guide explains how to adjust a
rigid baseline, while :doc:`cli_guide` covers stage reuse and scripted batches.
