Command-line guide
==================

The ``topoalign`` command registers a Moving image into the pixel coordinates of
a Fixed reference. The local registration commands do not require an API key.
Run ``topoalign --help`` or ``topoalign COMMAND --help`` for accepted arguments;
running ``topoalign`` without arguments opens the interactive shell.

.. list-table:: Commands
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Purpose
   * - ``run``
     - Run segmentation, feature extraction, matching, transform estimation, and resampling as needed.
   * - ``segment``
     - Segment one image with Cellpose-SAM.
   * - ``features``
     - Extract cell features from an instance label mask.
   * - ``match``
     - Match two feature tables or masks.
   * - ``transform``
     - Fit a Moving-to-Fixed transform to an existing match table.
   * - ``warp``
     - Apply a saved transform to an image or label mask.
   * - ``inspect``
     - Print information about an input file, configuration, or result.
   * - ``agent``
     - Start the optional conversational interface; see :doc:`agent_guide`.

Examples below use one-line commands that work in PowerShell and most Unix
shells. Replace the input filenames and image dimensions with your own. Quote
paths containing spaces, and use a separate output directory for each run.

Choose your inputs
------------------

For each side of a ``run``, provide an image, an instance mask, or a feature CSV.
Inputs can be combined: images supply the pixels to resample and display, masks
avoid segmentation, and features avoid feature extraction. The precedence is:

#. A supplied feature CSV supplies that side's features.
#. Otherwise, features are extracted from the supplied mask.
#. Otherwise, the supplied image is segmented first.
#. ``--matches`` skips automatic matching, but still requires features on both
   sides, either supplied or computed.

``--mode auto|image|mask|features`` records the intended workflow; the paths you
supply actually determine which stages run. Changing ``--mode`` does not clear
paths inherited from a configuration file.

Images and channels
~~~~~~~~~~~~~~~~~~~

TIFF, PNG, and JPEG are supported image formats. Use TIFF for label masks and
scientific outputs. Registration estimates a **2D** transform in pixel units;
it does not reconcile physical pixel sizes or image origins from metadata.

* A grayscale ``Y, X`` image works with ``--channel-axis auto`` or ``none``.
* For ``C, Y, X`` data, use ``--channel-axis first``.
* For ``Y, X, C`` data with up to four channels, use ``--channel-axis last`` or
  ``auto``. In ``auto`` mode, a three-dimensional array whose last dimension
  exceeds four is interpreted as a ``Z, Y, X`` stack.
* ``--registration-channel`` uses a zero-based index; ``-1`` selects the last
  channel. It selects the segmentation channel for explicit ``first``/``last``
  layouts, and for ``auto`` channel-last images. The same settings apply to both
  Fixed and Moving. Resampling retains all channels.

For an automatically detected Z-stack, segmentation uses Cellpose's 3D path and
exports a 2D maximum-label projection. A ``Z, Y, X`` stack can then be warped
plane by plane with the same 2D transform. This is not volumetric registration.
The auto Z-stack path uses the last channel of ``Z, Y, X, C`` data regardless of
``--registration-channel``.

.. warning::

   The released full pipeline cannot warp four-dimensional ``Z, Y, X, C`` data.
   Channel-last data with more than four channels also fail spatial-shape
   inference during export, even with ``--channel-axis last``. Prepare a 2D
   image or a ``C, Y, X`` array before running the full pipeline. The standalone
   ``segment`` command can still segment a ``Z, Y, X, C`` input.

Masks and coordinate tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

An instance mask is an integer label image: ``0`` is background, and every cell
has its own positive label. A binary foreground mask describes a single label,
not separately numbered cells. Negative labels are rejected. Supply integer
labels; the loader casts values to ``int32`` without validating fractional
values. Three-dimensional masks are reduced by taking the maximum label along
axis 0, so overlapping labels along Z can disappear in the projection.

Masks must describe the same pixel coordinates as their corresponding images.
Crop, resize, and orient paired inputs consistently before registration.

Feature CSVs use ``centroid_x`` for the column coordinate and ``centroid_y`` for
the row coordinate, measured from the top-left image origin in pixels. The
loader requires both columns and adds sequential ``cell_id`` values if that
column is absent. However, **automatic matching needs the complete feature
schema**, not just coordinates. Generate it with ``topoalign features``; the
column reference is in :doc:`outputs`.

A manually prepared match CSV must contain ``idx1`` and ``idx2``: zero-based row
positions in the Fixed and Moving feature CSVs, respectively. These are not
``cell_id`` values. Keep the feature row order unchanged after making matches.
A coordinate-only feature table is sufficient for ``transform`` or for ``run``
with supplied matches. Use at least two distinct pairs for rigid/similarity
fitting, or three non-collinear pairs for affine fitting.

Register an image pair
----------------------

.. code-block:: console

   topoalign run --fixed fixed.tif --moving moving.tif --mode image --channel-axis auto --registration-channel -1 --method rigid --no-gpu --output-dir outputs/image-pair

Use ``--gpu`` when a suitable GPU and PyTorch installation are available. This
switch controls Cellpose segmentation; it does not move the entire registration
pipeline to the GPU. Review the outputs as described in :doc:`outputs`.

Reuse existing masks
--------------------

Provide images alongside masks to skip segmentation and still export registered
intensity data:

.. code-block:: console

   topoalign run --fixed fixed.tif --moving moving.tif --fixed-mask fixed_mask.tif --moving-mask moving_mask.tif --mode mask --method rigid --output-dir outputs/from-masks

For a mask-only run, omit ``--fixed`` and ``--moving``:

.. code-block:: console

   topoalign run --fixed-mask fixed_mask.tif --moving-mask moving_mask.tif --mode mask --output-dir outputs/masks-only

In this case, ``registered_moving.tif`` contains the resampled Moving mask, with
nearest-neighbor interpolation. If a Moving image is supplied, that image is
the resampling source; ``run`` does not also export a separate registered mask.

Run individual stages
---------------------

The following example assumes a Fixed image of height 2048 and width 2048:

.. code-block:: console

   topoalign segment --image fixed.tif --output-dir work/fixed-seg --no-gpu
   topoalign segment --image moving.tif --output-dir work/moving-seg --no-gpu
   topoalign features --mask work/fixed-seg/mask.tif --output-dir work/fixed-features
   topoalign features --mask work/moving-seg/mask.tif --output-dir work/moving-features
   topoalign match --fixed-features work/fixed-features/features.csv --moving-features work/moving-features/features.csv --fixed-shape 2048 2048 --output-dir work/matches
   topoalign transform --fixed-features work/fixed-features/features.csv --moving-features work/moving-features/features.csv --matches work/matches/matches.csv --method rigid --output-dir work/transform
   topoalign warp --moving moving.tif --transform work/transform/transform.moving_to_fixed.json --fixed-shape 2048 2048 --output-dir work/warp

``--fixed-shape`` is always ``HEIGHT WIDTH``. Supply the actual reference size;
feature centroids cannot recover the empty border around the image. ``match``
can alternatively take ``--fixed-mask`` and ``--moving-mask`` and compute
features internally. If both a mask and features are supplied for one side,
features take precedence; a Fixed mask can still provide the reference shape.

For an additional mask export, apply the same transform separately:

.. code-block:: console

   topoalign warp --moving moving_mask.tif --transform work/transform/transform.moving_to_fixed.json --fixed-shape 2048 2048 --mask --output-dir work/warped-mask

``--mask`` selects nearest-neighbor interpolation. Intensity data use bilinear
interpolation. Pixels outside the Moving field of view are filled with zero.
Both cases use the filename ``registered_moving.tif``.

.. note::

   Standalone stages use their built-in defaults and explicit arguments, not
   the ``registration`` settings in a project configuration. In the release,
   ``match`` accepts transform-related flags in its help but does not apply
   them. Set the final transform method on ``transform``; use ``run`` to apply
   an initial coarse transform. See :doc:`parameters` for the full reference.

Configuration files
-------------------

There are two JSON formats. Keep their roles separate:

**Application settings** contain ``version``, ``cli``, ``agent``, and
``registration`` sections. Without ``run --config``, settings are merged in
this order, with later values overriding earlier ones:

#. Built-in defaults.
#. ``~/.topoalign/config.json`` in the user's home directory, if present.
#. ``topoalign.config.json`` in the current working directory, if present.
#. Explicit ``run`` command-line arguments.

The repository's ``topoalign.config.example.json`` is an application-settings
example. Copy it to ``topoalign.config.json``, edit its ``registration`` section,
then invoke ``topoalign run`` from that directory.

**Registration-only configuration** is passed to ``topoalign run --config FILE``.
It contains the contents of the ``registration`` section directly, without the
application-settings wrapper. Omitted values receive built-in defaults; this
file replaces the inherited registration settings. Explicit command-line
arguments still override it.

For example, save this as ``registration.json``:

.. code-block:: json

   {
     "mode": "mask",
     "fixed": "fixed.tif",
     "moving": "moving.tif",
     "fixed_mask": "fixed_mask.tif",
     "moving_mask": "moving_mask.tif",
     "transform": {
       "method": "rigid",
       "use_ransac": true,
       "ransac_residual_threshold": 2.0
     },
     "output": {
       "output_dir": "outputs/config-run"
     }
   }

.. code-block:: console

   topoalign run --config registration.json

Paths inside either format are relative to the **current working directory**,
not to the JSON file. Unknown registration keys are rejected. Startup still
loads application settings, so malformed application JSON can prevent even an
explicit ``--config`` command from starting.

For feature-only runs, ``fixed_shape`` is a JSON-only ``[height, width]`` option
on ``run``; there is no ``run --fixed-shape`` flag. A supplied Fixed mask takes
precedence over this value. Without either, the reference size is inferred from
the maximum centroid coordinates across both tables, which can omit borders.
A Fixed image path alone does not override this inference. If you provide a
Moving resampling source but no Fixed image or mask, set
``output.save_overlay`` to ``false``.

Inspect and repeat runs
-----------------------

.. code-block:: console

   topoalign inspect outputs/config-run/result.json
   topoalign inspect outputs/config-run/diagnostics.json --json
   topoalign inspect fixed.tif

``inspect`` reports JSON contents, CSV column names and a preview row count, or
image shape and dtype. It does not validate registration quality.

``run``, ``match``, ``transform``, ``warp``, and ``inspect`` accept ``--json``.
The release also prints a JSON summary without this flag. ``segment`` and
``features`` do not accept it. The flag does not disable file exports or promise
that processing libraries will produce no other terminal output.

For one Fixed image and many Moving images, invoke ``run`` once per pair. Reuse
the Fixed mask/features and choose unique output directories. Existing files
with the same names are overwritten, and older unrelated files are not removed.
Preserve ``config.resolved.json``, the input tables used to create matches, and
the terminal log. A successful local command returns exit code ``0``; handled
input/configuration errors return ``2``. Treat every nonzero exit as a failure
and check for a newly completed ``result.json`` before using an output.

Batch example with existing masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following Python loop works on Windows and Linux. Place one Fixed mask at
``data/fixed_mask.tif`` and the Moving masks in ``data/moving_masks/``. Save the
snippet as ``batch_masks.py`` in your working directory, then run
``python batch_masks.py`` in the TopoAlign environment.

.. code-block:: python

   from pathlib import Path
   import subprocess
   import sys

   fixed = Path("data/fixed_mask.tif")
   moving_masks = sorted(Path("data/moving_masks").glob("*.tif"))
   if not fixed.is_file() or not moving_masks:
       raise SystemExit("Check the Fixed path and Moving mask directory.")

   for moving in moving_masks:
       output = Path("outputs/batch") / moving.stem
       if output.exists():
           raise FileExistsError(f"Choose a new batch directory: {output}")
       subprocess.run([
           sys.executable, "-m", "cell_registration.cli", "run",
           "--fixed-mask", str(fixed), "--moving-mask", str(moving),
           "--mode", "mask", "--method", "rigid",
           "--output-dir", str(output),
       ], check=True)

This loop runs sequentially and stops on the first failure. Each Moving file
gets its own output directory, and existing directories are refused. It reuses
masks to avoid segmentation, but extracts features again for each pair. For
large batches, export and reuse the Fixed feature table as described above.
Review every pair's ``result.json`` and overlay; success for one pair does not
establish the quality of the rest of the batch.
