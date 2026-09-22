Installation
============

Choose an environment
---------------------

Use Python 3.10 for the analysis environments below. Windows and Linux commands
are shown as single lines that work in PowerShell or a typical Unix shell.
Run commands from the repository root unless stated otherwise.

Choose the smallest installation that supports your workflow:

* **Existing masks or feature tables:** install the core CLI.
* **Segment raw images with Cellpose-SAM:** add the image dependencies and
  choose one PyTorch build, CPU or CUDA.
* **Interactive work:** add the napari plugin.
* **Conversational assistance:** add the optional Agent client.
* **Build this manual only:** follow `docs/README.md
  <https://github.com/DNale-11/TopoAlign/blob/main/docs/README.md>`_ in a
  separate Python 3.11 environment.

An API key is needed only for Agent conversations. Local registration does not
require an API account.

Get the source
--------------

.. code-block:: console

   git clone https://github.com/DNale-11/TopoAlign.git
   cd TopoAlign

The source revision covered by this manual is ``693c6781e162``. To reproduce
that exact revision in a fresh clone, run ``git checkout 693c6781e162``.
The new example script in this manual is not present in that older revision;
download it separately from :doc:`quickstart` when using the pinned checkout.
Otherwise, check the documentation against your current ``git rev-parse HEAD``.

.. _install-core:

Core CLI for masks and features
-------------------------------

.. code-block:: console

   conda create -n topoalign_core python=3.10 -y
   conda activate topoalign_core
   python -m pip install --upgrade pip
   python -m pip install -e . "matplotlib>=3.7,<3.10"
   python -m pip check

Matplotlib is included explicitly because the package imports visualization
utilities at startup, but the released root package metadata does not declare
it. This installation can extract features, match cells, estimate transforms,
and warp arrays. It does not provide image segmentation or napari.

Verify the CLI:

.. code-block:: console

   topoalign --version
   topoalign --help

The documented package reports version ``0.1.0``. Continue with the synthetic
mask example in :doc:`quickstart`.

.. _install-images:

Image segmentation environment
------------------------------

The published requirements constrain PyTorch to ``>=2.2,<2.5``. The commands
below use PyTorch 2.4.1 with its matching torchvision 0.19.1 build and pin
Cellpose 4.0.8 within the project's supported Cellpose 4 range.

.. important::

   Do not combine the older README's PyTorch 2.11.0 command with the published
   requirements file. It conflicts with the ``torch<2.5`` constraint.
   Choose either the CPU or GPU recipe in a clean environment.

CPU
~~~

.. code-block:: console

   conda create -n topoalign_cpu python=3.10 -y
   conda activate topoalign_cpu
   python -m pip install --upgrade pip
   python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt "torch==2.4.1" "torchvision==0.19.1" "cellpose==4.0.8"
   python -m pip install -e .
   python -m pip check

Use ``--no-gpu`` for CLI segmentation and registration. CPU inference is
appropriate for checking a small image; large images may take considerably
longer.

GPU with NVIDIA CUDA
~~~~~~~~~~~~~~~~~~~~

Confirm that the computer has an NVIDIA GPU and a compatible driver. This
example uses the official CUDA 12.1 wheel; choose a compatible build of the
same PyTorch/torchvision pair from the `PyTorch previous-versions instructions
<https://pytorch.org/get-started/previous-versions/>`_ if needed.

.. code-block:: console

   conda create -n topoalign_gpu python=3.10 -y
   conda activate topoalign_gpu
   python -m pip install --upgrade pip
   python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
   python -m pip install -r requirements.txt "torch==2.4.1" "torchvision==0.19.1" "cellpose==4.0.8"
   python -m pip install -e .
   python -m pip check

Check the selected runtime before enabling GPU:

.. code-block:: console

   python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"

For the GPU recipe, ``CUDA available`` must be ``True``. For a CPU wheel,
``False`` is expected. GPU acceleration mainly concerns segmentation and
GPU-enabled plugin operations; file I/O and parts of the registration pipeline
still run on the CPU.

Check Cellpose and run a small image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   python -c "from cellpose import models; print('Cellpose import OK')"
   topoalign segment --image small_image.tif --channel-axis none --no-gpu --output-dir outputs/segmentation-check

Replace ``small_image.tif`` with your own 2D grayscale image. In a verified
CUDA environment, use ``--gpu`` instead. The first Cellpose-SAM invocation may
download model weights; an import check alone does not verify model download,
inference, or available GPU memory.

The root ``requirements.txt`` includes the napari application as well as the
image-processing packages. The plugin itself still requires the installation
step below.

Install the napari plugin
-------------------------

In the selected image environment:

.. code-block:: console

   python -m pip install -e ./napari-cell-registration
   python -m pip check
   napari

In napari, open **Plugins** and choose **Cell Registration**. See
:doc:`napari_guide` for the widget names and workflow.

Optional WSI support
~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

   python -m pip install -e "./napari-cell-registration[wsi]"
   python -c "import openslide; print(openslide.__version__)"

The ``[wsi]`` extra installs the OpenSlide Python binding. A native OpenSlide
library must also be available on your platform; follow the
`OpenSlide Python installation instructions
<https://openslide.org/api/python/>`_.

This extra does not install CellViT++, ``pathopatch``, or model weights.
The CellViT++ backend has additional prerequisites in :doc:`napari_guide`.
Ordinary 2D registration does not require these WSI components.

Install the optional Agent
--------------------------

In an existing working TopoAlign environment:

.. code-block:: console

   python -m pip install -e ".[agent]"
   topoalign agent --help

This adds the OpenAI-compatible API client. Configure the endpoint, model, and
credentials as described in :doc:`agent_guide`. Agent assistance with raw-image
segmentation also needs the image dependencies above.

Record the environment
----------------------

After a successful installation, record the package set and source revision
with your analysis:

.. code-block:: console

   python -m pip freeze > environment-installed.txt
   git rev-parse HEAD

These commands record the current state; they do not prove that a new machine
can run the same GPU workload. Run the mask example and a representative
segmentation check before starting a large analysis.

If a dependency conflict or import error occurs, see :doc:`troubleshooting`
before changing versions in an existing working environment.
