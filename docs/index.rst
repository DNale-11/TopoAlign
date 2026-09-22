TopoAlign User Manual
=====================

.. image:: _static/topoalign-release-logo.png
   :alt: TopoAlign
   :width: 360px
   :align: center

TopoAlign registers cellular microscopy images using cell morphology and local
spatial relationships. This manual explains how to install the software, choose
inputs, run registration, and assess the result before using it in an analysis.

The public repository provides a command-line interface, a napari plugin, and
an optional conversational Agent. Start with :doc:`installation` and the
reproducible mask example in :doc:`quickstart`. For interactive work, use
the :doc:`napari_guide`.

.. important::

   **Fixed** is the reference image and defines the output coordinate system.
   **Moving** is the source image to transform. CLI transform files map
   ``moving_to_fixed`` in pixel coordinates.

Choose a starting point
-----------------------

* **Check the installation without a GPU or model download:** :doc:`quickstart`.
* **Register images, masks, or feature tables:** :doc:`cli_guide`.
* **Review and edit segmentation interactively:** :doc:`napari_guide`.
* **Get assistance through an API provider:** :doc:`agent_guide`.
* **Understand parameters and assess alignment:** :doc:`parameters` and
  :doc:`outputs`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   overview
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guides

   cli_guide
   napari_guide
   agent_guide

.. toctree::
   :maxdepth: 2
   :caption: Reference

   parameters
   outputs
   troubleshooting

Version and support
-------------------

This edition covers the public TopoAlign 0.1.0 source at commit
``693c6781e162``. The CLI and napari plugin have different pipelines and export
conventions; their guides describe those differences explicitly. A Web server is
not included in this source revision; see :doc:`web_guide` for the status of
older Web instructions.

Source code is available at `DNale-11/TopoAlign
<https://github.com/DNale-11/TopoAlign>`_. Report reproducible problems through
`GitHub Issues <https://github.com/DNale-11/TopoAlign/issues>`_, using the
information checklist in :doc:`troubleshooting`.
