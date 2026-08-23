TopoAlign 用户手册
====================

.. image:: ../topoalign-logo.png
   :alt: TopoAlign
   :width: 360px
   :align: center

TopoAlign 是一个面向显微与空间组学图像的拓扑引导细胞配准工具。
它以细胞为配准地标，完成实例分割、特征提取、细胞匹配、几何变换估计、
图像重采样和结果质检，并提供 Web、CLI、napari 与可选 AI Agent 四种入口。

.. important::

   ``Fixed`` 是参考图像和最终坐标空间，``Moving`` 是待变换的源图像。
   TopoAlign 输出的变换方向始终是 ``moving_to_fixed``。

首次使用建议先阅读 :doc:`installation` 和 :doc:`quickstart`；需要可视化操作时阅读
:doc:`web_guide`，需要可复现脚本时阅读 :doc:`cli_guide`。

.. toctree::
   :maxdepth: 2
   :caption: 开始使用

   overview
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: 操作指南

   web_guide
   cli_guide
   napari_guide
   agent_guide

.. toctree::
   :maxdepth: 2
   :caption: 参考

   parameters
   outputs
   troubleshooting

项目源码与问题反馈：`DNale-11/cell_registration <https://github.com/DNale-11/cell_registration>`_。
