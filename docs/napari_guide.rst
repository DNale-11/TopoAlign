napari 用户指南
================

安装与启动
----------

.. code-block:: powershell

   python -m pip install -e .\napari-cell-registration
   napari

在 **Plugins → Cell Registration** 下可用以下组件：

* **Cell Segmentation (Cellpose)**：Cellpose-SAM 自动分割。
* **Save Mask Layers**：保存已经检查或修订过的 mask 图层。
* **Manual Segmentation**：手工绘制和修改实例。
* **Registration Workflow**：从图像层和 mask 层执行配准。
* **WSI Segmentation**：使用 OpenSlide 读取 WSI，并用 CellViT++ 或 Cellpose-SAM 分割。

坐标方向
--------

napari 使用 Round 编号而不是 Fixed/Moving 标签：

* **Round 1 = Fixed**：参考空间。
* **Round 2 = Moving**：被变换到 Round 1。

所有保存的变换都应按 Round 2 → Round 1 解读。

常规图像工作流
--------------

#. 把 Round 1 和 Round 2 图像加载为 napari Image layers。
#. 分别运行 **Cell Segmentation**，或载入已有 Labels/mask 图层。
#. 检查分割边界；必要时用 Labels 工具或 **Manual Segmentation** 修订。
#. 打开 **Registration Workflow**。
#. 为 ``image_round1`` / ``mask_round1`` 选择 Fixed 图层，为 Round 2 选择 Moving 图层。
#. 确认多通道图像的 ``image_channel_axis``。
#. 先保留默认参数运行，检查匹配点、连线、配准图像和 mask。
#. 需要写出文件时显式启用 ``save_results`` 并设置独立 ``output_dir``。

分割模式
--------

Cell Segmentation 的 ``mode`` 有三种：

``auto - chunk only if large``
   只有图像大于阈值时才使用分块分割。

``full image - ignore chunk settings``
   整幅图一次送入 Cellpose；适合内存足够的中小图。

``chunked - use label stitching``
   始终使用重叠分块，并在边界拼接实例标签。通常保持 ``stitch_labels`` 开启。

Registration Workflow
---------------------

常用默认值包括 ``top_k=320``、``max_match_distance_px=100``、
``position_weight=1.0``、RANSAC 开启、GPU 开启、``save_results=false``。

.. important::

   ``save_results`` 默认关闭。默认运行只在 napari 中添加图层并显示诊断，不会写出
   TIFF、CSV 或变换文件。

配准模式
~~~~~~~~

``normal``
   先建立刚性基线。过滤后控制点达到 50 对时使用 TPS 非刚性细化；不足 50 对时保持
   刚性回退。应检查 TPS 是否产生不合理局部弯曲。

``fish - contour + topology``
   面向大视野 FISH mask，使用轮廓粗配准、局部拓扑匹配和刚性 RANSAC。

``wsi``
   在 WSI level-0 坐标中进行质心配准，并可使用 KNN 局部仿射或 4×4 局部平移网格。

``auto``
   只有 mask 至少达到 ``wsi_threshold_mp``（默认 80 MP）且两个图层都包含 WSI
   元数据时才自动进入 WSI；否则走 normal。明确处理 WSI 时建议直接选择 ``wsi``。

常规模式启用保存后，输出通常包括 ``registered_image.tif``、``registered_mask.tif``、
两轮特征表、``matches.csv``、配准后 Round 2 质心和 ``transform_info.txt``。

.. warning::

   当前 normal 流程在少于 50 个最终匹配而回退到刚性、同时 ``save_results=true`` 时，
   写 ``transform_info.txt`` 可能因 TPS 控制点变量未定义而失败。遇到该情况请先关闭
   ``save_results`` 完成图层质检，再手动导出，或改用 CLI 保存刚性结果。

WSI 工作流
----------

安装 WSI 支持：

.. code-block:: powershell

   python -m pip install -e ".\napari-cell-registration[wsi]"
   python -c "import openslide; print(openslide.__version__)"

还需单独安装平台原生 OpenSlide runtime。CellViT++ 当前需要 CUDA，且其软件包、
``pathopatch``、模型与权重不包含在 ``[wsi]`` extra 中。

推荐步骤：

#. 在 **WSI Segmentation** 中选择 Fixed WSI（通常为 HE）和 Moving WSI
   （通常为 mIF）。
#. Fixed 默认使用 CellViT++，Moving 默认使用 Cellpose-SAM；根据数据调整。
#. Moving 的 QPTIFF 通道会被识别，默认尝试选择 DAPI。
#. ``read_level`` 只控制 napari 预览；``cellpose_read_level`` 控制用于分割的层级。
#. 大型 Cellpose 输入按 tile 流式处理；``max_cellpose_tiles`` 防止误触发超长任务。
#. 检查生成的 mask 与质心图层后，在 **Registration Workflow** 显式选择 ``wsi``。
#. 启用 ``save_results`` 后再执行最终运行。

WSI 变换方向记录为 ``mIF level-0 pixel → HE level-0 pixel``，局部形变网格位于
HE level-0 坐标。保存结果包括 level-0 质心表、匹配表、
``local_deformation_grid_he_level0.csv``、``wsi_transform.json`` 和质心预览。

.. note::

   Registration Workflow 的 WSI 路径只用质心估计变换，不会在该步骤自动重采样并导出
   完整多通道 WSI。``add_moving_channel_stack`` 也只用于视觉检查。
