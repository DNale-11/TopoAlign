输出文件与质量判读
==================

建议检查顺序
------------

#. 确认 ``result.json`` 存在且任务没有记录错误。
#. 检查 Fixed/Moving 分割是否合理。
#. 检查匹配数量和匹配点是否覆盖整个有效视野。
#. 查看 ``overlay.tif`` 是否在组织边缘与细胞结构上同时对齐。
#. 查看平均、P95/最大残差，并寻找局部系统性偏差。
#. 确认有效重叠区域足以支持后续分析。

CLI 完整流程与 Web Normal
-------------------------

默认完整输出通常为：

.. code-block:: text

   <output_dir>/
   ├── segmentation_fixed/
   │   ├── mask.tif
   │   └── segmentation.json
   ├── segmentation_moving/
   │   ├── mask.tif
   │   └── segmentation.json
   ├── config.resolved.json
   ├── fixed_features.csv
   ├── moving_features.csv
   ├── matches.csv
   ├── transform.moving_to_fixed.json
   ├── registered_moving.tif
   ├── valid_overlap_mask.tif
   ├── overlay.tif
   ├── diagnostics.json
   └── result.json

提供已有 mask/features、关闭保存选项或没有可重采样的 Moving 图像时，部分文件不会生成。
``result.json`` 是完成清单；运行异常时可能已经留下部分中间文件，但还没有最终清单。

.. list-table:: 主要文件
   :header-rows: 1
   :widths: 34 66

   * - 文件
     - 含义
   * - ``config.resolved.json``
     - 本次运行实际使用的完整 registration 配置
   * - ``fixed_features.csv`` / ``moving_features.csv``
     - 细胞质心、形态、归一化位置和拓扑特征
   * - ``matches.csv``
     - 一对一匹配及最终 ``residual_px``
   * - ``transform.moving_to_fixed.json``
     - 3×3 像素坐标变换矩阵及方向说明
   * - ``registered_moving.tif``
     - Moving 重采样到 Fixed 尺寸后的图像或 mask
   * - ``valid_overlap_mask.tif``
     - Moving 有效像素域经过几何变换后在 Fixed 中的覆盖
   * - ``overlay.tif``
     - Fixed 为洋红、registered Moving 为绿色；共同高亮处接近白色
   * - ``diagnostics.json``
     - 匹配数、残差、耗时、警告与错误
   * - ``result.json``
     - 任务状态、阶段耗时、诊断和 artifact 路径的结构化清单

.. note::

   ``valid_overlap_mask.tif`` 表示几何有效覆盖，不是细胞重叠率，也不是配准正确率。

CLI 分阶段输出
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 命令
     - 固定输出
   * - ``segment``
     - ``mask.tif``、``segmentation.json``
   * - ``features``
     - ``features.csv``、``features.json``
   * - ``match``
     - ``matches.csv``、``matching_diagnostics.json``
   * - ``transform``
     - ``transform.moving_to_fixed.json``、``matches.with_residuals.csv``
   * - ``warp``
     - ``registered_moving.tif``、``valid_overlap_mask.tif``
   * - ``inspect``
     - 只打印检查结果，不写新的分析文件

``--json`` 只改变标准输出格式，不会阻止上述文件写入。即使 ``warp --mask``，输出文件名
仍为 ``registered_moving.tif``。

Web 输出差异
------------

Web Normal 与 CLI 完整流程使用同一注册服务，因此文件结构基本相同；任务目录位于
``data/tasks/<task_id>``。顶部 **Download** 只下载当前活动结果的
``registered_moving.tif``，不会打包整个目录或整个批次。

Web WSI 当前只写：

.. code-block:: text

   segmentation_fixed/
   segmentation_moving/
   registered_moving.tif
   registered_mask.tif
   overlay.tif
   matches.csv
   result.json

它不会生成 ``config.resolved.json``、``diagnostics.json``、有效重叠 mask、特征表或
可下载的 transform JSON。输出是普通 TIFF，而不是保留金字塔与 slide 元数据的 WSI。

napari 输出差异
---------------

napari 所有保存选项默认关闭；默认只有图层和通知消息。

Normal（TPS 成功路径）启用 ``save_results`` 后通常写出：

* ``registered_image.tif``、``registered_mask.tif``
* ``features_round1.csv``、``features_round2.csv``、``matches.csv``
* ``registered_features_round2.csv``、``registered_centroids_round2.csv``
* ``transform_info.txt``

.. warning::

   对同形 2D 图像，napari Normal 的 ``registered_image.tif`` 是 Fixed 与已配准
   Moving 的逐像素最大值融合图；``registered_mask.tif`` 也会用 Fixed mask 填补
   Moving 变换后的零区域。它们不是 CLI/Web 中的纯 registered Moving。

FISH 模式保存纯 Moving 结果，文件名为 ``registered_image_fish.tif``、
``registered_mask_fish_pure_moving.tif``、``matches_fish_topology.csv`` 和
``transform_fish.txt``。

napari WSI 启用保存后写 level-0 质心表、``wsi_centroid_matches.csv``、
``local_deformation_grid_he_level0.csv``、``wsi_transform.json``、标准化元数据 JSON
和 ``centroid_registration_preview.tif``。它不自动写完整的多通道 registered WSI。

数值解释
--------

匹配数量
   太少时模型不稳定；数量很多也不代表正确。优先检查是否覆盖视野四周和组织主要区域。

``residual_px``
   匹配点经过拟合后在 Fixed 像素坐标中的欧氏误差。低残差可能来自少量或局部聚集的
   匹配，因此必须结合空间覆盖。

平均与最大/P95
   平均值描述总体拟合，P95 或最大值帮助发现离群和局部错配。阈值需要按像素尺寸与
   细胞直径校准。

Overlay
   观察全局旋转/平移、组织边缘、局部细胞结构和空白边界。不要只看最亮的一小块。

数据类型
--------

CLI 完整流程和 Web Normal 读取强度图时通常转换为 ``float32``，因此
``registered_moving.tif`` 往往也是 float32。独立 ``topoalign warp`` 和 napari
常规/FISH 路径会尽量转换回输入 dtype。下游软件若依赖位深，应在导入前检查 dtype。
