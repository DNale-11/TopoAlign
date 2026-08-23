参数说明与调优
==============

调参原则
--------

#. 先确认 Fixed/Moving 方向和分割通道。
#. 先用 ``rigid`` 建立基线。
#. 一次只改变一组参数，并使用新的输出目录。
#. 匹配数量、残差和叠加图必须一起判断。
#. 只有存在明确倍率差或剪切证据时，才提高变换自由度。

CLI 默认值
-----------

.. list-table:: 分割与输入
   :header-rows: 1
   :widths: 30 20 50

   * - 参数
     - 默认值
     - 说明
   * - ``channel_axis``
     - ``auto``
     - 自动判断通道轴；可显式设为 ``first``、``last`` 或 ``none``
   * - ``registration_channel``
     - ``-1``
     - 用于分割的通道索引；``-1`` 为最后一个通道
   * - ``gpu``
     - ``false``
     - 是否使用 GPU 运行 Cellpose-SAM
   * - ``min_area`` / ``max_area``
     - ``null``
     - 特征提取时过滤过小或过大的实例

.. list-table:: 匹配
   :header-rows: 1
   :widths: 30 20 50

   * - 参数
     - 默认值
     - 说明
   * - ``top_k``
     - ``50``
     - 最终保留的代表性高质量匹配数量上限
   * - ``feature_weight``
     - ``1.0``
     - 形态/强度等细胞特征在匹配代价中的权重
   * - ``topology_weight``
     - ``0.35``
     - 局部邻域拓扑一致性的权重
   * - ``position_weight``
     - ``4.0``
     - 粗对齐后空间位置一致性的权重
   * - ``distance_threshold``
     - ``2.0``
     - 候选匹配距离阈值
   * - ``spatial_window_size``
     - ``100.0`` px
     - 粗对齐后搜索对应细胞的空间窗口
   * - ``use_spatial_clusters``
     - ``false``
     - 是否按空间簇组织候选
   * - ``n_clusters``
     - ``9``
     - 启用空间簇后的簇数
   * - ``use_topology_filtering``
     - ``false``
     - 是否启用额外拓扑一致性筛选

高级拓扑参数 ``k_pos_nei=5``、``k_neighbor=5``、``tau_pos=0.5``、
``tau_nei=0.3``、``tau_map=0.3`` 只在相应筛选启用后发挥作用。建议先保留默认值。

.. list-table:: 变换
   :header-rows: 1
   :widths: 30 20 50

   * - 参数
     - 默认值
     - 说明
   * - ``method``
     - ``rigid``
     - ``rigid``、``similarity`` 或 ``affine``
   * - ``use_ransac``
     - ``false``
     - 使用 RANSAC 抑制离群匹配；当前只对 ``rigid`` 生效
   * - ``ransac_max_trials``
     - ``1000``
     - RANSAC 最大尝试次数
   * - ``ransac_residual_threshold``
     - ``2.0`` px
     - RANSAC 内点残差阈值
   * - ``residual_prune_quantile``
     - ``null``
     - 首次拟合后按残差分位数剔除高残差匹配并重新拟合
   * - ``initial_coarse_transform``
     - ``null``
     - 可选的 3×3 初始 moving-to-fixed 变换

常见调优路径
------------

分割明显错误
~~~~~~~~~~~~

* 先确认 ``registration_channel`` 和 ``channel_axis``。
* 检查背景是否被分成大量小实例，必要时设置 ``min_area``。
* 如果粘连或漏分严重，优先在 napari 中修订 mask，再用 mask 模式配准。
* GPU 开关只影响执行设备，不会自动改善分割质量。

匹配太少
~~~~~~~~

* 确认两幅图确实有足够视野重叠。
* 检查两侧分割出的细胞规模是否相近。
* 在已有候选充足时适度提高 ``top_k``。
* 粗位移较大时可扩大 ``spatial_window_size``；扩大后要更严格检查离群点。
* 跨模态形态差异大时，可降低 ``feature_weight`` 或提高 ``topology_weight``，每次小幅调整。

匹配多但离群明显
~~~~~~~~~~~~~~~~

* 刚性模型可启用 ``--use-ransac``。
* 根据预期定位误差调整 ``ransac_residual_threshold``。
* 可设置 ``residual_prune_quantile`` 做二次残差裁剪。
* 不要只追求更低平均残差；要检查匹配是否覆盖整个视野。

倍率不同
~~~~~~~~

先用刚性模型确认旋转和平移方向正确，再改为 ``similarity``。如果拟合出的缩放比例
不符合显微镜元数据，应回到输入像素尺寸和匹配质量排查。

考虑仿射
~~~~~~~~

只有当叠加图显示稳定的方向性缩放或剪切，并且匹配点数量足够、覆盖整个视野时，
才使用 ``affine``。仿射至少需要 3 对匹配点，但实际可靠拟合通常需要远多于最低数量。

质量阈值的使用
--------------

Agent/诊断中的阈值是筛查提示而非硬性验收标准。经验上，少于 3 对匹配无法支持可靠
模型；P95 残差大于约 5 px 应人工复核，大于约 10 px 通常提示明显问题；有效重叠比例
低于约 0.25 也需要复核。像素大小、细胞直径和实验目标不同，阈值应随数据校准。
