CLI 用户指南
==============

命令概览
--------

安装后运行 ``topoalign --help`` 查看当前命令。直接运行 ``topoalign`` 会进入交互式
命令界面。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 子命令
     - 作用
   * - ``run``
     - 执行完整配准流程
   * - ``segment``
     - 图像分割为实例标签 mask
   * - ``features``
     - 从标签 mask 提取细胞特征
   * - ``match``
     - 匹配 Fixed 与 Moving 的细胞
   * - ``transform``
     - 根据匹配点估计 moving-to-fixed 变换
   * - ``warp``
     - 将图像或 mask 重采样到 Fixed 坐标
   * - ``inspect``
     - 检查配置、结果清单或输入文件
   * - ``agent``
     - 启动可选 AI Agent

完整图像配准
------------

.. code-block:: powershell

   topoalign run `
     --fixed fixed.tif `
     --moving moving.tif `
     --mode image `
     --channel-axis auto `
     --registration-channel -1 `
     --method rigid `
     --gpu `
     --output-dir outputs/sample-001

``--registration-channel -1`` 表示最后一个通道。CPU 环境把 ``--gpu`` 改为
``--no-gpu``。

.. note::

   当前 ``--mode`` 记录输入意图，但实际执行阶段由显式提供的 image、mask、features
   和 matches 路径决定。不要只靠切换 ``--mode`` 改变工作流。

复用已有 mask
-------------

已有实例标签可跳过分割。为了仍然输出配准后的强度图像，可同时提供原图：

.. code-block:: powershell

   topoalign run `
     --fixed fixed.tif `
     --moving moving.tif `
     --fixed-mask fixed_mask.tif `
     --moving-mask moving_mask.tif `
     --mode mask `
     --method rigid `
     --output-dir outputs/from-masks

只提供 mask 也可以估计变换，此时 ``registered_moving.tif`` 是重采样后的 Moving mask。

分阶段运行
----------

分阶段命令适合调试、复用中间结果或批处理。

.. code-block:: powershell

   topoalign segment --image fixed.tif --output-dir work/fixed-seg --gpu
   topoalign segment --image moving.tif --output-dir work/moving-seg --gpu

   topoalign features `
     --mask work/fixed-seg/mask.tif `
     --output-dir work/fixed-features

   topoalign features `
     --mask work/moving-seg/mask.tif `
     --output-dir work/moving-features

   topoalign match `
     --fixed-features work/fixed-features/features.csv `
     --moving-features work/moving-features/features.csv `
     --fixed-shape 2048 2048 `
     --method rigid `
     --output-dir work/matches

   topoalign transform `
     --fixed-features work/fixed-features/features.csv `
     --moving-features work/moving-features/features.csv `
     --matches work/matches/matches.csv `
     --method rigid `
     --output-dir work/transform

   topoalign warp `
     --moving moving.tif `
     --transform work/transform/transform.moving_to_fixed.json `
     --fixed-shape 2048 2048 `
     --output-dir work/warp

其中 ``--fixed-shape`` 的顺序是 ``HEIGHT WIDTH``。对标签 mask 执行 ``warp`` 时增加
``--mask``，以使用最近邻插值并保持整数标签。

使用配置文件
------------

仓库根目录的 ``topoalign.config.example.json`` 同时包含 CLI、Agent 和 registration
设置。复制为项目配置后，``topoalign run`` 会自动读取其中的 ``registration`` 段：

.. code-block:: powershell

   Copy-Item topoalign.config.example.json topoalign.config.json
   # 编辑 topoalign.config.json 中的 registration.fixed / moving / output
   topoalign run

.. warning::

   ``topoalign run --config FILE`` 读取的是 **registration-only** JSON，不能直接传入
   含有 ``version``、``cli``、``agent`` 和 ``registration`` 外层结构的全局示例文件。

一个最小的 registration-only 文件如下：

.. code-block:: json

   {
     "mode": "image",
     "fixed": "fixed.tif",
     "moving": "moving.tif",
     "segmentation": {
       "channel_axis": "auto",
       "registration_channel": -1,
       "gpu": false
     },
     "transform": {
       "method": "rigid",
       "use_ransac": false
     },
     "output": {
       "output_dir": "outputs/config-run"
     }
   }

.. code-block:: powershell

   topoalign run --config registration.json

命令行中显式提供的参数会覆盖配置文件中的对应值。

批处理建议
--------------------

* 每个 Fixed/Moving 对使用独立的 ``--output-dir``。
* 同一输出目录中的固定文件名会被后续运行覆盖。
* 保存实际使用的 ``config.resolved.json`` 和运行日志。
* 先在少量样本上固定参数，再批量执行。
* 批处理结束后统一检查失败任务、匹配数量和残差异常值。

查看结构化结果
--------------

.. code-block:: powershell

   topoalign inspect outputs/sample-001/result.json
   topoalign inspect outputs/sample-001/diagnostics.json --json

输出文件的含义和质量检查顺序见 :doc:`outputs`。
