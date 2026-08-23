故障排查与已知限制
==================

安装与启动
----------

找不到 topoalign 命令
~~~~~~~~~~~~~~~~~~~~~

确认当前环境已执行 ``python -m pip install -e .``，然后重新激活环境。源码树中也可用：

.. code-block:: powershell

   python topoalign.py --help

Cellpose 或 matplotlib 导入失败
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

根 ``pyproject.toml`` 只声明数值核心依赖。完整图像/Web 环境应先安装：

.. code-block:: powershell

   python -m pip install -r webapp/requirements.txt
   python -m pip install -e .

CUDA available 为 False
~~~~~~~~~~~~~~~~~~~~~~~~

检查当前 Python 是否来自预期 Conda 环境，并确认 PyTorch wheel、显卡驱动和 CUDA
兼容。未修复前关闭 TopoAlign GPU 选项，避免任务直接失败。

CUDA 显存不足
~~~~~~~~~~~~~~~~~~~~

先缩小 ROI、使用分块分割、降低并发，或暂时改为 CPU。Web 多 Moving 会串行执行，
但单张超大图仍可能耗尽内存。

分割与匹配
----------

没有细胞或细胞数量异常
~~~~~~~~~~~~~~~~~~~~~~

* 确认选择的是核/细胞通道而不是空白通道。
* 检查 ``channel_axis`` 和 ``registration_channel``。
* 打开 mask 查看是否全为 0、标签粘连或背景碎片过多。
* ``min_area`` / ``max_area`` 只过滤进入特征和匹配的对象，不会修改原 mask。

匹配为 0 或非常少
~~~~~~~~~~~~~~~~~~~~~~~~

* 确认 Fixed/Moving 没有反选且视野实际重叠。
* 检查两侧分割尺度和细胞密度是否相近。
* 先扩大空间窗口，再小幅调整权重。
* 外部 feature CSV 应直接使用 TopoAlign 生成的完整 ``features.csv``。只有
  ``centroid_x`` / ``centroid_y`` 的表不足以支持默认形态与拓扑匹配。

配准看似成功但方向相反
~~~~~~~~~~~~~~~~~~~~~~

重新确认 Fixed 是目标坐标、Moving 是源坐标。变换必须按 ``moving_to_fixed`` 使用。
不要把矩阵反向套到 Moving；需要反变换时应显式求逆并记录新方向。

CLI
---

配置文件提示 unknown keys
~~~~~~~~~~~~~~~~~~~~~~~~~

``topoalign run --config`` 只接受 registration-only JSON。全局
``topoalign.config.example.json`` 应复制为 ``topoalign.config.json`` 并由程序自动读取，
不能直接传给 ``--config``。

mode 与实际输入不一致
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当前 ``--mode auto|image|mask|features`` 会写入配置，但完整服务实际根据你是否提供
image、mask、features 和 matches 路径选择可用阶段；``mode`` 目前不强制分派或验证。
应以明确传入的文件参数为准。

纯 features 结果被裁切
~~~~~~~~~~~~~~~~~~~~~~

主 ``topoalign run`` 没有 ``--fixed-shape`` 参数。纯 features 工作流应在
registration-only JSON 顶层设置 ``"fixed_shape": [HEIGHT, WIDTH]``，否则边界会按
两张特征表中的最大质心推断，可能小于真实 Fixed 图像。

旧结果被覆盖
~~~~~~~~~~~~

TopoAlign 不会保护已有同名输出。每次运行和每个批次项目使用不同 ``output_dir``。

Web
---

Mask 页签提示先运行分割
~~~~~~~~~~~~~~~~~~~~~~~

完整 Registration 生成的分割文件不会回填 Web 的 Mask 页签。需要页签预览时，
对当前活动图像另点一次 **Run Segmentation**；这会重复分割。

刷新后批次消失
~~~~~~~~~~~~~~

批次队列保存在页面内存。刷新或关闭页面后，未提交的后续 Moving 不会继续；已经提交
到服务器的单个任务可能仍在运行。当前版本没有批次恢复或取消功能。

服务重启后 Task not found
~~~~~~~~~~~~~~~~~~~~~~~~~

任务状态索引保存在服务进程内存中。重启后 API 无法恢复索引，但结果文件通常仍在
``data/tasks/<task_id>``，可从磁盘检查。

CellViT 或 RANSAC 看起来没有变化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这是当前 Web 实现限制：CellViT 下拉框实际仍调用 Cellpose-SAM；Normal 固定为
similarity，而 RANSAC 只对 rigid 实现，因此开关不生效。

WSI 任务内存过高
~~~~~~~~~~~~~~~~~~~~~~~~

Web WSI 会完整加载普通 TIFF/PNG，不是 OpenSlide 分块管线。缩小 ROI，或使用 napari
的 **WSI Segmentation** 与显式 WSI registration。

napari 与 WSI
-------------

OpenSlide 导入或 DLL 错误
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

安装 ``openslide-python`` 后仍需平台原生 OpenSlide runtime。确认
``python -c "import openslide"`` 在启动 napari 的同一环境成功。

CellViT++ 无法启动
~~~~~~~~~~~~~~~~~~~~~~~~

确认 CUDA 可用，resolution 为 0.25 或 0.5，并另行安装 CellViT++、``pathopatch``、
模型权重及其运行依赖。``[wsi]`` extra 只增加 OpenSlide Python binding。

导出再重载的 WSI mask 不能注册
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Save Mask Layers** 只保存 TIFF，不保存 napari layer 中的
``wsi_centroid_metadata``。WSI Registration 需要两侧成对元数据。建议在同一 napari
会话中从 WSI Segmentation 直接进入 Registration；若跨会话，需同时保存并重建元数据。

napari 保存刚性回退时失败
~~~~~~~~~~~~~~~~~~~~~~~~~

Normal 最终少于 50 对匹配时会采用刚性回退。当前版本在此路径启用
``save_results`` 可能在写出部分文件后报未定义 TPS 控制点。关闭保存完成图层质检，
再从 napari 手动保存所需图层，或使用 CLI 生成刚性结果。

安全与隐私
----------

Web 服务无认证、允许跨域并监听所有网卡。仅在可信本机或受控内网运行；不要直接暴露
到公网。Web Assistant 密钥存于浏览器 localStorage，共享设备使用后应清除。
