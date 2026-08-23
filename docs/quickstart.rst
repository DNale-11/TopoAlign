快速开始
========

本仓库当前不附带公开演示图像。下面命令是操作模板，请把 ``fixed.tif`` 和
``moving.tif`` 替换为自己的文件。

准备输入
--------

* Fixed：参考图像，定义最终输出坐标。
* Moving：待变换图像，将被映射到 Fixed。
* 优先选择两张图中都清晰、细胞数量足够且视野有较大重叠的核/细胞通道。
* 先裁出有代表性的中小视野调参，再运行整幅图像或 WSI。

Web 快速开始
--------------

从仓库根目录启动：

.. code-block:: powershell

   conda activate cell_registration_gpu
   python webapp/backend.py

浏览器打开 http://localhost:8000 ，按以下顺序操作：

#. 在 **Fixed Image** 中选择参考图像。
#. 在 **Moving Images** 中选择一张或多张待配准图像。
#. 第一次运行保留 **Normal registration** 和默认参数；当前 Web Normal 固定使用
   similarity。需要刚性基线时使用下方 CLI 示例。
#. 确认 GPU 设置与当前 PyTorch 环境一致。
#. 提交配准并等待每张 Moving 顺序完成。
#. 在 Result、Overlay 与 Match Lines 间切换，最后再下载结果。

界面细节和已知限制见 :doc:`web_guide`。

CLI 快速开始
--------------

GPU：

.. code-block:: powershell

   topoalign run `
     --fixed fixed.tif `
     --moving moving.tif `
     --mode image `
     --method rigid `
     --gpu `
     --output-dir outputs/first-run

CPU：

.. code-block:: powershell

   topoalign run `
     --fixed fixed.tif `
     --moving moving.tif `
     --mode image `
     --method rigid `
     --no-gpu `
     --output-dir outputs/first-run-cpu

检查结果：

.. code-block:: powershell

   topoalign inspect outputs/first-run/result.json

重点查看 ``overlay.tif``、``matches.csv``、``diagnostics.json`` 和 ``result.json``。
如果刚性基线合理，再按 :doc:`parameters` 逐项调整，不要一次改变许多参数。

下一步
------

* 需要复用 mask、特征或单独执行阶段：参阅 :doc:`cli_guide`。
* 需要人工修订分割：参阅 :doc:`napari_guide`。
* 需要解释输出和判断质量：参阅 :doc:`outputs`。
