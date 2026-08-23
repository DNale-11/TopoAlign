安装
====

系统要求
--------

* Windows 或 Linux；以下示例以 PowerShell 为主。
* Python 3.10。
* CPU 模式无需 NVIDIA GPU 或 CUDA。
* GPU 模式需要 NVIDIA GPU、兼容驱动和 CUDA 版 PyTorch。
* 第一次运行 Cellpose-SAM 时可能需要联网下载模型。
* 只有 CLI Agent 或 Web AI Assistant 需要 OpenAI-compatible API Key。

获取源码
--------

.. code-block:: powershell

   git clone https://github.com/DNale-11/cell_registration.git
   cd cell_registration

CPU 安装
--------

CPU 环境适合先验证流程。分割大图会更慢，但 CLI、Web 和 napari 入口均可使用。

.. code-block:: powershell

   conda create -n cell_registration_cpu python=3.10 -y
   conda activate cell_registration_cpu
   python -m pip install --upgrade pip

   python -m pip install "torch>=2.7,<2.13" "torchvision>=0.22,<0.28" `
     --index-url https://download.pytorch.org/whl/cpu

   python -m pip install -r webapp/requirements.txt
   python -m pip install -e .

验证：

.. code-block:: powershell

   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

CPU 环境最后一项应为 ``False``。

GPU 安装
--------

下面以 CUDA 12.8 wheel 为例。实际安装时应根据显卡驱动，使用
`PyTorch 官方安装器 <https://pytorch.org/get-started/locally/>`_ 选择正确的 wheel 索引。

.. code-block:: powershell

   conda create -n cell_registration_gpu python=3.10 -y
   conda activate cell_registration_gpu
   python -m pip install --upgrade pip

   python -m pip install "torch>=2.7,<2.13" "torchvision>=0.22,<0.28" `
     --index-url https://download.pytorch.org/whl/cu128

   python -m pip install -r webapp/requirements.txt
   python -m pip install -e .

验证 GPU：

.. code-block:: powershell

   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

只有最后一项为 ``True`` 时才应在 TopoAlign 中启用 GPU。GPU 主要加速分割；
匹配、变换估计和部分 I/O 仍在 CPU 上执行。

安装可选组件
------------

Agent：

.. code-block:: powershell

   python -m pip install -e ".[agent]"

napari：

.. code-block:: powershell

   python -m pip install -e .\napari-cell-registration
   napari

napari 的 OpenSlide WSI 支持：

.. code-block:: powershell

   python -m pip install -e ".\napari-cell-registration[wsi]"
   python -c "import openslide; print(openslide.__version__)"

.. note::

   ``openslide-python`` 只是 Python binding，系统还必须安装原生 OpenSlide runtime。
   ``[wsi]`` 也不会自动安装 CellViT++、``pathopatch`` 或模型权重。

安装验证
--------

.. code-block:: powershell

   topoalign --version
   topoalign --help
   python -c "import cell_registration; print('TopoAlign import OK')"

Windows 的 ``.\webapp\start_server.bat`` 固定激活名为 ``cell_registration_gpu`` 的
Conda 环境。环境名称不同时，请手动激活后运行 ``python webapp/backend.py``。
