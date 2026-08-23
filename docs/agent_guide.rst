Agent 使用指南
==============

Agent 是可选入口。CLI、Web 本地配准和 napari 都不依赖 API Key。

安装与配置
----------

.. code-block:: powershell

   python -m pip install -e ".[agent]"
   $env:OPENAI_API_KEY = "YOUR_API_KEY"

使用 OpenAI-compatible 服务时还可设置：

.. code-block:: powershell

   $env:OPENAI_BASE_URL = "https://example.com/v1"
   $env:TOPOALIGN_AGENT_MODEL = "MODEL_NAME"

优先使用环境变量，不要把真实密钥写入版本控制中的 JSON、脚本或截图。

交互模式
--------

.. code-block:: powershell

   topoalign agent --workspace .

也可以直接运行 ``topoalign``，在交互菜单中进入 Agent。常见请求包括：

* 检查 Fixed/Moving 图像或已有 mask。
* 为一对图像建议刚性基线参数。
* 执行获准的本地配准阶段。
* 读取 ``result.json``、匹配表和重叠 mask，解释可能的问题。

单次请求
--------

.. code-block:: powershell

   topoalign agent `
     --workspace . `
     --prompt "检查 fixed.tif 和 moving.tif，并给出第一轮刚性配准参数。"

也支持顶层简写：

.. code-block:: powershell

   topoalign --workspace . --prompt "检查 outputs/sample-001/result.json"

项目配置
--------

``topoalign.config.json`` 的 ``agent`` 段可以设置 ``base_url``、``model``、超时和
最大工具轮数。``api_key_env`` 用于指定读取密钥的环境变量名。即使配置支持
``api_key`` 字段，也不建议把密钥直接写入项目文件。

结果判断
--------

Agent 会综合匹配数量、残差分布和有效重叠比例给出 ``good``、``review`` 或 ``poor``
等提示。这些结论明确属于启发式筛查：

* 匹配少于 3 对通常无法可靠估计变换。
* P95 残差高于约 5 px 建议人工复核，高于约 10 px 常提示明显异常。
* 有效几何覆盖低于约 0.25 建议检查视野重叠与变换方向。

这些阈值不是生物学真值，也不能替代叠加图、匹配空间覆盖和实验元数据检查。

Web Assistant
-------------

Web 右侧 Assistant 与配准后端相互独立；未配置或不可用时不影响本地配准。
其 API URL、模型和 Key 会以明文形式保存在当前浏览器的 ``localStorage`` 中。
共享计算机使用后应清除设置，并避免保存生产密钥。
