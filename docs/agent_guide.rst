Agent user guide
================

The optional TopoAlign Agent accepts natural-language requests and calls a
defined set of local inspection and registration tools. It requires an API
key and an endpoint supporting the OpenAI **Responses API**, including
function calling and response continuation. Support for a chat-completions
endpoint alone is insufficient.

The standard :doc:`cli_guide` and :doc:`napari_guide` do not require an API
key. The Agent runs registration locally; its model calls use the configured
remote service.

Install and configure
---------------------

From the repository root, in the environment where TopoAlign is installed:

.. code-block:: console

   python -m pip install -e ".[agent]"

Set the API key in your shell. For PowerShell:

.. code-block:: powershell

   $env:OPENAI_API_KEY = "YOUR_API_KEY"

For Bash or Zsh:

.. code-block:: bash

   export OPENAI_API_KEY="YOUR_API_KEY"

Start an interactive session:

.. code-block:: console

   topoalign agent --workspace .

Version 0.1.0 defaults to endpoint ``https://api.openai.com/v1`` and model
``gpt-5``. To select a different endpoint or a model available to your account,
pass explicit options:

.. code-block:: console

   topoalign agent --workspace . --base-url https://example.com/v1 --model YOUR_MODEL

.. important::

   Although the code uses ``OPENAI_BASE_URL`` and ``TOPOALIGN_AGENT_MODEL``,
   startup overwrites these variables from the loaded settings. In version
   0.1.0, use ``--base-url`` and ``--model``, or the JSON settings below, to
   configure them reliably. ``OPENAI_API_KEY`` works when the settings do not
   contain a direct ``api_key`` value.

Persistent settings
~~~~~~~~~~~~~~~~~~~

The Agent reads ``~/.topoalign/config.json`` followed by
``<workspace>/topoalign.config.json``. Project settings override user settings;
unspecified values use defaults. An example project file is:

.. code-block:: json

   {
     "version": 1,
     "cli": {
       "language": "en"
     },
     "agent": {
       "base_url": "https://api.openai.com/v1",
       "api_key": "",
       "api_key_env": "OPENAI_API_KEY",
       "model": "gpt-5",
       "max_tool_rounds": 8
     }
   }

``api_key_env`` selects the environment variable containing the key. A
nonempty ``agent.api_key`` takes precedence over that variable. Explicit
``--api-key``, ``--base-url``, and ``--model`` arguments override the loaded
Agent connection settings; prefer an environment variable for the key so it
does not appear in command history.

The bundled example also contains ``enabled`` and ``timeout_seconds``.
``enabled`` is not an access-control switch for ``topoalign agent``. In this
release, ``timeout_seconds`` is not passed to the Agent client: normal API
calls use a 30-second timeout and connection tests use 15 seconds, with SDK
retries disabled.

You can also run ``topoalign`` with no subcommand and choose the Agent or the
connection-configuration option in the interactive menu. That configuration
option saves the entered key as plain text in the selected settings file.
Keep ``api_key`` empty and use an environment variable if you do not want the
key stored there. Do not commit credential-bearing settings files.

Choose a workspace and start a task
-----------------------------------

Use a project directory containing the intended inputs and a dedicated
output subdirectory. Agent tool paths are resolved relative to
``--workspace``; their path checks are intended to keep file operations within
that directory. Select the smallest practical workspace rather than your
home directory.

For example:

.. code-block:: text

   my-registration/
     topoalign.config.json
     data/
       fixed_mask.tif
       moving_mask.tif
     outputs/

Launch the Agent with ``my-registration`` as the workspace and state the
reference image, moving image, and output location explicitly. A useful
first request is:

.. code-block:: text

   Inspect data/fixed_mask.tif and data/moving_mask.tif. Round 1 is fixed.
   Report their dimensions and propose a rigid registration configuration.
   Do not run registration yet.

After reviewing the proposal, a run request can specify:

.. code-block:: text

   Run the proposed rigid registration from moving to fixed using these
   masks. Save the results in outputs/pair-001. Then read result.json and
   report the match count, residuals, and output paths.

The Agent executes allowed tools when the model requests them; there is no
separate confirmation prompt for every operation. State whether you want
inspection, a proposal, or execution. Use a new output directory for each
run so existing artifacts are not overwritten.

For a single request that exits after completion:

.. code-block:: console

   topoalign agent --workspace . --prompt "Inspect data/fixed_mask.tif and data/moving_mask.tif; report their shapes without running registration."

The top-level shorthand also works:

.. code-block:: console

   topoalign --workspace . --prompt "Read outputs/pair-001/result.json and summarize its recorded results."

Capabilities and boundaries
---------------------------

Use ``/tools`` to list the tools available to the model:

.. list-table:: Agent tools
   :header-rows: 1
   :widths: 35 65

   * - Tools
     - Purpose
   * - ``inspect_project``, ``search_code``, ``read_code_file``
     - List sampled project files, search Python source, and read bounded source/configuration excerpts.
   * - ``inspect_input``, ``validate_config``
     - Inspect image dimensions and value ranges, preview CSV columns/rows, read JSON, and validate a registration configuration.
   * - ``segment``, ``extract_features``
     - Generate a mask from an image or extract cell features from a mask.
   * - ``match``, ``estimate_transform``
     - Match feature tables and estimate a moving-to-fixed transform.
   * - ``warp``, ``run_registration``
     - Warp moving data or execute the configured registration pipeline.
   * - ``read_result``, ``list_artifacts``
     - Read a result manifest and list an output directory's files.

These are wrappers around TopoAlign operations, with the same dependency and
input requirements as the CLI. Installing ``[agent]`` supplies the API client;
it does not by itself install a segmentation model or GPU runtime. For
reproducible settings and supported input modes, see :doc:`parameters` and
:doc:`cli_guide`.

The model cannot execute arbitrary shell commands or edit source files
through these tools. Image inspection returns numerical metadata rather than
a rendered overlay. CSV inspection previews only the first five rows, and
source reading is limited to supported text suffixes and files no larger than
200,000 bytes. A successful configuration check does not prove that masks are
biologically correct or that two fields of view overlap.

Prompts and tool results are sent to the configured model endpoint. These can
include paths, source excerpts, CSV previews, and complete JSON contents.
Keep secrets out of files you ask the Agent to read. ``/config`` redacts the
key in its own local display, but that does not redact arbitrary JSON read by
other tools.

Interactive commands
--------------------

Commands beginning with ``/`` are handled locally:

.. list-table:: Session commands
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Action
   * - ``/+`` or ``/help``
     - Show command help.
   * - ``/model``
     - Show the current model.
   * - ``/model MODEL_ID``
     - Test the model, switch to it, save the setting, and reset conversational context.
   * - ``/api``
     - Test the endpoint, key, and selected model with a small Responses API request.
   * - ``/config``
     - Show active settings with the API key redacted.
   * - ``/reload``
     - Reload saved settings and reset conversational context.
   * - ``/tools``
     - List model-callable tools.
   * - ``/local COMMAND``
     - Run a TopoAlign ``run``, ``segment``, ``features``, ``match``, ``transform``, ``warp``, or ``inspect`` subcommand.
   * - ``/result``
     - Display the most recently modified ``result.json`` found in the workspace.
   * - ``/artifacts``
     - Display the artifact mapping from that latest result.
   * - ``/clear``
     - Reset the conversation context used for subsequent model requests.
   * - ``/exit``
     - End the Agent session; return to the main shell if launched from its menu.

``exit``, ``quit``, and ``:q`` also end the session. ``/clear`` does not delete
local files or previously submitted data from the API service.

``/local`` uses the process working directory for ordinary CLI relative paths;
it does not change directory to ``--workspace``. Start TopoAlign from the
workspace directory or supply absolute paths when mixing these commands.
``/result`` selects by modification time, so specify an exact manifest path in
a prompt when several runs are present.

Review results and recover from errors
--------------------------------------

Ask the Agent to identify the input files, transform direction, match count,
residuals recorded in the result, and saved artifacts. Then inspect overlays
and match coverage yourself. Version 0.1.0 does not include an Agent tool
that computes a standardized ``good``/``review``/``poor`` quality grade;
descriptive model judgments are not a validated registration score.

* **Missing API key or package:** set the configured key environment variable
  and install the Agent extra in the active environment.
* **Connection or model error:** run ``/api``. Check Responses API support,
  endpoint URL, account access to the model, and credentials. Use explicit
  connection options or JSON settings as described above.
* **Path outside the workspace:** move the task inputs under the selected
  workspace, or restart with an appropriate common parent directory.
* **Maximum tool-call rounds reached:** a model turn defaults to eight rounds.
  Inspect the existing output directory before asking it to continue; an
  earlier tool call may already have written results.
* **API failure after a local run:** check the run directory or use
  ``/result`` and ``/artifacts``. A failed model response does not undo local
  files produced by completed tools.

For registration failures, use :doc:`troubleshooting`; the local CLI remains
available independently of Agent connectivity.
