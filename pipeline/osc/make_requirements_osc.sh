#!/bin/bash
# Training subset of the laptop freeze. Torch, spconv, and cumm are installed
# from their CUDA indexes. Dash, Flask, Jupyter, and Open3D are left out.
set -euo pipefail
cd "$(dirname "$0")"
grep -E -v \
  -e '^(dash|Flask|blinker|itsdangerous|Werkzeug|retrying|plotly|ipython|ipywidgets|widgetsnbextension|jupyter_core|jupyterlab_widgets|nbformat|comm|matplotlib-inline|prompt_toolkit|jedi|parso|pexpect|ptyprocess|stack-data|traitlets|wcwidth|executing|asttokens|pure_eval|decorator|nest-asyncio|fastjsonschema|jsonschema|jsonschema-specifications|referencing|rpds-py|open3d-cpu|torch|torch_scatter|torchvision|spconv-cu118|cumm-cu118|triton|uv|click|Jinja2|MarkupSafe|narwhals|portalocker)==' \
  -e '^nvidia-' \
  requirements_laptop.txt > requirements_osc.txt
echo "numpy==1.23.5" > constraints.txt
echo "requirements_osc.txt $(wc -l < requirements_osc.txt) lines"
