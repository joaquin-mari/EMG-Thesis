## How to run

This thesis uses **uv** (`pyproject.toml` + `uv.lock`) for reproducible dependencies.

1. Install **uv**:
```bash
pip install uv
```

2. Install dependencies:
```bash
uv sync
```

3. Open the **`.ipynb` notebooks** in your editor (VSCode / Jupyter / etc.) and make sure the notebook kernel uses the **uv environment** (you may need to set the Python interpreter / environment variable in your editor).

4. Run the notebooks in the `models/` folder to compare results across different models for **spontaneous activity detection**.
