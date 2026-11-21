"""
Run a notebook's code cells in order in one namespace (for long sweeps under nohup).

    nohup venv/bin/python -u research/run_notebook.py train_cnn_runs.ipynb >> runs/sweep_clean.log 2>&1 &
    MPLBACKEND=Agg venv/bin/python research/run_notebook.py plot_runs.ipynb

Paths are relative to the repo root, which is also the working directory and the first sys.path entry.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(nb_path: str) -> None:
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    nb = json.load(open(nb_path))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        print(f"===== cell {i} =====", flush=True)
        src = "".join(cell["source"])
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        exec(compile(src, f"{nb_path}:cell{i}", "exec"), ns)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
