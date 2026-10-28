"""Quick reproducibility check (about 1 minute).

Re-runs the 12 methods on the first two 100-task test instances with the
shipped policies and tuned parameters, and compares M, E, C, LB and F with
results/main_uniform_100.csv. Also checks that data/ matches the generator.

    python src/verify.py
"""
import os, sys, shutil, subprocess, tempfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
subprocess.run([sys.executable, "src/export_dataset.py", "--check"], check=True)

tmp = tempfile.mkdtemp()
shutil.copytree("src", f"{tmp}/src"); shutil.copytree("policies", f"{tmp}/policies")
os.makedirs(f"{tmp}/results")
for f in os.listdir("results"):
    if f.startswith("tune_"):
        shutil.copy(f"results/{f}", f"{tmp}/results/")
subprocess.run([sys.executable, "src/evaluate.py", "main", "100", "--reps", "2"], cwd=tmp, check=True,
               stdout=subprocess.DEVNULL)
k, cols = ["rep", "method"], ["M", "E", "C", "LB", "F"]
a = pd.read_csv(f"{tmp}/results/main_uniform_100.csv").set_index(k).sort_index()[cols]
b = pd.read_csv("results/main_uniform_100.csv").query("rep < 2").set_index(k).sort_index()[cols]
d = (a - b).abs().max().max()
shutil.rmtree(tmp)
print(f"{len(a)} runs re-computed, max abs difference = {d:.3g}")
print("REPRODUCED" if d < 1e-9 else "DIFFERENT (check library versions)")
