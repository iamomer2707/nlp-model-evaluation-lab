# Reproducing the benchmark

This folder contains the source notebook and Python script for the matched-split Transformer experiment summarized in `../data/benchmark_results.json`.

Run the notebook in Google Colab with a GPU runtime, or install the experiment dependencies and run `python run_experiments.py` from this folder. The script downloads the SMS Spam Collection and Hugging Face checkpoints, then writes metrics and figures to `outputs/`. Those runtime files and downloaded data are ignored by the repository.

The benchmark values in the web dashboard came from the supplied Colab T4 run (seed 42, stratified 70/15/15 split). Reruns can differ because timing, framework versions, and accelerator load can vary.
