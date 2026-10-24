# Signal Lab — NLP Model Evaluation

An interactive, cyber-themed workspace for inspecting a pretrained-model benchmark and trying text classification on your own data.

## What you can do

- Classify a message in the live sandbox with a real TF-IDF + Logistic Regression model trained at runtime on the public SMS Spam Collection.
- Compare the supplied matched-split results for RNN, LSTM, attention-LSTM, a scratch Transformer, DistilBERT, BERT, RoBERTa, and MiniLM.
- Inspect accuracy, spam precision/recall/F1, parameter counts, loss curves, and confusion matrices.
- Upload a labeled CSV to train a fresh baseline and evaluate it on a stratified holdout. This also works for other binary text classification tasks.

The live sandbox baseline is deliberately separate from the Transformer benchmark. It does not claim to run the fine-tuned Transformer checkpoints from the benchmark.

## Run locally

Requires Python 3.10 or newer and an internet connection for the first launch. The app downloads the public UCI dataset into memory, trains the small baseline, and serves the interface locally.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000). The benchmark dashboard loads even if the UCI download is unavailable; live classification then displays an offline status.

Alternatively, build and run the included container:

```bash
docker build -t signal-lab .
docker run --rm -p 7860:7860 signal-lab
```

Then open [http://localhost:7860](http://localhost:7860). The container is compatible with Docker-based hosting, including Hugging Face Spaces.

## Evaluate your own dataset

Upload a UTF-8 CSV with one text column and one label column. Accepted text headers are `text`, `message`, `sms`, or `content`; accepted label headers are `label`, `target`, `class`, or `category`. The CSV must contain exactly two label values, at least 20 non-empty rows, and at least five rows for each class. The app uses a seed-42 stratified 75/25 train/test split. It reports accuracy, positive-class precision/recall/F1, and a confusion matrix.

Download `static/assets/starter_binary_text.csv` from the interface to try the upload flow. The positive class is the second label in sorted order; for `ham` and `spam`, that is `spam`.

Uploaded CSVs and typed messages are processed in memory and are not written to disk by the app. For a public deployment, use an instance you control and review its hosting provider's request-logging and data-retention settings.

## Benchmark provenance

The benchmark panel reproduces the supplied experiment on the SMS Spam Collection: 5,574 messages, stratified seed-42 70/15/15 split, and one Google Colab NVIDIA T4 run. The benchmark charts and data are included under `static/assets/` and `data/benchmark_results.json`.

| Result | Model | Value |
| --- | --- | ---: |
| Highest spam F1 | RoBERTa | 98.23% |
| Highest accuracy | RoBERTa and MiniLM | 99.52% |
| Efficient high scorer | MiniLM | 98.21% spam F1; 33.36M parameters |

Assignment 2 historical scores used a different split and are not mixed into the matched-split model table. One split and one timing run do not establish performance on newer messages or production traffic. The SMS collection is an older English-language benchmark and this demo is for education and exploration, not production spam filtering.

Dataset: Almeida and Hidalgo, [SMS Spam Collection](https://doi.org/10.24432/C5CC84), UCI Machine Learning Repository. The app fetches the data from [UCI](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) on first launch; the dataset is not bundled.

## Project layout

```text
main.py                          FastAPI API and local classifier
static/index.html                App interface
static/styles.css                Responsive cyber visual system
static/app.js                    Prediction, benchmark, and upload interactions
static/assets/                   Benchmark figures and starter CSV
data/benchmark_results.json      Matched-split benchmark results
requirements.txt                 Runtime dependencies
Dockerfile                       Container deployment
research/                        Source notebook and script for the original Transformer benchmark
```
