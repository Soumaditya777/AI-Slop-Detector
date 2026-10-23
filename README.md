# AI Slop Detector

A Streamlit application that scores text with the Hugging Face model
[`gouwsxander/slop-detector-bert`](https://huggingface.co/gouwsxander/slop-detector-bert).

Dataset: [`gouwsxander/wikipedia-human-ai`](https://huggingface.co/datasets/gouwsxander/wikipedia-human-ai).

## Features

- AI-class score and binary prediction
- Overlapping windows for longer text
- Per-window score table and decoded text
- Optional dataset evaluation script
- Model loaded once per Streamlit session cache

## 1. Requirements

Python 3.10 or 3.11 is recommended. The first run downloads model files from Hugging Face,
so an internet connection is needed. Model inference runs locally after download.

## 2. Create an environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails on your Mac, follow the official instructions at
https://pytorch.org/get-started/locally/ and then install the remaining requirements.

## 3. Run the app

From the project root:

```bash
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`.

## 4. Evaluate

The evaluation script defaults to the `test` split and up to 500 samples:

```bash
python evaluation/evaluate.py --split test --max-samples 500
```

If the dataset does not expose a `test` split or uses different field names, inspect it:

```bash
python -c "from datasets import load_dataset; d=load_dataset('gouwsxander/wikipedia-human-ai'); print(d)"
```

Then choose an available split and pass its column names, for example:

```bash
python evaluation/evaluate.py --split train --text-column text --label-column label --max-samples 500
```

Check the dataset's label mapping before trusting metrics. The evaluation script expects
0 = human and 1 = AI, consistent with the model card's `LABEL_1` AI class.

## Project layout

```text
ai-slop-detector/
├── app.py
├── detector.py
├── evaluation/
│   └── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Important limitations

This model is not a universal AI detector. Its training data is domain-specific and the
model card reports variable generalization across genres. A score is not proof of authorship.
Do not use the result as the sole basis for high-stakes decisions.

The document score is the mean of overlapping-window AI scores. It is a useful display
heuristic, not a calibrated probability that the entire document was AI-generated.
