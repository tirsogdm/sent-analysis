# Sentiment Analysis Coursework

This repository contains a sentiment classification workflow for movie reviews using:

- spaCy-based text processing
- multiple feature-engineering experiments (`experiments/exp_*.py`)
- `sklearn.naive_bayes.MultinomialNB`
- a custom Naive Bayes implementation (`components/CustomNB.py`)

`main.py` runs selected experiments from `config.json`, then evaluates the final feature set on the test split.

## Repository layout

- `/home/runner/work/sent-analysis/sent-analysis/main.py` – experiment runner and final evaluation entrypoint
- `/home/runner/work/sent-analysis/sent-analysis/config.json` – list of experiment modules to execute
- `/home/runner/work/sent-analysis/sent-analysis/experiments/` – experiment definitions (`description`, token filtering, processed text, classifier feature logic)
- `/home/runner/work/sent-analysis/sent-analysis/components/` – reusable pipeline, dataset, classifier, and custom NB components
- `/home/runner/work/sent-analysis/sent-analysis/setup/nltk_setup.py` – helper script for downloading NLTK resources

## Data format

Create a `data/` directory in the repository root with:

- `data/pos/` for positive reviews (label `1`)
- `data/neg/` for negative reviews (label `-1`)

Each file is expected to follow:

`<id>_<rating>.txt`

Example: `12345_9.txt`

The loader parses:
- `id` from the prefix
- `rating` from the suffix
- `label` from the directory (`pos` or `neg`)

## Setup

From `/home/runner/work/sent-analysis/sent-analysis`:

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Ensure spaCy model is available (included in requirements, but you can also install manually):
   - `python -m spacy download en_core_web_sm`
3. Download required NLTK resources:
   - Update the download path in `/home/runner/work/sent-analysis/sent-analysis/setup/nltk_setup.py`
   - Run: `python setup/nltk_setup.py`

## Running experiments

1. Select experiments in `/home/runner/work/sent-analysis/sent-analysis/config.json` under `exps_to_run`.
2. Run:
   - `python main.py`

Execution flow:

1. Reviews are loaded from `data/pos` and `data/neg`.
2. A `Dataset` is built and split into train/eval/test.
3. Each selected experiment module is imported and run.
4. After experiment runs, the script evaluates the final feature setup (from `exp_16`) on the test set with:
   - `MultinomialNB` (scikit-learn)
   - `CustomNB` (repository implementation)

## Notes

- The repository includes experimental utility scripts like `testing_special_features.py` and `experimenting_with_bert.ipynb` that are not part of the main execution path.
- `data/` is ignored by git and must be provided locally.
