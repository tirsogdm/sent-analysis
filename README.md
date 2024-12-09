SENTIMENT ANALYSIS COURSEWORK

main.py launches all experiments listed in config.json, under "exps_to_run".

- By default is set to run all experiments.
- When experiments are finished, model is evaluated on test set using final feature set - using sklearn's MultinomialNB(), and the customNB().

Requirements

- requirements.txt
- running setup/nltk_setup.py (altering paths)
- spacy "en_core_web_sm" download
- data/ directory at root with pos/ and neg/ subdirectories.
