import importlib
import json

from components.Pipeline import Pipeline
from components.Dataset import Dataset
from spacy.tokens import Doc, Token
from pathlib import Path

from utils import read_in

"""
RUN EXPERIMENTS
"""

# Load the data
reviews = read_in('data/pos', 1)
reviews += read_in('data/neg', -1)

# Initialise the pipeline and dataset
pipeline = Pipeline()
pipeline.show()
dataset = Dataset([1, -1], data=reviews, pipeline=pipeline)
train_set, eval_set, test_set = dataset.split()

# Load config
with open("config.json") as config_file:
    config = json.load(config_file)
experiments_to_run = config.get("exps_to_run", [])

# Run experiments
exp_path = Path('experiments')
exp_files = sorted(exp_path.glob('exp_*.py'), key=lambda x: int(x.stem.split('_')[-1]))

# for exp_file in exp_files:
#     module_name = f"{exp_path.name}.{exp_file.stem}"
#     if exp_file.stem not in experiments_to_run:
#         continue

#     module = importlib.import_module(module_name)

#     # Get classifier class and filtering and text processing methods
#     description = getattr(module, 'description')
#     CustomClassifier = getattr(module, 'CustomClassifier')
#     filtered = getattr(module, 'filtered')
#     processed_text = getattr(module, 'processed_text')

#     # Set the filtered and processed_text extensions and initialise classifier
#     Token.set_extension('filtered', method=filtered, force=True)
#     Doc.set_extension('processed_text', method=processed_text, force=True)
#     classifier = CustomClassifier(train_set, eval_set)
    
#     # Run the classifier and print info
#     print("="*50)
#     print(f"Running {module_name}")
#     print(description)
#     classifier.run()


"""
EVALUATE MODEL ON TEST SET
"""
from experiments.exp_16 import CustomClassifier, filtered, processed_text

Token.set_extension('filtered', method=filtered, force=True)
Doc.set_extension('processed_text', method=processed_text, force=True)

# Run the classifier and print info
print("="*50)
print(f"Evaluating sklearn naive bayes model on test set using final exp_16 feature set...")
print("="*50)
# Sklearn
classifier = CustomClassifier(train_set, test_set)
classifier.run()
# Custom Naive Bayes
print("="*50)
print(f"Evaluating custom naive bayes model on test set using final exp_16 feature set...")
print("="*50)
# Sklearn
classifier = CustomClassifier(train_set, test_set)
counts = classifier.train(type='custom')
predictions = classifier.evaluate()