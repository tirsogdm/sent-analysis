# PYTHON MODULES
from sklearn import metrics
import spacy

# CUSTOM MODULES
from utils import read_in
from Dataset import Dataset
from FeatureExtractor import FeatureExtractor
from FeatureVectoriser import FeatureVectoriser

class CustomPipeline():
    def __init__(self):
        pass

# Base Feature Extractor
class BaseFeatureExtractor(FeatureExtractor):
    def __init__(self, dataset):
        super().__init__(dataset)

    def filter(self):
        """
        Base filter removes stop words, punctuation, and spaces.
        """
        for doc in self.dataset:
            for token in doc:
                if token.is_punct or token.is_space:
                    token._.filtered = True

    def extract(self):
        """
        Base extractor returns the text of the tokens.
        """
        for doc in self.dataset:
            text = []
            for token in doc:
                if not token._.filtered:                    
                    text.append(token.text)
            doc._.processed_text = " ".join(text)

# Base Feature Vectoriser
class BaseFeatureVectoriser(FeatureVectoriser):
    def __init__(self, train_data, train_labels, test_data, test_labels):
        super().__init__(train_data, train_labels, test_data, test_labels)

    def vectorise(self, data):
        return data


if __name__ == "__main__":
    reviews = read_in('data/pos', 1)
    reviews += read_in('data/neg', -1)

    # Pipeline
    pipeline = spacy.load('en_core_web_sm')
    pipeline.remove_pipe('ner')
    pipeline.remove_pipe('tok2vec')

    dataset = Dataset([1, -1], data=reviews, pipeline=pipeline)
    train_set, eval_set, test_set = dataset.split()

    train_data, train_labels = BaseFeatureExtractor(train_set).run()
    eval_data, eval_labels = BaseFeatureExtractor(eval_set).run()

    predictions = BaseFeatureVectoriser(train_data, train_labels, eval_data, eval_labels).run()
    print(metrics.confusion_matrix(eval_labels, predictions))
    print(metrics.classification_report(eval_labels, predictions))