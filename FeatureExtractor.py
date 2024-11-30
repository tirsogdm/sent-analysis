# PYTHON MODULES
from abc import ABC, abstractmethod
from spacy.tokens import Doc, Token

# COULD BE SET USING METHOD EXTENSIONS ON ATTRIBUTES
Doc.set_extension('processed_text', default=None)
Token.set_extension('filtered', default=False)


class FeatureExtractor(ABC):
    def __init__(self, dataset):
        self.dataset = dataset

    def run(self):
        self.filter()
        self.extract()
        return self.dataset.flatten()

    @abstractmethod
    def filter(self):
        """
        Filter dataset by marking words (Tokens in Docs) which shouldn't be included (setting attribute Token._.filtered to True)
        """
        pass

    @abstractmethod
    def extract(self):
        """
        Extract desired data (text, lemma, or stem) from Tokens in Doc review objects (setting the value at attribute Doc._.processed_text) conditioned on Token._.filtered
        """
        pass