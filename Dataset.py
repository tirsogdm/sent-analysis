# PYTHON MODULES
from sklearn.model_selection import train_test_split
from collections import Counter
from spacy.tokens import Doc
from tabulate import tabulate
from tqdm import tqdm
import spacy

Doc.set_extension('rating', default=None)
Doc.set_extension('label', default=None)
Doc.set_extension('id', default=None)

# CUSTOM MODULES
from utils import read_in

"""
Reviews should store info about entire dataset.
"""
class Dataset(list):
    """
    A list of reviews as Doc objects.  
    """
    def __init__(self, labels, docs=None, pipeline=None, data=[], type="parent"):
        """
        Initialize with list of Doc object or with pipeline and reviews list.

        Parameters
        ----------
        laels : list[int]
            List of labels of passed reviews.

        docs : list[spacy.Doc], optional
            List of reviews of type Doc.

        pipeline : spacy.Language, optional
            The spaCy pipeline to process the reviews.
        
        data : list[tuple[str, dict[str, str, int]]], optional
            List of reviews to be processed by the pipeline.
        
        type : str, optional
            Type of dataset.
        """
        self.labels = labels

        if not docs:
            if pipeline and data:
                docs = []
                for review, context in tqdm(data, desc="Processing reviews"):
                    doc = pipeline(review)
                    doc._.id = context['id']
                    doc._.label = context['label']
                    doc._.rating = context['rating']
                    docs.append(doc)
            else:
                raise ValueError("Pipeline and data must be provided.")

        super().__init__(docs)

    def get_label_subsets(self):
        """
        Compute and return label subsets.

        Returns
        -------
        list[Dataset]
            List of label subsets.
        """
        return [Dataset([label], docs=[doc for doc in self if doc._.label == label], type="labelset") for label in self.labels] if len(self.labels) > 1 else None

    def split(self):
        """
        Split the dataset into training, evaluation, and testing sets.

        Returns
        -------
        Dataset, Dataset, Dataset
            Training, evaluation, and testing sets.
        """
        train_data, remaining_data = train_test_split(self, test_size=0.3, random_state=41)
        eval_data, test_data = train_test_split(remaining_data, test_size=0.5, random_state=41)

        print(f"Training set: {len(train_data)} docs")
        print(f"Evaluation set: {len(eval_data)} docs")
        print(f"Testing set: {len(test_data)} docs")

        return Dataset(self.labels, docs=train_data, type="subset"), Dataset(self.labels, docs=eval_data, type="subset"), Dataset(self.labels, docs=test_data, type="subset")

    def flatten(self):
        """
        Separates the review texts and labels into distinct lists.

        Parameters
        ----------
        exclude : list[str]
            List of words to exclude when filtering.

        Returns
        -------
        list[str], list[int]
            List of reviews and list of labels.
        """
        data = []
        labels = []
        for doc in self:
            data.append(doc._.processed_text(self))
            labels.append(doc._.label)
        return data, labels

    # --- ANALYSIS METHODS ---
    def frequency_distribution(self):
        """
        Compute frequency distribution of words in the dataset.

        Returns
        -------
        Counter
            Frequency distribution of word lemmas across dataset.
        """
        freq_dist = Counter()
        for doc in self:
            lemmas = [token.lemma_.lower() for token in doc]
            freq_dist.update(lemmas)
        return freq_dist

    # NOTE: CAN ADD SENTENCE ANALYSIS through doc.sents
    def get_statistics(self, verbose=False):
        """
        Get basic statistics about the dataset.

        Returns
        -------
        list[str]
        """
        stats = []
        for subset in self.get_label_subsets():
            vocab = [token.lemma_ for doc in subset for token in doc]
            total_lemmas = len(vocab)
            avg_rev_length = total_lemmas / len(subset)
            vocab_size = len(set(vocab))
            diversity = total_lemmas / vocab_size
            # Out
            subset_stats = ["Label", "Average Review Length (words)", "Vocabulary Size", "Diversity Ratio"], [subset.labels[0], avg_rev_length, vocab_size, diversity]
            stats.append(subset_stats)
            if verbose:
                print(tabulate(subset_stats))
        return stats

    def get_unique_vocab(self, verbose=False):
        """
        Get the vocabulary unique to each each subset.
        
        Returns
        -------
        list[str]
            List of unique words in both subsets.
        """
        subsets = self.get_label_subsets()
        vocab1 = [token.lemma_ for doc in subsets[0] for token in doc]
        vocab2 = [token.lemma_ for doc in subsets[1] for token in doc]
        
        unique_v1 = len(set(vocab1) - set(vocab2))
        unique_v2 = len(set(vocab2) - set(vocab1))

        if verbose:
            print(tabulate(["Unique to", "Size"], [subsets[0].labels[0], len(set(vocab1) - set(vocab2))], [subsets[1].labels[0], len(set(vocab2) - set(vocab1))]))
        return [unique_v1, unique_v2]
    


if __name__ == "__main__":
    config = {
        'tokeniser': {'method': 'spacy'},
        'word_normaliser': {'method': 'lemma'},
        'filter': {'method': 'punctuation'},
        'phrase_extractor': {'method': 'ngram', 'args': {'n': 2}},
        'boost': {'method': ''},
        'feature_normaliser': {'method': 'row'},
    }

    reviews = read_in('data/pos', 1)
    reviews += read_in('data/neg', -1)

    pipeline = spacy.load('en_core_web_sm')
    pipeline.remove_pipe('ner')
    pipeline.remove_pipe('tok2vec')
    pipeline.analyze_pipes(pretty=True)
    dataset = Dataset([1, -1], pipeline=pipeline, data=reviews)
    train_dataset, eval_dataset, test_dataset = dataset.split()
    dataset.get_statistics(verbose=True)