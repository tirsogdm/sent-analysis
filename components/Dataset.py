from sklearn.model_selection import train_test_split
from collections import Counter
from spacy.tokens import Doc
from tabulate import tabulate
from tqdm import tqdm
import spacy
import sys
import os

Doc.set_extension('rating', default=None)
Doc.set_extension('label', default=None)
Doc.set_extension('id', default=None)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

    @property
    def frequency_distribution(self):
        """
        Compute frequency distribution of words in the dataset.

        Returns
        -------
        Counter
            Frequency distribution of word lemmas across dataset.
        """
        if not hasattr(self, '_freq_dist'):
            freq_dist = Counter()
            for doc in self:
                lemmas = [token.lemma_.lower() for token in doc]
                freq_dist.update(lemmas)
            self._freq_dist = freq_dist
        return self._.freq_dist

    # --- ANALYSIS METHODS ---
    def get_base_statistics(self, verbose=False):
        """
        Get basic statistics about the dataset.

        Returns
        -------
        list[str]
        """
        stats =  []
        for subset in self.get_label_subsets():
            vocab = []
            n_sents = 0
            for doc in subset:
                vocab += [token.text for token in doc] # .text or .lemma_
                n_sents += len(list(doc.sents))
            word_count = len(vocab)
            avg_rev_length = word_count / len(subset)
            avg_sent_length = word_count / n_sents
            vocab_size = len(set(vocab))

            diversity = word_count / vocab_size
            
            subset_stats = [["Label", "Average Review Length", "Average Sentence Length", "Vocabulary Size", "Diversity Ratio"], [subset.labels[0], avg_rev_length, avg_sent_length, vocab_size, diversity]]
            stats.append(subset_stats)
            if verbose:
                print(tabulate(subset_stats, headers="firstrow"))
        return stats

    def get_unique_vocab(self, verbose=False):
        """
        Get the vocabulary unique to each each subset.
        
        Returns
        -------
        list[[str]]
            List of unique words in both subsets.
        """
        subsets = self.get_label_subsets()
        
        vocab_words = [[], []]
        vocab_lemmas = [[], []]
        for i, subset in enumerate(subsets):
            for doc in subset:
                vocab_lemmas[i] += [token.lemma_ for token in doc]
                vocab_words[i] += [token.text for token in doc]

        unique_words_pos = len(set(vocab_words[0]) - set(vocab_words[1]))
        unique_words_neg = len(set(vocab_words[1]) - set(vocab_words[0]))
        unique_lemmas_pos = len(set(vocab_lemmas[0]) - set(vocab_lemmas[1]))
        unique_lemmas_neg = len(set(vocab_lemmas[1]) - set(vocab_lemmas[0]))

        data = [['Label', 'Unique Words', 'Unique Lemmas']] 
        data += [
            [subsets[0].labels[0], unique_words_pos, unique_lemmas_pos],
            [subsets[1].labels[0], unique_words_neg, unique_lemmas_neg]
        ]
        if verbose:
            print(tabulate(data, headers="firstrow"))

        return data
    
    def get_pos_statistics(self, verbose=False):
        pass

if __name__ == "__main__":
    reviews = read_in('data/pos', 1)
    reviews += read_in('data/neg', -1)

    pipeline = spacy.load('en_core_web_sm')
    pipeline.remove_pipe('ner')
    dataset = Dataset([1, -1], pipeline=pipeline, data=reviews)
    train_dataset, eval_dataset, test_dataset = dataset.split()
    
    dataset.get_base_statistics(verbose=True)
    dataset.get_unique_vocab(verbose=True)
    dataset.get_pos_statistics(verbose=True)