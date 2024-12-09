from spacy.tokenizer import Tokenizer
from spacy.tokens import Token

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import spacy

from tabulate import tabulate

class Pipeline:
    def __init__(self, tokeniser='spacy'):
        """
        Initialise the pipeline with the default spaCy pipeline
        """
        self._pipeline = spacy.load('en_core_web_sm')
        self.configure_pipeline(tokeniser)

    def __call__(self, text):
        """
        Process the given text through the pipeline.

        Parameters
        ----------
        text : str
            The text to process.
        
        Returns
        -------
        spacy.Doc
            The processed text.
        """
        return self._pipeline(text)

    def __getattr__(self, name):
        """
        Delegate get attribute to the spaCy pipeline.

        Parameters
        ----------
        name : str
            The attribute name.
        
        Returns
        -------
        Any
            The attribute of the spaCy pipeline.
        """
        return getattr(self._pipeline, name)

    def configure_pipeline(self, tokeniser):
        """
        Configure spaCy nlp pipeline.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        spacy.Language
            The spaCy pipeline. 
        """
        self._pipeline.remove_pipe('ner')

        if tokeniser == 'whitespace':
            self._pipeline.tokenizer = self.whitespace_tokenizer(self._pipeline)
        elif tokeniser == 'nltk':
            self._pipeline.tokenizer = self.nltk_tokenizer(self._pipeline)

        # Add stem attribute to each Token 
        stemmer = PorterStemmer()
        def get_stem(token):
            return stemmer.stem(token.text)

        # Add nltk lemma to each Token
        lemmatiser = WordNetLemmatizer()
        def get_nltk_lemma(token):
            return lemmatiser.lemmatize(token.text)

        Token.set_extension('stem', getter=get_stem)
        Token.set_extension('nltk_lemma', getter=get_nltk_lemma)

    # -- Tokenisers
    def whitespace_tokeniser(nlp):
        return Tokenizer(nlp.vocab, token_match=None, rules={}, prefix_search=None, suffix_search=None)

    def nltk_tokeniser(nlp):
        pass

    def show(self):
        """
        Example output of pipeline.
        """
        text = "This is how this pipeline tokenises and extracts linguistic features, is it working as expected?"
        doc = self(text)

        data = [
            ["Token", "POS Tag", "Spacy Lemma", "NLTK Lemma", "Stem"],
        ]
        for token in doc:
            data.append([
                token.text,
                token.pos_,
                token.lemma_,
                token._.nltk_lemma,
                token._.stem
            ])

        print(tabulate(data, headers="firstrow"))

if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.show()