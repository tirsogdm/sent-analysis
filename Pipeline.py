import spacy
from spacy.tokens import Doc, Token
from nltk.stem import PorterStemmer

def build_pipeline(config):
    """
    Build a spaCy pipeline with the given configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    spacy.Language
        The spaCy pipeline. 
    """
    pipeline = spacy.load('en_core_web_sm')
    pipeline.remove_pipe('ner')
    pipeline.remove_pipe('tok2vec')

    # Add stemmer as attribute
    stemmer = PorterStemmer()

    def get_stem(token):
        return stemmer.stem(token.text)

    Token.set_extension('stem', getter=get_stem)

    return pipeline