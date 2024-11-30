# PYTHON MODULES
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from spacy.tokens import Doc, Token
from sklearn import metrics

# CUSTOM MODULES
from Pipeline import build_pipeline
from Dataset import Dataset
from utils import read_in

# Setting filtered attribute on Token and processed_text attribute on Doc
def filtered(token, dataset):
    """
    Method called on each token to check if it should be filtered.
    """
    if token.is_punct or token.is_space:
        return True

def processed_text(doc, dataset):
    """
    Method called on each doc to return the processed text containing only non-filtered tokens.
    """
    text = []
    for token in doc:
        if not token._.filtered(dataset):
            text.append(token.text)
    return " ".join(text)

Token.set_extension('filtered', method=filtered)
Doc.set_extension('processed_text', method=processed_text)

if __name__ == "__main__":
    reviews = read_in('data/pos', 1)
    reviews += read_in('data/neg', -1)

    # Pipeline
    pipeline = build_pipeline({})
    dataset = Dataset([1, -1], data=reviews, pipeline=pipeline)
    train_set, eval_set, test_set = dataset.split()

    train_data, train_labels = train_set.flatten()
    eval_data, eval_labels = eval_set.flatten()

    cv = CountVectorizer()
    train_counts = cv.fit_transform(train_data)

    clf = MultinomialNB().fit(train_counts, train_labels)
    eval_counts = cv.transform(eval_data)
    predictions = clf.predict(eval_counts)

    print(metrics.confusion_matrix(eval_labels, predictions))
    print(metrics.classification_report(eval_labels, predictions))