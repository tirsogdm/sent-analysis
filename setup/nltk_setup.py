import nltk

resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'sentiwordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
for resource in resources:
    # WARN DOWNLOAD_DIR !!!
    nltk.download(resource, download_dir='/Users/tirso/.pyenv/versions/nlp-cw/nltk_data')