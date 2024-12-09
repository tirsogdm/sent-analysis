from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import ngrams

from nltk.wsd import lesk
import spacy

from utils import get_lengths

pos_tags = {
    "NOUN": wn.NOUN,   # Nouns
    "VERB": wn.VERB,   # Verbs
    "ADJ": wn.ADJ,     # Adjectives
    "ADV": wn.ADV,     # Adverbs
}

def extract(text, fraction=0.5, sent_threshold=0.05):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    words = [token.text for token in doc]

    for word in words:
        wn_synset = lesk(words, word)
        if wn_synset:
            print(f"{wn_synset.name()}: {wn_synset.definition()}")
            swn_synset = swn.senti_synset(wn_synset.name())
            delta = abs(swn_synset.pos_score() - swn_synset.neg_score())
            if delta > sent_threshold:
                print("Strong sentiment -->", "Δ:", delta, "| Positive score:", swn_synset.pos_score(), ", Negative score:", swn_synset.neg_score())
            print(f"Hypernyms --> {wn_synset.hypernyms()}")
            print("-"*15, '\n')


def boosting(text, fraction=0.5, sent_threshold=0.2):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    words = [token.text for token in doc]
    hypernym_counts = {}

    for token in doc:
        wn_pos = pos_tags.get(token.pos_, None)
        if wn_pos:
            wn_synsets = wn.synsets(token.text, pos=wn_pos)
        else:
            wn_synsets = wn.synsets(token.text)

        wn_synset = wn_synsets[0] if wn_synsets else None
        if wn_synset:
            swn_synset = swn.senti_synset(wn_synset.name())
            delta = abs(swn_synset.pos_score() - swn_synset.neg_score())

            if delta > sent_threshold:
                print(f"{token.text}, {wn_synset.name()}: {wn_synset.definition()}")
                print("Strong sentiment -->", "Δ:", delta, "| Positive score:", swn_synset.pos_score(), ", Negative score:", swn_synset.neg_score())
                hypernyms = wn_synset.hypernyms()
                print(f"Hypernyms --> {wn_synset.hypernyms()}")
                for hypernym in hypernyms:
                    name = hypernym.name()
                    entry = hypernym_counts.get(name, {"count": 0, "value": 0})
                    entry["count"] = entry["count"] + 1
                    entry["value"] = entry["value"] + (fraction * delta)
                    hypernym_counts[name] = entry
                
                print("Found hypernyms -->", [hypernym.name() in words for hypernym in hypernyms])
                print("-"*15, '\n')

    print(f"{len(hypernym_counts)} Hypernym counts -->", hypernym_counts)


def identifying_phrases_wn(text, n=3):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    words = [token.text for token in doc]

    identified_phrases = []
    for size in range(2, n+1):
        for gram in ngrams(words, size):
            phrase = " ".join(gram)
            synsets = wn.synsets(phrase)
            print(phrase, synsets)
            for synset in synsets:
                swn_synset = swn.senti_synset(synset.name())
                delta = abs(swn_synset.pos_score() - swn_synset.neg_score())
                if delta > 0.2:
                    identified_phrases.append(phrase)

    return identified_phrases


def named_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return list(doc.ents)

if __name__ == "__main__":
    pos_text= "This film has a special place in my heart, as when I caught it the first time, I was teaching adult literacy. It rang very true to me and even an outstanding student I had at the time. There are scenes which make you gulp with sudden emotion, and those which even put a smile on your face through sheer identification with the characters and their situation. <br /><br />Excellent performances by Jane Fonda and Robert DeNiro that rank with their best work, a great turn by a young Martha Plimpton, an inspiring story line, and a haunting musical score makes for a most enjoyable and rewarding experience."
    neg_text = "This is it. This is the one. This is the worst movie ever made. Ever. It beats everything. I have never seen worse. Retire the trophy and give it to these people.....there's just no comparison.<br /><br />Even three days after watching this (for some reason I still don't know why) I cannot believe how insanely horrific this movie is/was. Its so bad. So far from anything that could be considered a movie, a story or anything that should have ever been created and brought into our existence.<br /><br />This made me question whether or not humans are truly put on this earth to do good. It made me feel disgusted with ourselves and our progress as a species in this universe. This type of movie sincerely hurts us as a society. We should be ashamed. I really cannot emphasize that our global responsibility as people living here and creating art, is that we need to prevent the creation of these gross distortions of our reality for our own good. It's an embarrassment. I don't know how on earth any of these actors, writers, or the director of this film sleeps at night knowing that they had a role in making 'Loaded'. I don't know what type of disgusting monsters enjoy watching these types of movies.<br /><br />That being said, I love a good 'bad' movie. I love Shark Attack 3, I love Bad Taste, they are HILARIOUS. I tell all my friends to see them because they are 'bad'.<br /><br />But this.......this crosses the line of 'bad' into a whole new dimension. This is awkward bad. This is the bad where you know everything that is going to happen, every line, every action, every death, every sequence BEFORE they happen; and not just like a second or two before, I mean like, after watching the first 5 minutes before.<br /><br />Every cheesy editing 'effect' is shamelessly used over and over again to a sickening point. I really never want to see the 'shaky' camera 'drug buzz rush' effect or jump cuts or swerve cuts or ANY FANCY CUT EVER AGAIN EVER. This is meticulously boring, repetitive and just tortures the audience.<br /><br />But.......and let me be specific here, the most DISTURBING thing about this movie is that given the production, it appears that a somewhat decent amount of money was actually put into this excrement. I personally will grab the shoulders of the director if I ever see him and shake him into submission, demanding that he run home and swallow two-gallons of Drain-O or I will do it for him.<br /><br />If we ever needed a new form of inhumane torture for our war prisoners abroad, just keep showing them this movie in a padded cell over and over again. Trust me, I think they will become more extravagant with suicide methods after the 72nd time of sitting through this.<br /><br />Stop these movies, they are just the most vile of all facets of our society. Please. Stop. NOW."
    ex_text = "The moive was not bad and the acting was top notch."

    boosting(neg_text)

    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(neg_text)