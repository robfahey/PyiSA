from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import pandas as pd

from pyisax import PyiSA


def make_dtm_from_texts(texts):
    stemmer = PorterStemmer()
    analyzer = CountVectorizer(stop_words='english').build_analyzer()

    def stemmed(doc):
        return (stemmer.stem(w) for w in analyzer(doc))

    stem_vectorizer = CountVectorizer(analyzer=stemmed, min_df=0.005)

    tdm = stem_vectorizer.fit_transform(texts)

    return tdm


if __name__ == '__main__':
    print('Testing pyiSA on Trump-related tweets...')

    with open('Trump.csv', newline='') as trump_file:
        trump_data = pd.read_csv(trump_file)

    print('Tweet data loaded. Creating term-document matrix ({} tweets).'.format(len(trump_data['text'])))
    X = make_dtm_from_texts(trump_data['text'])
    print('Term-Document Matrix created; {} features x {} tweets.'.format(X.shape[1], X.shape[0]))

    print('Preparing TDM for iSA processing... ', end='', flush=True)
    X = PyiSA.prep_data(X)
    print('[DONE]')
    y = trump_data['Sentiment']

    X_train = [X[i] for i in range(0, len(y)) if not pd.isnull(y[i])]
    X_test = [X[i] for i in range(0, len(y)) if pd.isnull(y[i])]
    y_train = [y[i] for i in range(0, len(y)) if not pd.isnull(y[i])]
    print('Training set (coded tweets): {}'.format(len(X_train)))
    print('                   Test set: {}'.format(len(X_test)))

    my_isa = PyiSA(verbose=True)
    my_isa.fit(X_train, X_test, y_train)

    print('\nResults:')
    print(my_isa.best_table)
