# author: Ing. Jan Jilecek
# license MIT
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import nltk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Analyzer():
    def __init__(self):
        with open("dataset.txt", "r") as data:  # otevreni datove sady
            self.dataset = data.readlines()
        self.X = None  # inicializace struktur pro datovou sadu
        self.y = np.array(40 * [1]) # 20 pozitivnich a 20 negativnich review
        self.y[20:] = [0] * 20  # 1 = pozitivni, 0 - negativni

    def lemmatize(self, text):
        stemmer = WordNetLemmatizer()
        sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # odstraneni spojek a predlozek
        sentence = re.sub(r'\W', ' ', sentence)  # odstraneni specialnich znaku
        sentence = sentence.lower().split()  # prevedeni na mala pismena a rozdeleni na slova
        sentence = [stemmer.lemmatize(word) for word in sentence] # lematizuje slova ve vete
        sentence = ' '.join(sentence) # a vrati je do vetne podoby
        return sentence

    def process_sentences(self):
        arr = []
        for sentence in self.dataset:
            arr.append(self.lemmatize(sentence))  # lematizuje kazdou vetu v datasetu

        vectorizer = CountVectorizer(max_features=20, # vzit prvnich TOP 20 slov
                                     min_df=2,        # minimalne dva vyskyty v review
                                     max_df=0.6,      # max v 60% review
                                     stop_words=stopwords.words('english'))
        tfidf_transformer = TfidfTransformer()
        self.X = vectorizer.fit_transform(arr).toarray()  # vektorizuje slova
        print(self.X)
        self.X = tfidf_transformer.fit_transform(self.X).toarray()  # zvazi frekvenci slov v celem dokumentu


    def train(self):
        # rozdelime dataset na trenovaci a testovaci podmnozinu
        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, # hodnoceni
                             self.y, # jejich trida (0-neg,1-pos)
                             test_size=0.3, # 30% je testovacich
                             random_state=9)

        # pouzijeme zakladni klasifikator K-NN
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, y_train) # natrenovani klasifikatoru

        # predikce
        y_pred = classifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred)) # tisk matice zamen
        print(classification_report(y_test, y_pred)) # vystup klasifikace
        print(accuracy_score(y_test, y_pred)) # presnost modelu

    def sentiment_analysis(self):
        analyser = SentimentIntensityAnalyzer()
        snt = analyser.polarity_scores(self.dataset[1])
        print("{:-<40} {}".format(self.dataset[1], str(snt)))

if __name__ == '__main__':
    analyzer = Analyzer()
    analyzer.process_sentences()
    analyzer.train()

    #analyzer.sentiment_analysis()
