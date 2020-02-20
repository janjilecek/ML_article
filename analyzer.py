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
        with open("dataset.txt", "r") as data:  # open the data-set for reading
            self.dataset = data.readlines() 
        self.X = None  # init the data structures
        self.y = np.array(40 * [1]) # 20 positive and 20 negative lines
        self.y[20:] = [0] * 20  # 1 = positive, 0 - negative

    def lemmatize(self, text):
        stemmer = WordNetLemmatizer()
        sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # remove single letter words
        sentence = re.sub(r'\W', ' ', sentence)  # remove special chars
        sentence = sentence.lower().split()  # convert to lowercase and split to words
        sentence = [stemmer.lemmatize(word) for word in sentence]
        sentence = ' '.join(sentence) # back into a space delimited sentence of words
        return sentence

    def process_sentences(self):
        arr = []
        for sentence in self.dataset:
            arr.append(self.lemmatize(sentence))  # lemmatize every data-set sentence

        vectorizer = CountVectorizer(max_features=20, # take only TOP 20 words
                                     min_df=2,        # at least 2 occurences
                                     max_df=0.6,      # at most in 60% of the text
                                     stop_words=stopwords.words('english'))
        tfidf_transformer = TfidfTransformer()
        self.X = vectorizer.fit_transform(arr).toarray()  # vectorize 
        print(self.X)
        self.X = tfidf_transformer.fit_transform(self.X).toarray()  # frequency analysis


    def train(self):
        # split the data-set to training and testing set
        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, # values, reviews
                             self.y, # class (0-neg,1-pos)
                             test_size=0.3, # 30% for testing
                             random_state=9)

        # basic K-nearest neighbors classifier
        classifier = KNeighborsClassifier(n_neighbors=5)
        classifier.fit(X_train, y_train) # classifier training

        # prediction
        y_pred = classifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred)) # print report
        print(classification_report(y_test, y_pred)) 
        print(accuracy_score(y_test, y_pred)) 

    def sentiment_analysis(self): # loads a sentence to VADER
        analyser = SentimentIntensityAnalyzer()
        snt = analyser.polarity_scores(self.dataset[1])
        print("{:-<40} {}".format(self.dataset[1], str(snt)))

if __name__ == '__main__':
    analyzer = Analyzer()
    analyzer.process_sentences()
    analyzer.train()

    #analyzer.sentiment_analysis()
