#Need to read and set up the data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

training = pd.read_csv('./train.csv')

x_train = np.array(training.iloc[:, :-1])
y_train = np.array(training.iloc[:, -1:])

stop_words = stopwords.words('english')
vector = CountVectorizer(stop_words = stop_words, binary = True)
