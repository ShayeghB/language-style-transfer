from file_io import load_sent

PATH = '../data/yelp/sentiment'
train0 = load_sent(PATH + '.train' + '.0')
# train1 = load_sent(PATH + '.train' + '.1')
dev0 = load_sent(PATH + '.dev' + '.0')
# dev1 = load_sent(PATH + '.dev' + '.1')
test0 = load_sent(PATH + '.test' + '.0')
# test1 = load_sent(PATH + '.test' + '.1')

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
vectorizer = TfidfVectorizer(
    stop_words=stop_words,
    max_features=10000,
    max_df=0.5,
    use_idf=True,
    ngram_range=(1,3)
)
fitting_data = [' '.join(sent) for sent in [train0, dev0]]
X = vectorizer.fit_transform(fitting_data)
terms = vectorizer.get_feature_names()

from sklearn.cluster import KMeans
num_clusters = 10
km = KMeans(n_clusters=num_clusters)
km.fit(X)
clusters = km.labels_.tolist()

# from sklearn.utils.extmath import randomized_svd
# U, Sigma, VT = randomized_svd(X, n_components=10, n_iter=100, random_state=122)
# for i, comp in enumerate(VT):
#     terms_comp = zip(terms, comp)
#     sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
#     print("Concept "+str(i)+": ")
#     for t in sorted_terms:
#         print(t[0])
#     print(" ")

test_data = [' '.join(sent) for sent in [test0]]
X_test = vectorizer.transform(test_data)
test_clusters = km.predict(X_test)

