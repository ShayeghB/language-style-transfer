from file_io import load_sent
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from functools import partial
from sklearn.neighbors import KNeighborsClassifier




class Evaluator:
    def __init__(
        self,
        train_data,
        num_clusters=10,
    ):
        # stop_words = stopwords.words('english')
        self.vectorizer = TfidfVectorizer(
            min_df=0.01,
            use_idf=True,
        )
        X = self.vectorizer.fit_transform(train_data)
        self.pca = TruncatedSVD(n_components=num_clusters)
        X_pca = self.pca.fit_transform(X)
        self.km = KMeans(n_clusters=num_clusters)
        self.km.fit(X_pca)
        # self.knn = KNeighborsClassifier(n_neighbors=3)
        # self.knn.fit(X_pca, self.km.labels_)



    def get_clusters(
        self,
        data
    ):
        X = self.vectorizer.transform(data)
        X_pca = self.pca.transform(X)
        # return self.knn.predict(X_pca)
        return self.km.predict(X_pca)


    def evaluate(
        self,
        source,
        target,
        print_table=False,
        metric=accuracy_score,
    ):
        source_clusters = self.get_clusters(source)
        target_clusters = self.get_clusters(target)
        if print_table:
            print(classification_report(source_clusters, target_clusters))
        return (np.array(source_clusters)==np.array(target_clusters)).mean()




def load_data(path):
    return [' '.join(sent) for sent in load_sent(path)]


def bidirectional_eval(
    train_data,
    test_data,
    tsf_test_data,
    num_clusters,
    print_table=False,
    metric=accuracy_score,
):
    evaluator0 = Evaluator(train_data=train_data[0], num_clusters=num_clusters)
    evaluator1 = Evaluator(train_data=train_data[1], num_clusters=num_clusters)
    return evaluator0.evaluate(test_data[0], tsf_test_data[0], print_table=print_table, metric=metric), \
           evaluator1.evaluate(test_data[1], tsf_test_data[1], print_table=print_table, metric=metric)


def plot_acc_cluster(file_path, *data, **args):
    cluster_counts = [2,5,10,20,50,100]
    accuracies = [bidirectional_eval(*data, num_clusters=i, **args) for i in tqdm(cluster_counts)]
    accuracies = list(zip(*accuracies))
    plt.plot(cluster_counts, accuracies[0], label='0')
    plt.plot(cluster_counts, accuracies[1], label='1')
    plt.legend()
    plt.xlabel('# clusters')
    plt.ylabel('Accuracy')
    plt.savefig(file_path)




def main():
    source_dir = '../data/yelp/'
    target_dir = '../tmp/'
    data_name = 'sentiment'
    train_key = 'train'
    dev_key = 'dev'
    test_key = 'test'
    tsf_key = 'tsf'

    train_data, test_data, tsf_test_data = {}, {}, {}
    for i in range(2):
        train_data[i] = load_data(source_dir+data_name+'.'+train_key+'.'+str(i)) + \
                        load_data(source_dir+data_name+'.'+dev_key+'.'+str(i))
        test_data[i] = load_data(source_dir+data_name+'.'+test_key+'.'+str(i))
        tsf_test_data[i] = load_data(target_dir+data_name+'.'+test_key+'.'+str(i)+'.'+tsf_key)
    data = [train_data, test_data, tsf_test_data]

    plot_acc_cluster('../f1.jpg', *data, metric=partial(f1_score, average='macro'))
    # bidirectional_eval(*data, num_clusters=10, print_table=True)



if __name__=='__main__':
    main()