import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, f1_score, recall_score
import numpy as np
random.seed(42)


###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X

def extract_features(samples):
    print("Extracting features ...")
    rows = [] #initiate list of dictionaries (one per news post)
    index = 0 # initiate index
    index_dict = {} # initiate dict to keep track of words and indexes
    for text in samples:
        row_counts = {} # one dict of index: count pairs per text/row
        words = text.split(" ") # tokenize by whitespace
        for word in words:
            if word.isalpha(): # remove numbers and punctuation
                word = word.lower() # lowercase
                if word in index_dict:
                    i = index_dict[word] # get index, if already assigned,
                else:
                    index_dict[word] = index # otherwise assign index
                    i = index
                    index += 1 # and raise index counter by one
                if i in row_counts: # get word counts per text
                    row_counts[i] += 1
                else:
                    row_counts[i] = 1
        rows.append(row_counts)
    features = np.zeros((len(rows), len(index_dict))) # instantiate empty ndarray
    for n, dict in enumerate(rows):
        for ind, val in dict.items():
            features[n, ind] = val # replace zeros by counts
    return features

##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    sv = TruncatedSVD(n_components=n)
    red_X = sv.fit_transform(X)
    return red_X



##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = SVC() # <--- REPLACE THIS WITH A SKLEARN MODEL
    elif clf_id == 2:
        clf = GaussianNB() # <--- REPLACE THIS WITH A SKLEARN MODEL
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf

#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)
    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)


def shuffle_split(X,y):  # TODO: Maybe easier to just randomize in train_test_split? Is it allowed?
    rand_list = []
    for i, row in enumerate(X):
        pair = (row, y[i])
        rand_list.append(pair)
    random.shuffle(rand_list)
    X_s = [i[0] for i in rand_list]
    y_s = [i[1] for i in rand_list]
    X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size = 0.2)
    return X_train, X_test, y_train, y_test


def train_classifer(clf, X, y):
    assert is_classifier(clf)
    clf.fit(X, y)


def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    accuracy = clf.score(X, y)
    precision = average_precision_score(y, clf.predict(X))
    f1 = f1_score(y, clf.predict(X))
    recall = recall_score(y, clf.predict(X))
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-measure:", f1)




######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            )
