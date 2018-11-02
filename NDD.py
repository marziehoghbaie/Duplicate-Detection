from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import math


class NDD:
    def __init__(self):
        self.data = load_files("./test", description='all', categories=None,
                               load_content=True, shuffle=False, encoding='utf-8', decode_error='ignore',
                               random_state=42)
        self.Number_of_docs = len(self.data.data)
        self.vectorizer = CountVectorizer(analyzer='word', max_df=0.5, min_df=3, stop_words=None)
        self.data = self.vectorizer.fit_transform(self.data.data)
        self.SimilarDocuments = []
        self.dictionary = self.vectorizer.vocabulary_
        self.dict_len = len(self.dictionary)
        self.n = len(self.dictionary)
        self.data = self.data.toarray()
        self.duplicates = [
            "1, 2",
            "1, 3",
            "2, 3",
            "15, 70",
            "7, 9",
            "7, 11",
            "9, 11",
            "20, 21",
            "20, 22",
            "20, 23",
            "21, 22",
            "21, 23",
            "22, 23",
            "80, 81",
            "80, 82",
            "80, 83",
            "80, 84",
            "81, 82",
            "81, 83",
            "81, 84",
            "82, 83",
            "82, 84",
            "83, 84",
            "65, 66",
            "65, 67",
            "65, 68",
            "38, 65",
            "65, 72",
            "66, 67",
            "66, 68",
            "38, 66",
            "66, 72",
            "67, 68",
            "38, 67",
            "67, 72",
            "38, 68",
            "68, 72",
            "38, 72",
        ]
        self.document_length = []
        self.threashold = 0.5  # the Average of similarity value used for comparision

    def PDSM(self, x, y):
        similarity = 0.0
        A = B = absent = present = 0.0

        for index in range(self.n):

            A += min(x[index], y[index])
            B += max(x[index], y[index])

            if x[index] == y[index] == 0:
                absent += 1

            if x[index] != 0 and y[index] != 0:
                present += 1

        absent = self.n - absent - 1
        present += 1
        similarity = (present / absent) * (A / B)
        return similarity

    def jaccrad_coeff(self, x, y):
        similarity = 0.0
        A = 0.0
        Lx = Ly = 0.0

        for index in range(self.n):
            A += x[index] * y[index]
            Lx += x[index] * x[index]
            Ly += y[index] * y[index]

        temp = Lx + Ly - A
        similarity = (A / temp)
        return similarity

    def cosine(self, x, y):
        similarity = 0.0
        A = 0.0
        Lx = Ly = 0.0

        for index in range(self.n):
            A += x[index] * y[index]
            Lx += x[index] * x[index]
            Ly += y[index] * y[index]

        temp = Lx * Ly
        temp = math.sqrt(temp)
        similarity = (A / temp)
        return similarity

    def EUC(self, x, y):
        similarity = 0.0
        for index in range(self.n):
            temp = x[index] - y[index]
            temp = temp * temp
            similarity += temp
        similarity = math.sqrt(similarity)
        return similarity

    def Manhattan(self, x, y):
        similarity = 0.0
        for index in range(self.n):
            temp = x[index] - y[index]
            similarity += math.fabs(temp)
        return similarity

    def duplicate_detection(self):

        for i in range(self.Number_of_docs):
            similarity = 0.0
            for j in range(i + 1, self.Number_of_docs):
                # similarity = self.PDSM(self.data[i], self.data[j])
                # similarity = self.cosine(self.data[i], self.data[j])
                # similarity = self.Manhattan(self.data[i], self.data[j])
                # similarity = self.EUC(self.data[i], self.data[j])
                # similarity = self.jaccrad_coeff(self.data[i], self.data[j])
                if similarity > self.threashold:
                    self.SimilarDocuments.append("%d, %d" % (i, j))

    def precision_recal(self):
        temp = 0.0
        self.duplicate_detection()

        for similar in self.SimilarDocuments:
            for duplicate in self.duplicates:

                if similar == duplicate:
                    temp += 1

        recall = temp / len(self.duplicates)
        precision = temp / (len(self.SimilarDocuments))
        print(recall)
        print(precision)

# create a new instance of NDD
instance = NDD()
# call function for near duplicate detection
instance.precision_recal()
