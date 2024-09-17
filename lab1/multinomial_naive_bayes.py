import numpy as np


class MultinomialNB:

    def __init__(self):
        # P(C) for each class
        self.class_prob = {}
        # Store document counts per class
        self.class_counts = {}
        # Store word counts per class
        self.class_word_counts = {}
        # All the vocabulary
        self.vocab = set()
        # Store log P(token|class)
        self.token_log_prob = {}

    def tokenize(self, doc):
        return doc.split() # now a list of tokens

    def fit(self, X, y):
        total_docs = 0
        for doc, label in zip(X, y):
            tokens = self.tokenize(doc)
            total_docs += 1
            if label not in self.class_counts:
                self.class_counts[label] = 0
                self.class_word_counts[label] = {}

            self.class_counts[label] += 1

            for token in tokens:
                if token not in self.class_word_counts[label]:
                    self.class_word_counts[label][token] = 0
                self.class_word_counts[label][token] += 1
                self.vocab.add(token)

        # Calculate P(C)
        for label in self.class_counts:
            self.token_log_prob[label] = {}
            self.class_prob[label] = self.class_counts[label] / total_docs
            total_word_count = sum(self.class_word_counts[label].values())
            vocab_size = len(self.vocab)
            for token in self.vocab:
                word_count = self.class_word_counts[label].get(token, 0) + 1
                self.token_log_prob[label][token] = np.log(word_count / (total_word_count + vocab_size))

    # def conditional_prob(self, token, label):
    #     total_word_count = sum(self.class_word_counts[label].values())
    #     word_count = self.class_word_counts[label].get(token, 0) + 1
    #     vocab_size = len(self.vocab)
    #     return word_count / (total_word_count + vocab_size)

    def predict_doc(self,doc):
        tokens = self.tokenize(doc)
        scores = {}
        for label in self.class_counts:
            scores[label] = self.class_prob[label]
            for token in tokens:
                scores[label] += self.token_log_prob[label].get(token, np.log(
                    1 / (sum(self.class_word_counts[label].values()) + len(self.vocab))))
        return max(scores, key=scores.get)

    def predict(self, X):
        return [self.predict_doc(doc) for doc in X]

