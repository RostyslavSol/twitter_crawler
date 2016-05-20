from sklearn.naive_bayes import MultinomialNB
import json

#############################################
## to use perform following steps
##1) create obj
##2) read_sample_file() (to read JSON sample)
##3) create_X_Y() (to build training sample)
##4) fit_direct() (to train NB)
##5) predict_with_NB() (to get the results)
#############################################
class HelperForNB(object):
    #parse json and form X & Y
    def create_X_Y(self, training_sample_arr):
        self.X = []
        self.Y = []
        for json_obj in training_sample_arr:
            try:
                if len(json_obj['cluster']) > 0:
                    self.X.append(json_obj['context_vector'])
                    self.Y.append(json_obj['cluster_index'])
            except:
                continue

        return self.X, self.Y if len(self.X) > 0 else None

    def fit_direct(self, X, Y):
        #train classifier
        self.classifier = MultinomialNB()
        self.classifier.fit(X, Y)

    def predict_with_NB(self, huge_sample):
        return self.classifier.predict(huge_sample)
