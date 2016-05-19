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
    def read_sample_file(self, json_sample_filename):
        #get json sample from txt file
        json_sample_filename = json_sample_filename if '.txt' in json_sample_filename else json_sample_filename + '.txt'
        json_sample_file = open(json_sample_filename, 'r')
        text = json_sample_file.read()

        self.json_sample = text.split('\n')
        if '' in self.json_sample:
            self.json_sample.remove('')

    #parse json and form X & Y
    def create_X_Y(self):
        self.X = []
        self.Y = []
        for json_str in self.json_sample:
            try:
                json_obj = json.loads(json_str)
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
