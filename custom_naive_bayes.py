from sklearn.naive_bayes import MultinomialNB
import json

class HelperForNB(object):
    def __init__(self, json_sample_filename):
        #get json sample from txt file
        json_sample_filename = json_sample_filename if '.txt' in json_sample_filename else json_sample_filename + '.txt'
        json_sample_file = open(json_sample_filename, 'r')
        text = json_sample_file.read()

        self.json_sample = text.split('\n')
        self.json_sample.remove('')

        #parse json and form X & Y
        X = []
        Y = []
        for json_str in self.json_sample:
            json_obj = json.loads(json_str)
            if len(json_obj['cluster']) > 0:
                X.append(json_obj['context_vector'])
                Y.append(json_obj['cluster_index'])

        #train classifier
        self.classifier = MultinomialNB()
        self.classifier.fit(X, Y)

    def predict_with_NB(self, huge_sample):
        return self.classifier.predict(huge_sample)

obj = HelperForNB('target_result')
print(obj.predict_with_NB([[0,0,1,0,0,0,0,0,0,0,0,0]]))