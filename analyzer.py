####################################################################
######################### RAW ANALYZER #############################
####################################################################

# Import necessary libs
import json
import pandas as pd
import matplotlib.pyplot as plt
import re


class TwitterAnalyzer(object):
    # ctor
    def __init__(self, user_sample_file_name, raw_data_file_name):
        self.tweets_data = []
        self.tweets_dataframe = None
        self.raw_data = []

        self.data_file = open(raw_data_file_name, mode='r')
        self.user_sample_file = open(user_sample_file_name, mode='r')

    # methods
    #TODO: FINISH USER SAMPLE FILE ANALYSIS
    def extract_user_sample(self):
        pass

    def try_parse_data(self):
        for line in self.data_file:
            try:
                tweet = json.loads(line)
                self.tweets_data.append(tweet)
            except:
                continue

    def get_tweets_count(self):
        return len(self.tweets_data)

    def make_tweets_dataframe(self):
        self.tweets_dataframe = pd.DataFrame()

        self.tweets_dataframe['text'] = list(map(lambda x: x['text'] if 'text' in x else None, self.tweets_data))
        self.tweets_dataframe['lang'] = list(map(lambda x: x['lang'] if 'lang' in x else None, self.tweets_data))

    def word_in_text(self, word, text):
        is_in = False
        if word != None and text != None:
            word = word.lower()
            text = text.lower()

            is_in = re.search(word, text)
        return is_in

    def mine_with_words(self, words):
        arrs = []
        for word in words:
            self.tweets_dataframe[word] = self.tweets_dataframe['text'].apply(lambda text: self.word_in_text(word, text))
            arrs.append(len(self.tweets_dataframe[word].value_counts()))

        # plot gistograms
        x_pos = list(range(len(words)))
        width = 0.2
        fig, ax = plt.subplots()

        plt.bar(x_pos, arrs, width=width)

        # setting axes and labels
        ax.set_ylabel('Number of tweets')
        ax.set_title('Sports comparison')

        plt.grid()
        plt.show()

# def main():
#     analyzer = TwitterAnalyzer()
#
#     analyzer.try_parse_data()
#     analyzer.make_tweets_dataframe()
#     analyzer.mine_with_words(['Chelsea', 'football'])
#
# main()