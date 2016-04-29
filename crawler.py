# Import necessary libs
import json
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

# constants
ACCESS_TOKEN = "2291093965-a08Vm2RWf7IU45tMz44Nmwb4cc4bxztHmpujqVm"
ACCESS_TOKEN_SECRET = "Ajts6p6lPDSgIoEsUM8outlHQpdH6OWqPgsW6pRCPvnwT"

API_KEY = "JmiG2HI5OyK2mhMQxne7O7W1J"
API_SECRET = "keBkKDdhkjn2BbG6JJLkTu10wszh4qpSP7pOjDTTeNGOxKc7m7"

class StdOutListener(StreamListener):
    def __init__(self, filename, tweets_count):
        filename = filename if '.txt' in filename else filename + '.txt'
        self.outFile = open(filename, 'w')
        self.tweets_count = tweets_count
        self.tweets_index = 0

    def on_data(self, raw_data):
        #parse json
        tweet = json.loads(raw_data)
        try:
            write_json = json.dumps({
                                     "user_id": str(tweet['user']['id']),
                                     "text": tweet['text'],
                                     "lang": tweet['lang']
                                     })
        except:pass

        #write to file
        self.outFile.write(write_json + '\n')

        #cycle condition
        self.tweets_index += 1
        if self.tweets_index >= self.tweets_count:
            self.outFile.close()
            return False
        else:
            return True

    def on_error(self, status_code):
        self.outFile.write(str(status_code))

class TwitterCrawler(object):
    def __init__(self, filename, tweets_count):
        listnr = StdOutListener(filename, tweets_count)
        auth = OAuthHandler(API_KEY, API_SECRET)
        auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
        self.stream = Stream(auth, listnr)

    def filter_by_params(self, words=None, langs=None, follows=None, locations=None):
        self.stream.filter(track=words,
                           languages=langs,
                           follow=follows,
                           locations=locations)

# track=['Champions League','Bayern','Real Madrid', 'Barcelona', 'Chelsea', 'Arsenal', 'Roma'],
#           languages=['en'],)
# obj = TwitterCrawler('D:/somefile', 5)
# obj.filter_by_params(words=['Champions League','Bayern','Real Madrid', 'Barcelona', 'Chelsea', 'Arsenal', 'Roma'],
#                       langs=['en'])