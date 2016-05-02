# Import necessary libs
import json
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from LSA import LSA
from custom_naive_bayes import HelperForNB

#region constants
ACCESS_TOKEN = "2291093965-a08Vm2RWf7IU45tMz44Nmwb4cc4bxztHmpujqVm"
ACCESS_TOKEN_SECRET = "Ajts6p6lPDSgIoEsUM8outlHQpdH6OWqPgsW6pRCPvnwT"

API_KEY = "JmiG2HI5OyK2mhMQxne7O7W1J"
API_SECRET = "keBkKDdhkjn2BbG6JJLkTu10wszh4qpSP7pOjDTTeNGOxKc7m7"
#endregion

#############################
# requires
# 1) terms_filename + contexts_filename
# 3) log_filename
# 4) preserve_var_percentage
# 5) min_cos_value
#############################
class CustomListener(StreamListener):
    def __init__(self, terms_filename,
                        contexts_filename,
                        log_filename,
                        preserve_var_percentage,
                        min_cos_val,
                        tweets_count,
                        training_sample_size
                ):
        terms_filename = terms_filename if '.txt' in terms_filename else terms_filename + '.txt'
        contexts_filename = contexts_filename if '.txt' in contexts_filename else contexts_filename + '.txt'
        self.log_filename = log_filename if '.txt' in log_filename else log_filename + '.txt'

        #use LSA
        self.lsa_obj = LSA()
        self.lsa_obj.set_file_names(terms_filename=terms_filename,contexts_filename=contexts_filename)
        #set pars
        self.lsa_log_filename = log_filename if '.txt' in log_filename else log_filename + '.txt'
        self.lsa_var_percentage = preserve_var_percentage
        self.lsa_min_cos_val = min_cos_val

        #stop crawling
        self.tweets_count = tweets_count
        self.tweets_index = 0

        #use NB
        self.training_sample_size = training_sample_size
        self.training_sample_index = 0
        self.NB_helper = HelperForNB()
        self.NB_trained = False

    def on_data(self, raw_data):
        #parse json
        tweet_json = json.loads(raw_data)
        tweet_processed = None
        if self.training_sample_index < self.training_sample_size:
            try:
                #classify tweet with LSA
                tweet_processed_str = self.lsa_obj.apply_LSA_on_raw_data(log_file_name=self.lsa_log_filename,
                                                                        tweet_json=tweet_json,
                                                                        preserve_var_percentage=self.lsa_var_percentage,
                                                                        min_cos_value=self.lsa_min_cos_val)
                tweet_processed = json.loads(tweet_processed_str)
                if len(tweet_processed['cluster']) > 1:
                    self.training_sample_index += 1
            except:pass
        elif not self.NB_trained:
            self.NB_helper.read_sample_file(self.log_filename)
            X, Y = self.NB_helper.create_X_Y()
            self.NB_helper.fit_direct(X=X, Y=Y)
            self.NB_trained = True
        else:
            context_vector = self.lsa_obj.get_context_vector(tweet_json['text'])
            curr_cluster_index = self.NB_helper.predict_with_NB([context_vector])
            init_clusters = self.lsa_obj.get_init_clusters(self.lsa_var_percentage, self.lsa_min_cos_val)
            relevant_cluster = init_clusters[curr_cluster_index]
            ################################
            contexts = self.lsa_obj.get_contexts()
            for i in relevant_cluster:
                print(contexts[i-1])
            print(tweet_json['text'])
            print('------------------------')
            ################################


        ################################
        contexts = self.lsa_obj.get_contexts()
        if tweet_processed is not None:
            if len(tweet_processed['cluster']) > 0:
                for i in tweet_processed['cluster']:
                    print(contexts[i-1])
            print(tweet_processed['context'])
            print('************************')
        ################################

        #cycle condition
        if self.training_sample_index >= self.training_sample_size:
            self.tweets_index += 1
        if self.tweets_index >= self.tweets_count:
            return False
        else:
            return True

    def on_error(self, status_code):
        print(str(status_code))

class TwitterCrawler(object):
    def __init__(self, terms_filename,
                        contexts_filename,
                        log_filename,
                        preserve_var_percentage,
                        min_cos_val,
                        tweets_count,
                        training_sample_size
                 ):

        listnr = CustomListener(terms_filename,
                                contexts_filename,
                                log_filename,
                                preserve_var_percentage,
                                min_cos_val,
                                tweets_count,
                                training_sample_size
                                )
        auth = OAuthHandler(API_KEY, API_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        self.stream = Stream(auth, listnr)

    def filter_by_params(self, words=None, langs=None, follows=None, locations=None):
        self.stream.filter(track=words,
                           languages=langs,
                           follow=follows,
                           locations=locations)

# crawler = TwitterCrawler('LSA_pdf_test/LSA_pdf_test_terms', 'LSA_pdf_test/LSA_pdf_test_contexts','log.txt',0.2,0.8,20,10)
# crawler.filter_by_params(['computer','user','system','trees','binary','graph'],['en'])