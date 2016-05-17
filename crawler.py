# Import necessary libs
import json
import numpy as np
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from LSA import LSA, EPS
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
# 6) tweets_count
# 7) training_sample_size
#############################
class CustomListener(StreamListener):
    def __init__(self, tracking_words,
                        terms_filename,
                        contexts_filename,
                        log_filename,
                        preserve_var_percentage,
                        min_cos_val,
                        max_cos_val_NB,
                        tweets_count,
                        training_sample_size
                ):
        terms_filename = terms_filename if '.txt' in terms_filename else terms_filename + '.txt'
        contexts_filename = contexts_filename if '.txt' in contexts_filename else contexts_filename + '.txt'
        self.log_filename = log_filename if '.txt' in log_filename else log_filename + '.txt'

        #use LSA
        self.lsa_obj = LSA(tracking_words)
        self.lsa_obj.set_file_names(terms_filename=terms_filename,contexts_filename=contexts_filename)
        #set pars
        self.lsa_log_filename = log_filename if '.txt' in log_filename else log_filename + '.txt'
        self.lsa_var_percentage = preserve_var_percentage
        self.lsa_min_cos_val = min_cos_val
        self.max_cos_val_NB = max_cos_val_NB
        self.init_clusters = self.lsa_obj.get_init_clusters(preserve_var_percentage, min_cos_val)
        self.init_contexts = self.lsa_obj.get_raw_contexts()

        #stop crawling
        self.tweets_count = tweets_count
        self.tweets_index = 0

        #use NB
        self.training_sample_size = training_sample_size
        self.training_sample_index = 0
        self.NB_helper = HelperForNB()
        self.NB_trained = False

        #result
        self.result_str = '\nCLASSIFICATION BY LSA\n\n'
        self.record_sample_counts = []
        self.quality_cos_arr = []

    def get_result_str(self):
        return self.result_str

    def get_cluster_names_hash(self):
        return self.lsa_obj.get_cluster_names_hash()

    def get_record_sample_counts(self):
        return self.record_sample_counts

    def ncos(self, v1, v2):
        _v1 = np.mat(v1)
        _v2 = np.mat(v2)
        norm_v1 = np.sqrt(_v1*_v1.T)
        norm_v2 = np.sqrt(_v2*_v2.T)

        if norm_v1 > EPS and norm_v2 > EPS:
            return float(_v1*_v2.T / norm_v1 / norm_v2)
        else:
            return 0

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
                self.training_sample_index += 1

                ################################################################
                contexts = self.lsa_obj.get_contexts()
                self.result_str += 'Num# ' + str(self.training_sample_index) + \
                                   ' | cluster #' + \
                                   str(tweet_processed['cluster_index']+1) + '\n'
                self.result_str += tweet_json['text'] + \
                    '\n===============================================\n'
                ################################################################
            except Exception as ex:
                print(ex.args[0])
        elif not self.NB_trained:
            self.NB_helper.read_sample_file(self.log_filename)
            X, Y = self.NB_helper.create_X_Y()
            self.NB_helper.fit_direct(X=X, Y=Y)
            self.NB_trained = True

            self.result_str += '\nCLASSIFICATION BY NB\n\n'
        else:
            #classify tweet with NB
            _context = self.lsa_obj.process_text(tweet_json['text'])
            context_vector = self.lsa_obj.get_context_vector(_context)
            curr_cluster_index = self.NB_helper.predict_with_NB([context_vector])
            init_clusters = self.lsa_obj.get_init_clusters(self.lsa_var_percentage, self.lsa_min_cos_val)
            relevant_cluster = init_clusters[curr_cluster_index]
            ################################
            #add quality control
            contexts = self.lsa_obj.get_contexts()
            ncos_arr = []
            for rc_index in relevant_cluster:
                context_in_cluster = contexts[rc_index-1]
                tmp_vector = self.lsa_obj.get_context_vector(context_in_cluster)
                ncos_arr.append(self.ncos(context_vector, tmp_vector))
            #record results
            mean_ncos_arr = np.mean(ncos_arr)
            min_ncos_arr = min(ncos_arr)
            max_ncos_arr = max(ncos_arr)

            #cycle condition
            if max_ncos_arr > self.max_cos_val_NB and self.NB_trained:
                self.record_sample_counts.append(curr_cluster_index[0])
                self.quality_cos_arr.append((mean_ncos_arr, min_ncos_arr, max_ncos_arr))

                self.result_str += 'Num# ' + str(self.tweets_index) + ' | cluster #' + str(curr_cluster_index[0]+1) + '\n'
                self.result_str += tweet_json['text'] + \
                    '\n\nAverage cos in cluster: ' + str(mean_ncos_arr) + \
                    '\nMin cos in cluster: ' + str(min_ncos_arr) + \
                    '\nMax cos in cluster: ' + str(max_ncos_arr) + \
                    '\n-----------------------------------------------------------------------------------------------\n'
                self.tweets_index += 1
            ################################

        if self.tweets_index >= self.tweets_count:
            total_mean_arr = np.mean(self.quality_cos_arr, axis=0)
            self.result_str += '\n\n Total average cos: ' + str(total_mean_arr[0]) + \
                                '\n Total average min cos: ' + str(total_mean_arr[1]) + \
                                '\n Total average max cos: ' + str(total_mean_arr[2])
            return False
        else:
            return True

    def on_error(self, status_code):
        print(str(status_code))

class TwitterCrawler(object):
    def __init__(self, tracking_words,
                        terms_filename,
                        contexts_filename,
                        log_filename,
                        preserve_var_percentage,
                        min_cos_val,
                        max_cos_val_NB,
                        tweets_count,
                        training_sample_size
                 ):

        self.listener = CustomListener(tracking_words,
                                terms_filename,
                                contexts_filename,
                                log_filename,
                                preserve_var_percentage,
                                min_cos_val,
                                max_cos_val_NB,
                                tweets_count,
                                training_sample_size
                                )
        auth = OAuthHandler(API_KEY, API_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        self.stream = Stream(auth, self.listener)

    #region Helper methods
    def get_cluster_names_hash(self):
        return self.listener.get_cluster_names_hash()

    def get_result_str(self):
        return self.listener.get_result_str()

    def get_init_clusters(self):
        return self.listener.init_clusters

    def get_init_contexts(self):
        return self.listener.init_contexts

    def get_sample_counts(self):
        record_sample_counts = self.listener.get_record_sample_counts()
        np_arr = np.array(record_sample_counts)
        sample_counts = np.bincount(np_arr).tolist()
        return sample_counts
    #endregion

    def filter_by_params(self, words=None, langs=None, follows=None, locations=None):
        self.stream.filter(track=words,
                           languages=langs,
                           follow=follows,
                           locations=locations)

# crawler = TwitterCrawler('LSA_pdf_test/LSA_pdf_test_terms', 'LSA_pdf_test/LSA_pdf_test_contexts','log.txt',0.2,0.8,20,10)
# crawler.filter_by_params(['computer','user','system','trees','binary','graph'],['en'])
