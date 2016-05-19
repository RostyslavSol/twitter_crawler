# Import necessary libs
import sys
import os
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
# TwitterCrawler is public class
# CustomListener is private class
# TwitterCrawler requires
# 1) tracking_words (as names for clusters)
# 2) terms_filename + contexts_filename
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
        self._lsa_model = LSA(tracking_words)
        self._lsa_model.set_file_names(terms_filename=terms_filename, contexts_filename=contexts_filename)
        self._init_clusters = self._lsa_model.get_init_clusters(preserve_var_percentage, min_cos_val)
        self._init_contexts = self._lsa_model.get_raw_contexts()
        #set pars
        self.lsa_log_filename = log_filename if '.txt' in log_filename else log_filename + '.txt'
        self.lsa_var_percentage = preserve_var_percentage
        self.lsa_min_cos_val = min_cos_val
        self.max_cos_val_NB = max_cos_val_NB

        #stop crawling
        self.tweets_count = tweets_count
        self.tweets_index = 0

        #use NB
        self.training_sample_size = training_sample_size
        self.training_sample_index = 0
        self._naive_bayes_helper = HelperForNB()
        self.NB_trained = False

        #result
        self._result_str = '\nCLASSIFICATION BY LSA\n\n'
        self._record_sample_counts = []
        self._record_sample_counts_LSA = []
        self._quality_cos_arr = []

    # region Public methods
    def get_result_str(self):
        return self._result_str

    def get_cluster_names_hash(self):
        return self._lsa_model.get_cluster_names_hash()

    def get_record_sample_counts(self):
        return self._record_sample_counts.copy()

    def get_record_sample_counts_LSA(self):
        return self._record_sample_counts_LSA.copy()

    def get_init_clusters(self):
        return self._init_clusters.copy()

    def get_init_contexts(self):
        return self._init_contexts.copy()

    # endregion

    def _ncos(self, v1, v2):
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
                tweet_processed_str = self._lsa_model.apply_LSA_on_raw_data(log_file_name=self.lsa_log_filename,
                                                                            tweet_json=tweet_json,
                                                                            preserve_var_percentage=self.lsa_var_percentage,
                                                                            min_cos_value=self.lsa_min_cos_val)
                if tweet_processed_str is not None:
                    self.training_sample_index += 1

                    tweet_processed = json.loads(tweet_processed_str)

                    ################################################################
                    self._record_sample_counts_LSA.append(int(tweet_processed['cluster_index']))
                    self._result_str += 'Num# ' + str(self.training_sample_index) + \
                                       ' | cluster #' + \
                                        str(tweet_processed['cluster_index']+1) + '\n'
                    self._result_str += tweet_json['text'] + \
                        '\n===============================================\n'
                    ################################################################
            except Exception as ex:
                print(ex.args[0])
        elif not self.NB_trained:
            self._naive_bayes_helper.read_sample_file(self.log_filename)
            X, Y = self._naive_bayes_helper.create_X_Y()
            self._naive_bayes_helper.fit_direct(X=X, Y=Y)
            self.NB_trained = True

            self._result_str += '\nCLASSIFICATION BY NB\n\n'
        else:
            try:
                #classify tweet with NB
                _context = self._lsa_model.process_text(tweet_json['text'])
                context_vector = self._lsa_model.get_context_vector(_context)
                curr_cluster_index = self._naive_bayes_helper.predict_with_NB([context_vector])
                init_clusters = self._lsa_model.get_init_clusters(self.lsa_var_percentage, self.lsa_min_cos_val)
                relevant_cluster = init_clusters[curr_cluster_index]
                ################################
                #add quality control
                contexts = self._lsa_model.get_contexts()
                ncos_arr = []
                for rc_index in relevant_cluster:
                    context_in_cluster = contexts[rc_index-1]
                    tmp_vector = self._lsa_model.get_context_vector(context_in_cluster)
                    ncos_arr.append(self._ncos(context_vector, tmp_vector))
                #record results
                mean_ncos_arr = np.mean(ncos_arr)
                min_ncos_arr = min(ncos_arr)
                max_ncos_arr = max(ncos_arr)

                #cycle condition
                if max_ncos_arr > self.max_cos_val_NB and self.NB_trained:
                    self._record_sample_counts.append(curr_cluster_index[0])
                    self._quality_cos_arr.append((mean_ncos_arr, min_ncos_arr, max_ncos_arr))

                    self._result_str += 'Num# ' + str(self.tweets_index) + ' | cluster #' + str(curr_cluster_index[0] + 1) + '\n'
                    self._result_str += tweet_json['text'] + \
                        '\n\nAverage cos in cluster: ' + str(mean_ncos_arr) + \
                        '\nMin cos in cluster: ' + str(min_ncos_arr) + \
                        '\nMax cos in cluster: ' + str(max_ncos_arr) + \
                        '\n-----------------------------------------------------------------------------------------------\n'
                    self.tweets_index += 1
                ################################
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

        if self.tweets_index >= self.tweets_count:
            total_mean_arr = np.mean(self._quality_cos_arr, axis=0)
            self._result_str += '\n\n Total average cos: ' + str(total_mean_arr[0]) + \
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

        self._listener = CustomListener(tracking_words,
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
        self.stream = Stream(auth, self._listener)

    def get_cluster_names_hash(self):
        return self._listener.get_cluster_names_hash()

    def get_result_str(self):
        return self._listener.get_result_str()

    def get_init_clusters(self):
        return self._listener.get_init_clusters()

    def get_init_contexts(self):
        return self._listener.get_init_contexts()

    def get_sample_counts(self):
        record_sample_counts = self._listener.get_record_sample_counts()
        np_arr = np.array(record_sample_counts)
        sample_counts = np.bincount(np_arr).tolist()
        return sample_counts

    def get_sample_counts_LSA(self):
        record_sample_counts = self._listener.get_record_sample_counts_LSA()
        np_arr = np.array(record_sample_counts)
        sample_counts = np.bincount(np_arr).tolist()
        return sample_counts

    def filter_by_params(self, words=None, langs=None, follows=None, locations=None):
        self.stream.filter(track=words,
                           languages=langs,
                           follow=follows,
                           locations=locations)
