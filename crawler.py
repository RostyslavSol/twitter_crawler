# Import necessary libs
import sys
import os
import datetime
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
                        raw_terms,
                        raw_contexts,
                        log_filename,
                        preserve_var_percentage,
                        min_cos_val,
                        max_cos_val_NB,
                        tweets_count,
                        training_sample_size
                ):
        #use LSA
        self._lsa_model = LSA(cluster_names=tracking_words, raw_terms=raw_terms, raw_contexts=raw_contexts)
        self._init_clusters = self._lsa_model.get_init_clusters(preserve_var_percentage, min_cos_val)
        self._init_contexts = self._lsa_model.get_raw_contexts()
        #set pars
        lsa_log_filename = log_filename if '.json' in log_filename else log_filename + '.json'
        self.lsa_log_file = open(lsa_log_filename, 'w')
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
        time_stamp = str(datetime.datetime.now()).replace(' ', '-').replace(':', '_')
        self._result_filename = 'result_' + time_stamp + '.json'
        self._result_file = open(self._result_filename, 'w')
        self._result_file.write('[')

        self._record_sample_counts = []
        self._record_sample_counts_LSA = []
        self._quality_cos_arr = []

    # region Public methods
    def get_result_text(self):
        file = open(self._result_filename, 'r')
        json_text = file.read()
        json_arr = json.loads(json_text)

        text = '\nCLASSIFICATION BY LSA (=) CLASSIFICATION BY NB (-) \n\n'
        for obj in json_arr:
            for prop in obj:
                text += obj[prop]

        return text

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
                tweet_processed = self._lsa_model.apply_LSA_on_raw_data(raw_data_obj=tweet_json,
                                                                        preserve_var_percentage=self.lsa_var_percentage,
                                                                        min_cos_value=self.lsa_min_cos_val
                                                                        )
                if tweet_processed is not None:
                    self.training_sample_index += 1

                    ################################################################
                    ## writting to file ############################################
                    ################################################################
                    self._record_sample_counts_LSA.append(int(tweet_processed['cluster_index']))

                    buf_text_label = 'lsa_' + str(self.training_sample_index)
                    buf_text = 'Num# ' + str(self.training_sample_index) + \
                                            ' | Cluster #' + \
                                            str(tweet_processed['cluster_index']+1) + '\n' + \
                                            tweet_json['text'] + \
                                            '\n===============================================\n'
                    self._result_file.write(json.dumps({buf_text_label: buf_text,
                                                        "cluster_index": tweet_processed['cluster_index']
                                                        }))
                    self._result_file.write(',')
                    ################################################################
            except Exception as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
        elif not self.NB_trained:
            training_sample_arr = self._lsa_model.get_training_sample()
            self.lsa_log_file.write(json.dumps(training_sample_arr))
            self.lsa_log_file.close()

            X, Y = self._naive_bayes_helper.create_X_Y(training_sample_arr)
            self._naive_bayes_helper.fit_direct(X=X, Y=Y)

            self.NB_trained = True
        else:
            try:
                #classify tweet with NB
                _context = self._lsa_model.process_text(tweet_json['text'])
                context_vector = self._lsa_model.get_context_vector(_context)
                curr_cluster_index = self._naive_bayes_helper.predict_with_NB([context_vector])[0]
                init_clusters = self._lsa_model.get_init_clusters(self.lsa_var_percentage, self.lsa_min_cos_val)
                relevant_cluster = init_clusters[curr_cluster_index]

                ################################################################
                ## forming result ##############################################
                ################################################################
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
                    self._record_sample_counts.append(curr_cluster_index)
                    self._quality_cos_arr.append((mean_ncos_arr, min_ncos_arr, max_ncos_arr))

                    buf_text_label = 'nb_' + str(self.tweets_index)
                    buf_text = 'Num# ' + str(self.tweets_index+1) + ' | Cluster #' +\
                                str(curr_cluster_index + 1) + '\n' +\
                                tweet_json['text'] + \
                                '\n\nAverage cos in cluster: ' + str(mean_ncos_arr) + \
                                '\nMin cos in cluster: ' + str(min_ncos_arr) + \
                                '\nMax cos in cluster: ' + str(max_ncos_arr) + \
                                '\n-----------------------------------------------------------------------------------------------\n'
                    self._result_file.write(json.dumps({buf_text_label: buf_text,
                                                        "cluster_index": str(curr_cluster_index)
                                                        }))
                    self._result_file.write(',')

                    self.tweets_index += 1
                ################################################################
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

        if self.tweets_index >= self.tweets_count:
            total_mean_arr = np.mean(self._quality_cos_arr, axis=0)
            total_values_text = '\n\nTotal average cos: ' + str(total_mean_arr[0]) + \
                                '\nTotal average min cos: ' + str(total_mean_arr[1]) + \
                                '\nTotal average max cos: ' + str(total_mean_arr[2]) + '\n'

            self._result_file.write(json.dumps({"total_values": total_values_text}))
            self._result_file.write(']')
            self._result_file.close()
            return False
        else:
            return True

    def on_error(self, status_code):
        print(str(status_code))

class TwitterCrawler(object):
    def __init__(self, tracking_words,
                        raw_terms,
                        raw_contexts,
                        log_filename,
                        preserve_var_percentage,
                        min_cos_val,
                        max_cos_val_NB,
                        tweets_count,
                        training_sample_size
                 ):

        self._listener = CustomListener(tracking_words,
                                        raw_terms,
                                        raw_contexts,
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

    def get_result_text(self):
        return self._listener.get_result_text()

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
