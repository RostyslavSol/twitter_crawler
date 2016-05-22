# Import necessary libs
import sys
import os
import time
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

        #initialize sample control
        self._overfitting_control_arr = [0 for i in self._init_clusters]
        self._sample_slice = int(training_sample_size / len(self._init_clusters)) + 1

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
        self._result_filename = 'result/result_' + time_stamp + '.json'
        self._result_file = open(self._result_filename, 'w')
        self._result_file.write('[')

        self._poisson_flow_intensities = [0 for i in self._init_clusters]
        self._flow_time_start = None
        self._intensities_set_flag = False
        self._quality_cos_arr = []

    # region Public methods
    def get_result_text(self):
        file = open(self._result_filename, 'r')
        json_text = file.read()
        json_arr = json.loads(json_text)

        text = '\nCLASSIFICATION BY LSA (=) CLASSIFICATION BY NB (-) \n\n'
        for json_obj in json_arr:
            text += json_obj['text'] + \
            '\n-----------------------\n'

        return text

    def get_cluster_names_hash(self):
        return self._lsa_model.get_cluster_names_hash()

    def get_init_clusters(self):
        return self._init_clusters.copy()

    def get_init_contexts(self):
        return self._init_contexts.copy()

    def get_sample_counts(self):
        if self.tweets_index >= self.tweets_count:
            return self._overfitting_control_arr.copy()
        else:
            return None

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
        if self.training_sample_index < self.training_sample_size:
            try:
                #classify tweet with LSA
                tweet_processed = self._lsa_model.apply_LSA_on_raw_data(raw_data_obj=tweet_json,
                                                                        preserve_var_percentage=self.lsa_var_percentage,
                                                                        min_cos_value=self.lsa_min_cos_val,
                                                                        sample_slice=self._sample_slice
                                                                        )
                overfitting_control_arr = self._lsa_model.get_overfitting_control_arr()
                if tweet_processed is not None and \
                    overfitting_control_arr[int(tweet_processed['cluster_index'])] < self._sample_slice:
                    #inc index
                    self.training_sample_index = sum(overfitting_control_arr)

                    ################################################################
                    ## writting to file ############################################
                    ################################################################
                    buf_text = 'Num# ' + str(self.training_sample_index) + \
                                            ' | Cluster #' + \
                                            str(tweet_processed['cluster_index']+1) + '\n' + \
                                            tweet_json['text']
                    cluster_names_hash = self.get_cluster_names_hash()
                    curr_cluster_name = cluster_names_hash[str(tweet_processed['cluster_index'])]

                    self._result_file.write(json.dumps({"text": buf_text,
                                                        "cluster_index": tweet_processed['cluster_index'],
                                                        "cluster_name": curr_cluster_name
                                                        }))
                    self._result_file.write(',')
                    ################################################################
            except Exception as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
        elif not self.NB_trained:
            training_sample_arr = self._lsa_model.get_training_sample()
            self._overfitting_control_arr = self._lsa_model.get_overfitting_control_arr()

            self.lsa_log_file.write(json.dumps(training_sample_arr))
            self.lsa_log_file.close()

            X, Y = self._naive_bayes_helper.create_X_Y(training_sample_arr)
            self._naive_bayes_helper.fit_direct(X=X, Y=Y)

            self.NB_trained = True
            self._flow_time_start = time.time()
        elif not self._intensities_set_flag:
            #classify tweet with NB
            _context = self._lsa_model.process_text(tweet_json['text'])
            context_vector = self._lsa_model.get_context_vector(_context)
            curr_cluster_index = self._naive_bayes_helper.predict_with_NB([context_vector])[0]

            if self._poisson_flow_intensities[curr_cluster_index] == 0:
                fin_time = time.time()
                download_time = fin_time - self._flow_time_start
                self._poisson_flow_intensities[curr_cluster_index] = float(1 / download_time)

            if not(0 in self._poisson_flow_intensities):
                self._intensities_set_flag = True
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
                max_ncos_arr = max(ncos_arr)

                #cycle condition
                if max_ncos_arr > self.max_cos_val_NB and self.NB_trained:
                    self._overfitting_control_arr[curr_cluster_index] += 1

                    self._quality_cos_arr.append((mean_ncos_arr, max_ncos_arr))

                    buf_text = 'Num# ' + str(self.training_sample_index + self.tweets_index + 1) + ' | Cluster #' +\
                                str(curr_cluster_index + 1) + '\n' +\
                                tweet_json['text']
                    cluster_names_hash = self.get_cluster_names_hash()
                    curr_cluster_name = cluster_names_hash[str(curr_cluster_index)]

                    self._result_file.write(json.dumps({"text": buf_text,
                                                        "cluster_index": str(curr_cluster_index),
                                                        "cluster_name": curr_cluster_name,
                                                        'avg_cos': str(mean_ncos_arr),
                                                        'max_cos': str(max_ncos_arr)
                                                        }))
                    self._result_file.write(',')

                    self.tweets_index += 1
                ################################################################
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

        if self.tweets_index >= self.tweets_count:
            flow_time_fin = time.time()
            t = flow_time_fin - self._flow_time_start
            poisson_means = [int(lambda_i*t) for lambda_i in self._poisson_flow_intensities]
            poisson_probs = [(lambda_i*t)**n_i / np.math.factorial(n_i) * np.exp(-lambda_i*t)
                                     for lambda_i, n_i in zip(self._poisson_flow_intensities, poisson_means)]

            total_mean_arr = np.mean(self._quality_cos_arr, axis=0)
            total_values_text = '\n\nTotal average cos: ' + str(total_mean_arr[0]) + \
                                '\nTotal average max cos: ' + str(total_mean_arr[1]) + \
                                '\nPoisson flow intensities: ' + str(["{0:.5} ".format(l)
                                                                      for l in self._poisson_flow_intensities]) + \
                                '\nTime elapsed (sec): ' + str(t)

            self._result_file.write(json.dumps({"text": total_values_text}))
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
        self._stream = Stream(auth, self._listener)
        self._filtering_done = False

    def get_cluster_names_hash(self):
        return self._listener.get_cluster_names_hash()

    def get_result_text(self):
        if self._filtering_done:
            return self._listener.get_result_text()
        else:
            return None

    def get_ratings_json(self):
        json_str = ''
        if self._filtering_done:
            json_arr = []
            ratings = self.get_sample_counts()
            names = self.get_cluster_names_hash()
            for i in range(len(names)):
                json_arr.append({names[str(i)]: str(ratings[i])})
            json_str = json.dumps(json_arr)
        return json_str

    def get_init_clusters(self):
        return self._listener.get_init_clusters()

    def get_init_contexts(self):
        return self._listener.get_init_contexts()

    def get_sample_counts(self):
        if self._filtering_done:
            return self._listener.get_sample_counts()
        else:
            return None

    def filter_by_params(self, words=None, langs=None, follows=None, locations=None):
        self._stream.filter(track=words,
                            languages=langs,
                            follow=follows,
                            locations=locations)
        self._filtering_done = True
