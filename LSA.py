import sys
import os
import numpy as np
import re
import json

#region constants
URL_REGEX = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
EPS = 1e-16
#endregion

######################################################################
## to use perform following steps
##1) create obj
##2) set_file_names() (to read terms and contexts)
##3) apply_LSA_on_raw_data() (for tweet json)
##4) get_init_clusters() (to get first formed clusters)
##5) get_cluster_names_hash() (to get names for first formed clusters)
######################################################################
class LSA(object):
    # read terms and contexts
    def __init__(self, cluster_names, raw_terms, raw_contexts):
        #LSA items
        self._cluster_names = cluster_names
        self._terms = [term.lower() for term in raw_terms]
        self._raw_contexts = raw_contexts
        self._contexts = [self.process_text(context) for context in raw_contexts]

        self._M = []
        self._init_clusters = None
        self._cluster_names_hash = {}

        #sample holder
        self._training_sample_arr = []
        self._overfitting_control_arr = []

    # region Helper methods

    # region Private methods
    def _define_cluster_names(self):
        if self._init_clusters is not None:
            contexts = self.get_contexts()
            tmp_init_clusters = [el for el in self._init_clusters]

            for name in self._cluster_names:
                processed_name = name.lower()
                for context in contexts:
                    if processed_name in context:
                        for cluster in tmp_init_clusters:
                            context_index = contexts.index(context)+1
                            if context_index in cluster:
                                cluster_index = tmp_init_clusters.index(cluster)
                                self._cluster_names_hash.update({str(cluster_index): name})
                                break
                        break
        else:
            raise Exception('Init clusters are None')

    def _fill_M(self, terms, contexts):
        self._M = []
        for i in range(len(terms)):
            self._M.append([])
            for j in range(len(contexts)):
                self._M[i].append(np.log(1 + self.count_word_in_text(terms[i], contexts[j])))
    # endregion

    # region Public methods
    def get_overfitting_control_arr(self):
        return self._overfitting_control_arr.copy()

    def get_training_sample(self):
        if len(self._training_sample_arr) > 1:
            return self._training_sample_arr.copy()
        else:
            return None

    def get_cluster_names_hash(self):
        return self._cluster_names_hash.copy()

    def get_terms(self):
        return self._terms.copy()

    def get_contexts(self):
        return self._contexts.copy()

    def get_raw_contexts(self):
        return self._raw_contexts.copy()

    def get_context_vector(self, context):
        if len(self._terms) == 0 or len(self._contexts) == 0:
            raise Exception('Error: empty terms or contexts')
        context_vector = []
        for i in range(len(self._terms)):
            context_vector.append(self.count_word_in_text(self._terms[i], context))

        return context_vector

    def process_text(self, text):
        #remove links
        try:
            pattern = URL_REGEX
            pattern_obj = re.compile(pattern=pattern, flags=re.MULTILINE)
            text = pattern_obj.sub('', text)
        except Exception as ex:
            print(ex.args)

        #remove symbols
        text = text.replace(',', ' ') \
            .replace('.', ' ') \
            .replace(':', ' ') \
            .replace('-', ' ') \
            .replace('!', ' ') \
            .replace('?', ' ') \
            .replace('\n',' ')\
            .replace("'", '')\
            .replace('"', '')\
            .replace('#', '')\
            .replace('@', '')\
            .replace('&gt;', '')\
            .replace('&amp;', '')\
            .replace("'s", '')\
            .lower()

        return text

    def count_word_in_text(self, word, text):
        count = 0
        if word != None and text != None:
            word = word.lower()
            text = self.process_text(text)

            count = text.split().count(word)
        return count

    def get_init_clusters(self, preserve_var_percentage, min_cos_value):
        if self._init_clusters is None:
            init_terms = self.get_terms()
            init_contexts = self.get_contexts()
            self._init_clusters = self._apply_LSA(init_terms,
                                                  init_contexts,
                                                  preserve_var_percentage,
                                                  min_cos_value
                                                  )
            self._define_cluster_names()
            self._overfitting_control_arr = [0 for i in self._init_clusters]
        return self._init_clusters.copy()
    # endregion

    # endregion

    #LSA itself
    # private method
    def _apply_LSA(self, terms, contexts, preserve_var_percentage, min_cos_value):
        if len(terms) == 0 or len(contexts) == 0:
            raise Exception('terms or contexts empty')

        # region Helpers
        def get_reduced_dim(S, percentage):
            D = 0
            for i in range(1, len(S)):
                D += 0.5 * (S[i - 1] + S[i])
            # redused dispersion is 70%
            reduced_D = percentage * D
            D = 0
            reduced_dim = 1
            for i in range(1, len(S)):
                reduced_dim += 1
                D += 0.5 * (S[i - 1] + S[i])
                if D > reduced_D:
                    break

            return reduced_dim

        def inner_split(all_relations, rel_matr):
            for i in range(len(all_relations) - 1):
                for j in range(i + 1, len(all_relations)):
                    for v1 in all_relations[i]:
                        for v2 in all_relations[j]:
                            if v1 == v2:
                                if get_min_cos(v1, all_relations[i], rel_matr) > get_min_cos(v2, all_relations[j],
                                                                                             rel_matr):
                                    all_relations[j].remove(v2)
                                else:
                                    all_relations[i].remove(v1)
            return all_relations

        def cos(u, v):
            if not (type(u) == type(v) == np.matrixlib.defmatrix.matrix):
                raise Exception('u v not vectors')

            norm_u = np.sqrt(u.T * u)
            norm_v = np.sqrt(v.T * v)
            if norm_u > EPS and norm_v > EPS:
                return (u.T * v) / (norm_u * norm_v)
            else:
                return 0

        def get_min_cos(v1, rels, rel_matr):
            cos_s = []
            for rel in rels:
                if rel != v1:
                    try:
                        if v1 - 2 < 0:
                            raise Exception('unknown exception in get_min_cos')
                        cos_s.append(rel_matr[v1 - 2][rel - 1])
                    except:
                        if rel - 2 < 0:
                            raise Exception('unknown exception in get_min_cos')
                        cos_s.append(rel_matr[rel - 2][v1 - 1])
            return min(cos_s)

        def clusterize(rel_matr, contexts_count, min_cos_value):
            all_relations = []
            contexts_range = [i + 1 for i in range(1, contexts_count)]

            clusterize_recursive(rel_matr=rel_matr,
                                 all_relations=all_relations,
                                 contexts_range=contexts_range,
                                 vector_index=1,
                                 min_cos_value=min_cos_value
                                 )
            clusters = inner_split(all_relations, rel_matr)

            return clusters

        def clusterize_recursive(rel_matr,
                                 all_relations,
                                 contexts_range,
                                 vector_index,
                                 min_cos_value
                                 ):
            rels = [vector_index]
            for i in range(len(rel_matr) - vector_index + 1):
                if rel_matr[i + vector_index - 1][vector_index - 1] > min_cos_value:
                    rels.append(i + vector_index + 1)

            all_relations.append(rels)

            # substract ranges
            all_relations_joined = [all_relations[i][j]
                                    for i in range(len(all_relations))
                                    for j in range(len(all_relations[i]))]
            least_range = [i if i not in all_relations_joined else -1 for i in contexts_range]
            least_range = list(filter(lambda e: e != -1, least_range))
            if len(least_range) == 0:
                return
            else:
                clusterize_recursive(rel_matr=rel_matr,
                                     all_relations=all_relations,
                                     contexts_range=contexts_range,
                                     vector_index=least_range[0],
                                     min_cos_value=min_cos_value
                                     )

        # endregion

        # fill matrix and preserve it on class level
        self._fill_M(terms=terms, contexts=contexts)

        # SVD decomposition
        M = np.mat(self._M)

        T, S, D = np.linalg.svd(M)
        reduced_dim = get_reduced_dim(S, preserve_var_percentage)
        svd_reconstruction = T[:, 0:reduced_dim] * np.diag(S[0:reduced_dim]) * D[0:reduced_dim, :]

        # relations matrix
        rel_matr = []
        for i in range(1, len(contexts)):
            rel_matr.append([])
            for j in range(0, i):
                rel_matr[i - 1].append(float(cos(svd_reconstruction[:, i], svd_reconstruction[:, j])))
            #print(rel_matr[i - 1])
        self.rel_matr = rel_matr

        # clusterize contexts
        clusters = clusterize(rel_matr, len(contexts), min_cos_value)

        return clusters

    #working with raw_data in json format
    # public method
    def apply_LSA_on_raw_data(self, raw_data_obj, preserve_var_percentage, min_cos_value, sample_slice):
        json_obj = None
        try:
            #remove characters and lowercase the text
            tweet_text = self.process_text(raw_data_obj['text'])
            terms = self.get_terms()
            tmp_contexts = self.get_contexts()
            #perform analysis
            if len(terms) > 0 and len(tmp_contexts) > 0:
                #add new context
                tmp_contexts.append(tweet_text)

                clusters = self._apply_LSA(terms=terms,
                                           contexts=tmp_contexts,
                                           preserve_var_percentage=preserve_var_percentage,
                                           min_cos_value=min_cos_value
                                           )
                #define cluster of last context his index is len(contexts) (the new one)
                new_context_index = tmp_contexts.index(tweet_text) + 1
                vector = self.get_context_vector(tweet_text)

                for cluster in clusters:
                    if new_context_index in cluster:
                        cluster.remove(new_context_index)

                        #write to log file
                        init_clusters = self.get_init_clusters(preserve_var_percentage, min_cos_value)
                        if cluster in init_clusters:
                            json_obj = {
                                        "cluster_index": init_clusters.index(cluster),
                                        "cluster": cluster,
                                        "context": tweet_text,
                                        "context_vector": vector
                                        }
                            if self._overfitting_control_arr[init_clusters.index(cluster)] < sample_slice:
                                self._overfitting_control_arr[init_clusters.index(cluster)] += 1
                                self._training_sample_arr.append(json_obj)
                                print(self._overfitting_control_arr)
            else:
                raise Exception('Empty terms or contexts apply_LSA_to_raw_data')
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            error_str = str(exc_type) + '\n' + str(fname) + '\n' + str(exc_tb.tb_lineno)
            print(error_str)

        return json_obj
