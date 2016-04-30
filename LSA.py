import numpy as np
import re
import json

#region constants
URL_REGEX = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
EPS = 1e-16
#endregion

class LSA(object):
    # read terms and contexts
    def __init__(self):
        #LSA items
        self.terms = []
        self.contexts = []
        self.M = []

    # region Helper methods
    def set_file_names(self, termsFilename, contextsFilename):
        if not ('.txt' in termsFilename and '.txt' in contextsFilename):
            termsFilename += '.txt'
            contextsFilename += '.txt'

        termsFile = open(termsFilename, 'r')
        text = termsFile.read()
        self.terms = text.split('\n')
        self.terms = [term.lower() for term in self.terms]
        termsFile.close()

        contextsFile = open(contextsFilename, 'r')
        text = contextsFile.read()
        self.contexts = text.split('\n')
        self.contexts = [self.process_text(context) for context in self.contexts]
        contextsFile.close()

    def get_terms(self):
        return self.terms

    def get_contexts(self):
        return self.contexts

    def get_context_vector(self, context):
        if len(self.M) == 0 or len(self.contexts) == 0:
            raise Exception('Empty matrix M')
        vector = np.mat(self.M)[:,self.contexts.index(context)].T.tolist()[0]
        vector = [int(-1 + np.exp(v)) for v in vector]
        return vector

    def add_new_context_to_analyze(self, context):
        context = self.process_text(context)
        self.contexts.append(context)

    def remove_new_context_to_analyze(self):
        self.contexts.remove(self.contexts[-1])

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
            .lower()

        return text

    def count_word_in_text(self, word, text):
        count = 0
        if word != None and text != None:
            word = word.lower()
            text = self.process_text(text)

            count = text.split().count(word)
        return count
    # endregion

    #LSA itself
    def apply_LSA(self, preserve_var_percentage, min_cos_value):
        if len(self.terms) == 0 or len(self.contexts) == 0:
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

        # fill matrix
        # TODO: implement entropy instead of log
        self.M = []
        for i in range(len(self.terms)):
            self.M.append([])
            for j in range(len(self.contexts)):
                self.M[i].append(np.log(1 + self.count_word_in_text(self.terms[i], self.contexts[j])))

        # SVD decomposition
        M = np.mat(self.M)
        T, S, D = np.linalg.svd(M)
        reduced_dim = get_reduced_dim(S, preserve_var_percentage)
        svd_reconstruction = T[:, 0:reduced_dim] * np.diag(S[0:reduced_dim]) * D[0:reduced_dim, :]

        # relations matrix
        rel_matr = []
        for i in range(1, len(self.contexts)):
            rel_matr.append([])
            for j in range(0, i):
                rel_matr[i - 1].append(float(cos(svd_reconstruction[:, i], svd_reconstruction[:, j])))
            print(rel_matr[i - 1])
        self.rel_matr = rel_matr

        # clusterize contexts
        clusters = clusterize(rel_matr, len(self.contexts), min_cos_value)

        return clusters

    #working with raw_data in json format
    def apply_LSA_on_raw_data(self, raw_data_filename, target_filename, preserve_var_percentage, min_cos_value):
        raw_data_filename = raw_data_filename if '.txt' in raw_data_filename else raw_data_filename + '.txt'
        target_filename = target_filename if '.txt' in target_filename else target_filename + '.txt'

        raw_data_file = open(raw_data_filename, 'r')
        raw_data_text = raw_data_file.read()
        raw_data_lines = raw_data_text.split('\n')

        target_file = open(target_filename, 'w')
        for line in raw_data_lines:
            try:
                tweet = json.loads(line)
                tweet_text = self.process_text(tweet['text'])

                #perform analysis
                if len(self.terms) > 0 and len(self.contexts) > 0:
                    self.add_new_context_to_analyze(tweet_text)

                    contexts = self.get_contexts()

                    clusters = self.apply_LSA(preserve_var_percentage=preserve_var_percentage,
                                                min_cos_value=min_cos_value)
                    #define cluster of last context his index is len(contexts) (the new one)
                    new_context_index = len(contexts)
                    vector = self.get_context_vector(contexts[-1])

                    for cluster in clusters:
                        if new_context_index in cluster:
                            cluster.remove(new_context_index)

                            json_str = '{"cluster_index":'+\
                                       str(clusters.index(cluster))+\
                                       ',"cluster":'+\
                                       str(cluster)+\
                                       ',"context":"'+\
                                       tweet_text+\
                                       '","context_vector":'+\
                                       str(vector)+'}'
                            target_file.write(json_str + '\n')
                            break

                    self.remove_new_context_to_analyze()
                else:
                    raise Exception('Error in analysis apply_LSA_to_raw_data')
            except Exception as ex:
                print(ex.args)
                continue
        target_file.close()
        raw_data_file.close()

#################################
## to use perform following steps
##1) create obj
##2) set_file_names to read terms and contexts
##3) add new context to analyze
##4) apply LSA
##5) print contexts by clusters
#################################

#region Test area
print('Test')
obj = LSA()
obj.set_file_names('LSA_pdf_test/LSA_pdf_test_terms', 'LSA_pdf_test/LSA_pdf_test_contexts')
obj.apply_LSA_on_raw_data('raw_data', 'target_result', 0.2, 0.8)
#
# contexts = obj.get_contexts()
#
# clusters = obj.apply_LSA(preserve_var_percentage=0.20, min_cos_value=0.8)
# print(clusters)
# print(len(clusters))
#
# for cluster in clusters:
#     #print cluster of last context (the new one)
#     if len(contexts) in cluster:
#         for context_index in cluster:
#             print(contexts[context_index-1])

# print('\nThe texts\n')
# for cluster in clusters:
#     for v in cluster:
#         print(contexts[v - 1])
#     print('***************')
#endregion