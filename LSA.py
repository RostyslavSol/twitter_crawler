import numpy as np

class LSA(object):
    #read terms and contexts
    def __init__(self):
        self.terms = []
        self.contexts = []
        self.M = []
        pass

    def set_file_names(self, termsFilename, contextsFilename):
        if not ('.txt' in termsFilename and '.txt' in contextsFilename):
            termsFilename += '.txt'
            contextsFilename += '.txt'

        termsFile = open(termsFilename, 'r')
        text = termsFile.read()
        self.terms = text.split('\n')
        termsFile.close()

        contextsFile = open(contextsFilename, 'r')
        text = contextsFile.read()
        self.contexts = text.split('\n')
        contextsFile.close()

    #region props
    def get_terms(self):
        return self.terms
    def get_contexts(self):
        return self.contexts
    #endregion

    def apply_LSA(self, preserve_var_percentage, min_cos_value):
        if len(self.terms) == 0 or len(self.contexts) == 0:
            raise Exception()

        #region Helpers
        #TODO: get percentage from ui
        def get_reduced_dim(S, percentage):
            D = 0
            for i in range(1, len(S)):
                D += 0.5 * (S[i-1] + S[i])
            #redused dispersion is 70%
            reduced_D = percentage * D
            D = 0
            reduced_dim = 1
            for i in range(1, len(S)):
                reduced_dim += 1
                D += 0.5 * (S[i-1] + S[i])
                if D > reduced_D:
                    break

            return reduced_dim

        def inner_split(all_relations, rel_matr):
            for i in range(len(all_relations)-1):
                for j in range(i+1, len(all_relations)):
                    for v1 in all_relations[i]:
                        for v2 in all_relations[j]:
                            if v1 == v2:
                                if get_min_cos(v1, all_relations[i], rel_matr) > get_min_cos(v2, all_relations[j], rel_matr):
                                    all_relations[j].remove(v2)
                                else:
                                    all_relations[i].remove(v1)
            return all_relations

        def count_word_in_text(word, text):
            count = 0
            if word != None and text != None:
                word = word.lower()
                text = text.replace(',',' ').replace('.',' ').replace(':', ' ').replace('-',' ').lower()
                count = text.split().count(word)
            return count

        def cos(u,v):
            if not(type(u) == type(v) == np.matrixlib.defmatrix.matrix):
                raise Exception()
            return (u.T * v) / (np.sqrt(u.T * u) * np.sqrt(v.T * v))

        def get_min_cos(v1, rels, rel_matr):
            cos_s = []
            for rel in rels:
                if rel != v1:
                    try:
                        if v1-2 < 0:
                            raise Exception()
                        cos_s.append(rel_matr[v1-2][rel-1])
                    except:
                        if rel-2 < 0:
                            raise Exception()
                        cos_s.append(rel_matr[rel-2][v1-1])
            return min(cos_s)

        def clusterize(rel_matr, contexts_count, min_cos):
            all_relations = []
            contexts_range = [i+1 for i in range(1, contexts_count)]

            clusterize_recursive(rel_matr=rel_matr,
                                 all_relations=all_relations,
                                 contexts_range=contexts_range,
                                 vector_index=1,
                                 min_cos=min_cos
                                 )
            clusters = inner_split(all_relations, rel_matr)

            return clusters

        def clusterize_recursive(rel_matr,
                                 all_relations,
                                 contexts_range,
                                 vector_index,
                                 min_cos
                                 ):
            rels = [vector_index]
            for i in range(len(rel_matr) - vector_index + 1):
                if rel_matr[i + vector_index - 1][vector_index - 1] > min_cos:
                    rels.append(i + vector_index + 1)

            all_relations.append(rels)

            #substract ranges
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
                                     min_cos=min_cos
                                     )
        #endregion

        #fill matrix
        #TODO: implement entropy instead of log
        for i in range(len(self.terms)):
            self.M.append([])
            for j in range(len(self.contexts)):
                self.M[i].append(np.log(1 + count_word_in_text(self.terms[i], self.contexts[j])))

        #SVD decomposition
        #TODO: implement dimention control
        ####################################
        M = np.mat(self.M)
        T, S, D = np.linalg.svd(M)
        reduced_dim = get_reduced_dim(S, preserve_var_percentage)
        svd_reconstruction = T[:,0:reduced_dim]*np.diag(S[0:reduced_dim])*D[0:reduced_dim,:]

        #relations matrix
        rel_matr = []
        for i in range(1, len(self.contexts)):
            rel_matr.append([])
            for j in range(0, i):
                rel_matr[i-1].append(float(cos(svd_reconstruction[:,i], svd_reconstruction[:,j])))
            print(rel_matr[i-1])
        self.rel_matr = rel_matr
        ####################################

        #clusterize contexts
        #TODO: get min_cos=0.8 from interface
        clusters = clusterize(rel_matr, len(self.contexts), min_cos_value)

        return clusters


print('Test')
obj = LSA()
obj.set_file_names('LSA_pdf_test/LSA_pdf_test_terms', 'LSA_pdf_test/LSA_pdf_test_contexts')

terms = obj.get_terms()
contexts = obj.get_contexts()

clusters = obj.apply_LSA(preserve_var_percentage=0.23, min_cos_value=0.8)

print('\nThe texts\n')
for cluster in clusters:
    for v in cluster:
        print(contexts[v-1])
    print('***************')