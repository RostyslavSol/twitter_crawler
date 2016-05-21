import sys
import os
import json
import numpy as np
from PyQt4 import QtGui, QtCore
from crawler import TwitterCrawler
from custom_plotter import CustomPlotter


class WindowNew(QtGui.QMainWindow):
    def __init__(self, print_string):
        super(WindowNew, self).__init__()
        self.setGeometry(50, 50, 500, 400)
        self.setWindowTitle('Initial clusters')

        self.layout(print_string)

        self.show()

    def layout(self, print_string):
        richTxt = QtGui.QTextEdit(self)
        richTxt.setStyleSheet('font-size: 11pt;')
        richTxt.resize(495, 395)
        richTxt.move(5,5)
        richTxt.setText(print_string)

class Window(QtGui.QMainWindow):
    def __init__(self):
        self.new_window = None

        #set null filenames
        self.track_words_filename = None
        self.langs_filename = None
        self.follows_filename = None
        self.locs_filename = None
        self.pic_filename = None

        #params for crawler
        self.terms_filename = None
        self.contexts_filename = None
        self.log_filename = None

        #UI initialization
        super(Window, self).__init__()
        self.setGeometry(10, 10, 1500, 1000)
        self.setWindowTitle('Main window')

        #visualize layout
        self.default_layout()
        self.analysis_layout()
        self.result_layout()

        self.show()

    def reset_all(self):
        #set null filenames
        self.track_words_filename = None
        self.langs_filename = None
        self.follows_filename = None
        self.locs_filename = None

        #params for crawler
        self.terms_filename = None
        self.contexts_filename = None
        self.log_filename = None

        #reset TextBoxes
        self.txtTrainingSampleSize.setText('')
        self.txtLogFilePath.setText('')
        self.txtTweetsCount.setText('')
        self.txtMinCos.setText('')
        self.txtVarPercent.setText('')
        self.richTxt.setText('')
        self.txtClusterCos.setText('')
        self.txtPicFilename.setText('')

    def read_from_json_file(self):
        try:
            #read file
            fileName = QtGui.QFileDialog.getOpenFileName()
            file = open(fileName, 'r')
            json_string = file.read()
            json_arr = json.loads(json_string)
            #form results
            text = ''
            tmp_sample_count_arr = []
            for json_obj in json_arr:
                try:
                    text += json_obj['text']
                    tmp_sample_count_arr.append(int(json_obj['cluster_index']))
                except:pass
            ratings = np.bincount(np.array(tmp_sample_count_arr))
            names_hash = {}
            for i in range(len(ratings)):
                for json_obj in json_arr:
                    if int(json_obj['cluster_index']) == i:
                        names_hash.update({str(i): json_obj['cluster_name']})
                        break
            #visualize results
            self.richTxt.setText(text)
            CustomPlotter.plot(ratings, names_hash)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            error_str = str(exc_type) + '\n' + str(fname) + '\n' + str(exc_tb.tb_lineno)
            print(error_str)

    #region Layouts
    def default_layout(self):
        # ----------- Upper label -----------
        lbl = QtGui.QLabel("Twitter clawler",self)
        lbl.setStyleSheet('font-size: 18pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(150,20)
        # ----------- Run button -----------
        btn = QtGui.QPushButton("Run crawler", self)
        btn.clicked.connect(self.run_click)
        btn.setStyleSheet('font-size:12pt')
        btn.resize(btn.sizeHint())
        btn.move(220, 90)
        # ----------- Reset button -----------
        btn = QtGui.QPushButton("Reset all", self)
        btn.clicked.connect(self.reset_all)
        btn.setStyleSheet('font-size:12pt')
        btn.resize(btn.sizeHint())
        btn.move(800, 20)
        # ----------- input WordsFilePath -----------
        lbl = QtGui.QLabel("Track words", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(20,70)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(20,90)
        btn.clicked.connect(self.show_file_dialog_track_words)
        # ----------- input LangsFilePath -----------
        lbl = QtGui.QLabel("Languages", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(20,130)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(20,150)
        btn.clicked.connect(self.show_file_dialog_langs)
        # ----------- input FollowsFilePath -----------
        lbl = QtGui.QLabel("Follows", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(20,190)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(20,210)
        btn.clicked.connect(self.show_file_dialog_follows)
        # ----------- input LocationsFilePath -----------
        lbl = QtGui.QLabel("Locations", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(20,250)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(20,270)
        btn.clicked.connect(self.show_file_dialog_locations)
        # ----------- read from JSON file -------------
        btn = QtGui.QPushButton("Read JSON file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(20,310)
        btn.clicked.connect(self.read_from_json_file)

    def analysis_layout(self):
        # ----------- input TermsFilePath -----------
        lbl = QtGui.QLabel("Terms", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(450,70)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(450,90)
        btn.clicked.connect(self.show_file_dialog_terms)
        # ----------- input ContextsFilePath -----------
        lbl = QtGui.QLabel("Contexts", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(450,130)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(450,150)
        btn.clicked.connect(self.show_file_dialog_contexts)
        # ----------- LogFilePath -----------
        lbl = QtGui.QLabel("Log file name", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(220,150)
        self.txtLogFilePath = QtGui.QLineEdit(self)
        self.txtLogFilePath.resize(self.txtLogFilePath.sizeHint())
        self.txtLogFilePath.move(220,170)
        # ----------- Tweets Count -----------
        lbl = QtGui.QLabel("Tweets number", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(220,200)
        self.txtTweetsCount = QtGui.QLineEdit(self)
        self.txtTweetsCount.resize(self.txtTweetsCount.sizeHint())
        self.txtTweetsCount.move(220,220)
        # ----------- Training sample size -----------
        lbl = QtGui.QLabel("Training sample size", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(220,250)
        self.txtTrainingSampleSize = QtGui.QLineEdit(self)
        self.txtTrainingSampleSize.resize(self.txtTweetsCount.sizeHint())
        self.txtTrainingSampleSize.move(220,270)
        # ----------- PreserveVariancePercentage -----------
        lbl = QtGui.QLabel("Variance percentage (%)", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(450,200)
        self.txtVarPercent = QtGui.QLineEdit(self)
        self.txtVarPercent.resize(self.txtVarPercent.sizeHint())
        self.txtVarPercent.move(450, 220)
        # ----------- MinCosValue -----------
        lbl = QtGui.QLabel("Min cos", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(450,250)
        self.txtMinCos = QtGui.QLineEdit(self)
        self.txtMinCos.resize(self.txtVarPercent.sizeHint())
        self.txtMinCos.move(450, 270)
        # ----------- ClusterCos -----------
        lbl = QtGui.QLabel("Max cos in cluster for NB", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(450,300)
        self.txtClusterCos = QtGui.QLineEdit(self)
        self.txtClusterCos.resize(self.txtVarPercent.sizeHint())
        self.txtClusterCos.move(450, 320)

    def result_layout(self):
        # ----------- richTxt -----------
        self.richTxt = QtGui.QTextEdit(self)
        self.richTxt.resize(600, 550)
        self.richTxt.move(700, 90)
        self.richTxt.setStyleSheet('font-size: 12pt')
        #------------ PicFilename --------
        lbl = QtGui.QLabel("Plot name",self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(220,300)
        self.txtPicFilename = QtGui.QLineEdit(self)
        self.txtPicFilename.resize(self.txtPicFilename.sizeHint())
        self.txtPicFilename.move(220,320)
    #endregion

    #region ShowDialog Helpers
    def show_file_dialog_track_words(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.track_words_filename = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_langs(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.langs_filename = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_follows(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.follows_filename = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_locations(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.locs_filename = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_terms(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.terms_filename = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_contexts(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.contexts_filename = fileName if '.txt' in fileName else fileName + '.txt'
    #endregion

    # read files method
    def extract_lines(self, filename):
        lines = None
        if filename != None:
            input_filename = filename if '.txt' in filename else filename + '.txt'
            f = open(input_filename, 'r')
            text = f.read()
            lines = text.split('\n')
            f.close()

        return lines

    def get_raw_terms_and_contexts_from_files(self, terms_filename, contexts_filename):
        if not ('.txt' in terms_filename and '.txt' in contexts_filename):
            terms_filename += '.txt'
            contexts_filename += '.txt'

        terms_file = open(terms_filename, 'r')
        text = terms_file.read()
        raw_terms = text.split('\n')
        terms_file.close()

        contexts_file = open(contexts_filename, 'r')
        text = contexts_file.read()
        raw_contexts = text.split('\n')
        contexts_file.close()
        return raw_terms, raw_contexts

    def run_click(self):
        try:
            #region set mining params
            #read tracking words
            tracking_words = None
            tracking_words = self.extract_lines(self.track_words_filename)

            #read langs
            langs = None
            langs = self.extract_lines(self.langs_filename)

            #read follows
            follows = None
            follows = self.extract_lines(self.follows_filename)

            #read locations
            locations = None
            locations = self.extract_lines(self.locs_filename)
            #endregion
            #region set crawler params
            self.log_filename = self.txtLogFilePath.text()

            preserve_var_percentage = float(self.txtVarPercent.text()) / 100.0
            min_cos_value = float(self.txtMinCos.text())
            max_cos_val_NB = float(self.txtClusterCos.text())
            tweets_count = int(self.txtTweetsCount.text())
            training_sample_size = int(self.txtTrainingSampleSize.text())
            pic_filename = self.txtPicFilename.text() if self.txtPicFilename.text() != '' else None
            #endregion
        except Exception as ex:
            self.log_filename = None
            preserve_var_percentage = None
            min_cos_value = None
            max_cos_val_NB = None
            tweets_count = None
            training_sample_size = None
            self.pic_filename = None

            msg = QtGui.QMessageBox(self)
            msg.setText(ex.args[0])
            msg.show()
        ###################################
        # pass parameters to Crawler entity
        # requires following
        # 1) terms_filename
        # 2) contexts_filename
        # 3) log_filename
        # 4) var_percentage
        # 5) min_cos
        # 6) tweets_count
        # 7) training_sample_size
        ###################################
        if self.terms_filename is not None and \
            self.contexts_filename is not None and \
            self.log_filename != '' and \
            preserve_var_percentage is not None and \
            min_cos_value is not None and \
            max_cos_val_NB is not None and \
            tweets_count is not None and \
            training_sample_size is not None and \
            self.track_words_filename is not None:

            try:
                raw_terms, raw_contexts = self.get_raw_terms_and_contexts_from_files(self.terms_filename,
                                                                                     self.contexts_filename)
                if raw_terms is None or len(raw_terms) < 1 or \
                    raw_contexts is None or len(raw_contexts) < 1:
                    raise Exception('Error: no terms or contexts read')
                crawler = TwitterCrawler(tracking_words=tracking_words,
                                         raw_terms=raw_terms,
                                         raw_contexts=raw_contexts,
                                         log_filename=self.log_filename,
                                         preserve_var_percentage=preserve_var_percentage,
                                         min_cos_val=min_cos_value,
                                         max_cos_val_NB=max_cos_val_NB,
                                         tweets_count=tweets_count,
                                         training_sample_size=training_sample_size
                                         )
                #######################################################
                init_clusters = crawler.get_init_clusters()
                init_contexts = crawler.get_init_contexts()
                cluster_names_hash = crawler.get_cluster_names_hash()
                print_string = 'INITIAL CLUSTERS:\n'
                for i in range(len(init_clusters)):
                    cluster_name = ''
                    try:
                        cluster_name = cluster_names_hash[str(i)]
                    except:
                        pass
                    print_string += 'Cluster# ' + str(i+1) + ' | ' + cluster_name + '\n'
                    for index_ in init_clusters[i]:
                        print_string += init_contexts[index_-1] + '\n'
                    print_string += '===================================\n'

                def show_new_window(_self, print_string_):
                    _self.new_window = WindowNew(print_string=print_string_)
                    _self.new_window.show()
                show_new_window(self, print_string)
                #######################################################

                crawler.filter_by_params(words=tracking_words, langs=langs, follows=follows, locations=locations)

                # visualize results
                self.richTxt.setText(crawler.get_result_text())
                CustomPlotter.plot(crawler.get_sample_counts(), cluster_names_hash, self.pic_filename, color='green')

            except Exception as ex:
                msg = QtGui.QMessageBox(self)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                error_str = str(exc_type) + '\n' + str(fname) + '\n' + str(exc_tb.tb_lineno)
                msg.setText(error_str)
                msg.show()
        else:
            msg = QtGui.QMessageBox(self)
            msg.setText('Required: Terms, Contexts, log, Percentage, Cos, Number of tweets, Training sample size and Tracking words')
            msg.show()

def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()