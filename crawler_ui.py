import sys
import PyQt4
from PyQt4 import QtGui, QtCore
from crawler import TwitterCrawler
from analyzer import TwitterAnalyzer


class Window(QtGui.QMainWindow):
    def __init__(self):
        self.window_analysis = None

        #set null filenames
        self.trackWordsFileName = None
        self.langsFileName = None
        self.followsFileName = None
        self.locationsFileName = None

        super(Window, self).__init__()
        self.setGeometry(150, 150, 500, 400)
        self.setWindowTitle('Main window')

        self.default_layout()

        self.show()

    #########################
    ##### define layout #####
    #########################
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
        # ----------- New window button -----------
        btn = QtGui.QPushButton("Analysis", self)
        btn.clicked.connect(self.new_window_click)
        btn.setStyleSheet('font-size:12pt')
        btn.resize(btn.sizeHint())
        btn.move(330, 90)
        # ----------- Exit button -----------
        btn = QtGui.QPushButton("Exit", self)
        btn.clicked.connect(QtCore.QCoreApplication.instance().quit)
        btn.setStyleSheet('font-size:12pt')
        btn.resize(btn.sizeHint())
        btn.move(420, 90)
        # ----------- input WordsFilePath -----------
        lbl = QtGui.QLabel("Track words", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(20,70)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(20,90)
        btn.clicked.connect(self.show_file_dialog_words)
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
        # ----------- output FilePath -----------
        lbl = QtGui.QLabel("Raw data file name", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(220,150)
        self.txtOutFilePath = QtGui.QLineEdit(self)
        self.txtOutFilePath.resize(self.txtOutFilePath.sizeHint())
        self.txtOutFilePath.move(220,170)
        # ----------- Tweets Count -----------
        lbl = QtGui.QLabel("Tweets number", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(220,200)
        self.txtTweetsCount = QtGui.QLineEdit(self)
        self.txtTweetsCount.resize(self.txtTweetsCount.sizeHint())
        self.txtTweetsCount.move(220,220)

    ################################
    ##### show dialogs methods #####
    ################################
    def show_file_dialog_words(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.trackWordsFileName = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_langs(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.langsFileName = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_follows(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.followsFileName = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_locations(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.locationsFileName = fileName if '.txt' in fileName else fileName + '.txt'

    ####################################
    ##### read input files methods #####
    ####################################
    def extract_follows(self):
        if self.followsFileName != None:
            fileNameText = self.followsFileName
            inputFileName = fileNameText if '.txt' in fileNameText else fileNameText + '.txt'
            f = open(inputFileName, 'r')
            text = f.read()
            follows = text.split('\n')
            f.close()

            return follows
        else:
            return None

    def extract_langs(self):
        if self.langsFileName != None:
            fileNameText = self.langsFileName
            inputFileName = fileNameText if '.txt' in fileNameText else fileNameText + '.txt'
            f = open(inputFileName, 'r')
            text = f.read()
            langs = text.split('\n')
            f.close()

            return langs
        else:
            return None

    def extract_tracking_words(self):
        if self.trackWordsFileName != None:
            fileNameText = self.trackWordsFileName
            inputFileName = fileNameText if '.txt' in fileNameText else fileNameText + '.txt'
            f = open(inputFileName, 'r')
            text = f.read()
            tracking_words = text.split('\n')
            f.close()

            return tracking_words
        else:
            return None

    def extract_locations(self):
        if self.locationsFileName != None:
            fileNameText = self.locationsFileName
            inputFileName = fileNameText if '.txt' in fileNameText else fileNameText + '.txt'
            f = open(inputFileName, 'r')
            text = f.read()
            locations = text.split('\n')
            f.close()

            return locations
        else:
            return None

    ##################################
    ##### Move to analysis stage #####
    ##################################
    def new_window_click(self):
        if self.window_analysis is None:
            self.window_analysis = WindowAnalysis()
        self.window_analysis.show()

        #################################

    ##### main crawler launcher #####
    #################################
    def run_click(self):
        #read tracking words
        tracking_words = None
        tracking_words = self.extract_tracking_words()

        #read langs
        langs = None
        langs = self.extract_langs()

        #read follows
        follows = None
        follows = self.extract_follows()

        #read locations
        locations = None
        locations = self.extract_locations()

        #read output file name and tweets_count
        if self.txtOutFilePath.text() != '' and self.txtTweetsCount.text() != '' and self.trackWordsFileName != None:
            fileNameText = self.txtOutFilePath.text()
            outputFileName =  fileNameText if '.txt' in fileNameText else fileNameText + '.txt'

            tweets_count = int(self.txtTweetsCount.text())

            msg = QtGui.QMessageBox(self)
            msg.setText('Done')
            msg.show()

            #pass parameters to Crawler entity
            crawler = TwitterCrawler(filename=outputFileName, tweets_count=tweets_count)
            crawler.filter_by_params(words=tracking_words, langs=langs, follows=follows, locations=locations)
        else:
            msg = QtGui.QMessageBox(self)
            msg.setText('Please input output file name, number of tweets and choose file with tracking words')
            msg.show()

#TODO:FIX LAYOUT ADD STUFF
class WindowAnalysis(QtGui.QMainWindow):
    def __init__(self):
        super(WindowAnalysis, self).__init__()
        self.setGeometry(150, 150, 800, 500)
        self.setWindowTitle('Analysis window')

        #define files
        self.userSampleFileName = None
        self.rawDataFileName = None
        self.teamsFileName = None

        #sample size and percentage
        self.training_sample_size = None
        self.training_percentage = None

        self.layout_for_analysis()

        self.show()

    def layout_for_analysis(self):
         # ----------- run button -----------
        btn = QtGui.QPushButton("Run", self)
        btn.clicked.connect(self.run_analysis_click)
        btn.setStyleSheet('font-size:12pt')
        btn.resize(btn.sizeHint())
        btn.move(700, 50)
         # ----------- Train button -----------
        btn = QtGui.QPushButton("Train Naive Bayes classifier", self)
        btn.clicked.connect(self.train_NB_classifier)
        btn.setStyleSheet('font-size:12pt')
        btn.resize(btn.sizeHint())
        btn.move(500,50)
        # ----------- input Sample percentage -----------
        lbl = QtGui.QLabel("Training percentage (%)", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(20,130)
        self.txtPercent = QtGui.QLineEdit(self)
        self.txtPercent.resize(self.txtPercent.sizeHint())
        self.txtPercent.move(20, 150)
        # ----------- input Sample size -----------
        lbl = QtGui.QLabel("Training sample size", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(20,80)
        self.txtSize = QtGui.QLineEdit(self)
        self.txtSize.resize(self.txtSize.sizeHint())
        self.txtSize.move(20, 100)
        # ----------- input UserSampleFilePath -----------
        lbl = QtGui.QLabel("User sample file", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(220,75)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(220,95)
        btn.clicked.connect(self.show_file_dialog_user_file)
        # ----------- input TeamsFilePath -----------
        lbl = QtGui.QLabel("Teams file", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(220,135)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(220,155)
        btn.clicked.connect(self.show_file_dialog_teams_file)
        # ----------- input RawDataFilePath -----------
        lbl = QtGui.QLabel("Raw data file", self)
        lbl.setStyleSheet('font-size: 12pt;')
        lbl.resize(lbl.sizeHint())
        lbl.move(220,195)
        btn = QtGui.QPushButton("Choose file", self)
        btn.setStyleSheet('font-size: 12pt;')
        btn.resize(btn.sizeHint())
        btn.move(220,215)
        btn.clicked.connect(self.show_file_dialog_raw_data_file)

    ####################################################
    ############## show dialogs methods ################
    ####################################################
    def show_file_dialog_user_file(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.userSampleFileName = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_teams_file(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.teamsFileName = fileName if '.txt' in fileName else fileName + '.txt'

    def show_file_dialog_raw_data_file(self):
        fileName = QtGui.QFileDialog.getOpenFileName()
        self.rawDataFileName = fileName if '.txt' in fileName else fileName + '.txt'

    #TODO: FINISH MEHTOD WITH FULL ANALYSIS
    def run_analysis_click(self):
        #define file names
        user_sample_file_name = self.userSampleFileName
        raw_data_file_name = self.rawDataFileName
        teams_file_name = self.teamsFileName

        #define training sample size and percentage
        try:
            self.training_percentage = int(self.txtPercent.text())
            self.training_sample_size = int(self.txtSize.text())
        except: pass

        if (user_sample_file_name == None or user_sample_file_name == '') or \
            (raw_data_file_name == None or raw_data_file_name == '') or\
            (teams_file_name == None or teams_file_name == '') or\
            (self.training_sample_size == None or self.training_percentage == None):
            #show error message
            msg = QtGui.QMessageBox(self)
            msg.setText('Please input files: Teams, User sample, Raw data\nInput training sample size and percentage')
            msg.show()
        else:
            #perform analysis
            analyzer = TwitterAnalyzer(user_sample_file_name=user_sample_file_name, raw_data_file_name=raw_data_file_name)
            analyzer.try_parse_data()
            analyzer.make_tweets_dataframe()

            words = self.extract_mine_words(teams_file_name)
            analyzer.mine_with_words(words=words)

    def extract_mine_words(self, filename):
        file = open(filename,'r')
        text = file.read()
        words = text.split('\n')
        file.close()

        return words

    #TODO: FINISH METHOD
    def train_NB_classifier(self):
        msg = QtGui.QMessageBox(self)
        msg.setText('Done')
        msg.show()

def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()
