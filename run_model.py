__author__='ShubhamTripathi'

import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_fscore_support, classification_report
import re
from pandas import DataFrame as df
from nltk import word_tokenize as wt
import sys

NUM_ID_TEST = 1198
NUM_ID_TRAIN = 3594


class getData:
    main_frame = df()
    age_frame = df()
    main_frame_test = df()
    age_frame_test = df()
    def readData(self, filename, mode = 'train'):
        print 'Reading Data'
        with open("%s" %filename,'r') as fp:
            file = fp.readlines()
        id = []
        time = []
        feat_L1 = []
        feat_L2 = []
        feat_L3 = []
        feat_L4 = []
        feat_L5 = []
        feat_L6 = []
        feat_L7 = []
        feat_L8 = []
        feat_L9 = []
        feat_L10 = []
        feat_L11 = []
        feat_L12 = []
        feat_L13 = []
        feat_L14 = []
        feat_L15 = []
        feat_L16 = []
        feat_L17 = []
        feat_L18 = []
        feat_L19 = []
        feat_L20 = []
        feat_L21 = []
        feat_L22 = []
        feat_L23 = []
        feat_L24 = []
        feat_L25 = []
        feat_T1 = []
        feat_T2 = []
        feat_T3 = []
        feat_T4 = []
        feat_T5 = []
        feat_T6 = []
        for i in file:
            if re.search(r'^[0-9]+', i):
                id_time_lab = re.split(r',',i)
                id.append(id_time_lab[0])
                time.append(id_time_lab[1])
                feat_L1.append(id_time_lab[2])
                feat_L2.append(id_time_lab[3])
                feat_L3.append(id_time_lab[4])
                feat_L4.append(id_time_lab[5])
                feat_L5.append(id_time_lab[6])
                feat_L6.append(id_time_lab[7])
                feat_L7.append(id_time_lab[8])
                feat_L8.append(id_time_lab[9])
                feat_L9.append(id_time_lab[10])
                feat_L10.append(id_time_lab[11])
                feat_L11.append(id_time_lab[12])
                feat_L12.append(id_time_lab[13])
                feat_L13.append(id_time_lab[14])
                feat_L14.append(id_time_lab[15])
                feat_L15.append(id_time_lab[16])
                feat_L16.append(id_time_lab[17])
                feat_L17.append(id_time_lab[18])
                feat_L18.append(id_time_lab[19])
                feat_L19.append(id_time_lab[20])
                feat_L20.append(id_time_lab[21])
                feat_L21.append(id_time_lab[22])
                feat_L22.append(id_time_lab[23])
                feat_L23.append(id_time_lab[24])
                feat_L24.append(id_time_lab[25])
                feat_L25.append(id_time_lab[26])
        if mode == 'train':
            self.main_frame['id'] = id
            self.main_frame['time'] = time
            self.main_frame['feat_L1'] = feat_L1
            self.main_frame['feat_L2'] = feat_L2
            self.main_frame['feat_L3'] = feat_L3
            self.main_frame['feat_L4'] = feat_L4
            self.main_frame['feat_L5'] = feat_L5
            self.main_frame['feat_L6'] = feat_L6
            self.main_frame['feat_L7'] = feat_L7
            self.main_frame['feat_L8'] = feat_L8
            self.main_frame['feat_L9'] = feat_L9
            self.main_frame['feat_L10'] = feat_L10
            self.main_frame['feat_L11'] = feat_L11
            self.main_frame['feat_L12'] = feat_L12
            self.main_frame['feat_L13'] = feat_L13
            self.main_frame['feat_L14'] = feat_L14
            self.main_frame['feat_L15'] = feat_L15
            self.main_frame['feat_L16'] = feat_L16
            self.main_frame['feat_L17'] = feat_L17
            self.main_frame['feat_L18'] = feat_L18
            self.main_frame['feat_L19'] = feat_L19
            self.main_frame['feat_L20'] = feat_L20
            self.main_frame['feat_L21'] = feat_L21
            self.main_frame['feat_L22'] = feat_L22
            self.main_frame['feat_L23'] = feat_L23
            self.main_frame['feat_L24'] = feat_L24
            self.main_frame['feat_L25'] = feat_L25
        elif mode == 'test':
            self.main_frame_test['id'] = id
            self.main_frame_test['time'] = time
            self.main_frame_test['feat_L1'] = feat_L1
            self.main_frame_test['feat_L2'] = feat_L2
            self.main_frame_test['feat_L3'] = feat_L3
            self.main_frame_test['feat_L4'] = feat_L4
            self.main_frame_test['feat_L5'] = feat_L5
            self.main_frame_test['feat_L6'] = feat_L6
            self.main_frame_test['feat_L7'] = feat_L7
            self.main_frame_test['feat_L8'] = feat_L8
            self.main_frame_test['feat_L9'] = feat_L9
            self.main_frame_test['feat_L10'] = feat_L10
            self.main_frame_test['feat_L11'] = feat_L11
            self.main_frame_test['feat_L12'] = feat_L12
            self.main_frame_test['feat_L13'] = feat_L13
            self.main_frame_test['feat_L14'] = feat_L14
            self.main_frame_test['feat_L15'] = feat_L15
            self.main_frame_test['feat_L16'] = feat_L16
            self.main_frame_test['feat_L17'] = feat_L17
            self.main_frame_test['feat_L18'] = feat_L18
            self.main_frame_test['feat_L19'] = feat_L19
            self.main_frame_test['feat_L20'] = feat_L20
            self.main_frame_test['feat_L21'] = feat_L21
            self.main_frame_test['feat_L22'] = feat_L22
            self.main_frame_test['feat_L23'] = feat_L23
            self.main_frame_test['feat_L24'] = feat_L24
            self.main_frame_test['feat_L25'] = feat_L25

    def readVitals(self, filename, mode = 'train'):
        print 'Reading Vitals'
        with open("%s" % filename, 'r') as fp:
            file = fp.readlines()
        icu_value = []
        feat_T1 = []
        feat_T2 = []
        feat_T3 = []
        feat_T4 = []
        feat_T5 = []
        feat_T6 = []
        for i in file:
            if re.search(r'^[0-9]+', i):
                id_time_vital = re.split(r',',i)
                icu_value.append(re.findall(r'^[0-9]+',id_time_vital[8])[0])
                feat_T1.append(id_time_vital[2])
                feat_T2.append(id_time_vital[3])
                feat_T3.append(id_time_vital[4])
                feat_T4.append(id_time_vital[5])
                feat_T5.append(id_time_vital[6])
                feat_T6.append(id_time_vital[7])
        if mode == 'train':
            self.main_frame['feat_T1'] = feat_T1
            self.main_frame['feat_T2'] = feat_T2
            self.main_frame['feat_T3'] = feat_T3
            self.main_frame['feat_T4'] = feat_T4
            self.main_frame['feat_T5'] = feat_T5
            self.main_frame['feat_T6'] = feat_T6
            self.main_frame['icu'] = icu_value
        elif mode == 'test':
            self.main_frame_test['feat_T1'] = feat_T1
            self.main_frame_test['feat_T2'] = feat_T2
            self.main_frame_test['feat_T3'] = feat_T3
            self.main_frame_test['feat_T4'] = feat_T4
            self.main_frame_test['feat_T5'] = feat_T5
            self.main_frame_test['feat_T6'] = feat_T6
            self.main_frame_test['icu'] = icu_value

    def readAge(self,filename, mode = 'train'):
        print 'Reading age'
        with open("%s" % filename, 'r') as fp:
            file = fp.readlines()
        id = []
        age = []
        for i in file:
            if re.search(r'^[0-9]+', i):
                id_time_lab = re.split(r',',i)
                id.append(id_time_lab[0])
                age.append(re.findall(r'^[0-9]+',id_time_lab[1])[0])
        if mode == 'train':
            self.age_frame['id'] = id
            self.age_frame['age'] = age
        elif mode == 'test':
            self.age_frame_test['id'] = id
            self.age_frame_test['age'] = age
        print 'Read Age'
    def readLabel(self,filename):
        #Read Labels too.
        print 'Reading Labels'
        with open("%s" % filename, 'r') as fp:
            file = fp.readlines()
        label = []
        for i in file:
            if re.search(r'^[0-9]+', i):
                id_time_lab = re.split(r',',i)
                label.append(re.findall(r'^[0-9]+',id_time_lab[1])[0])
        self.age_frame['label'] = label
        print 'Read label'

class features:

    def count_feat_timelab(self):

        if feat_mode == 'train':
            timecount_for_id = timeCount()
        elif feat_mode == 'test':
            timecount_for_id = timeCount_test()

        na_percent_2d_list_L1 = [];avg_2d_list_l1 = []; std_2d_list_l1 = []; rel_zero_val_diff_l1 = []; instant_val_diff_l1 = [];
        na_percent_2d_list_L2 = [];avg_2d_list_l2 = []; std_2d_list_l2 = []; rel_zero_val_diff_l2 = []; instant_val_diff_l2 = [];
        na_percent_2d_list_L3 = [];avg_2d_list_l3 = []; std_2d_list_l3 = []; rel_zero_val_diff_l3 = []; instant_val_diff_l3 = [];
        na_percent_2d_list_L4 = [];avg_2d_list_l4 = []; std_2d_list_l4 = []; rel_zero_val_diff_l4 = []; instant_val_diff_l4 = [];
        na_percent_2d_list_L5 = [];avg_2d_list_l5 = []; std_2d_list_l5 = []; rel_zero_val_diff_l5 = []; instant_val_diff_l5 = [];
        na_percent_2d_list_L6 = [];avg_2d_list_l6 = []; std_2d_list_l6 = []; rel_zero_val_diff_l6 = []; instant_val_diff_l6 = [];
        na_percent_2d_list_L7 = [];avg_2d_list_l7 = []; std_2d_list_l7 = []; rel_zero_val_diff_l7 = []; instant_val_diff_l7 = [];
        na_percent_2d_list_L8 = [];avg_2d_list_l8 = []; std_2d_list_l8 = []; rel_zero_val_diff_l8 = []; instant_val_diff_l8 = [];
        na_percent_2d_list_L9 = [];avg_2d_list_l9 = []; std_2d_list_l9 = []; rel_zero_val_diff_l9 = []; instant_val_diff_l9 = [];
        na_percent_2d_list_L10 = [];avg_2d_list_l10 = []; std_2d_list_l10 = []; rel_zero_val_diff_l10 = []; instant_val_diff_l10 = [];
        na_percent_2d_list_L11 = [];avg_2d_list_l11 = []; std_2d_list_l11 = []; rel_zero_val_diff_l11 = []; instant_val_diff_l11 = [];
        na_percent_2d_list_L12 = [];avg_2d_list_l12 = []; std_2d_list_l12 = []; rel_zero_val_diff_l12 = []; instant_val_diff_l12 = [];
        na_percent_2d_list_L13 = [];avg_2d_list_l13 = []; std_2d_list_l13 = []; rel_zero_val_diff_l13 = []; instant_val_diff_l13 = [];
        na_percent_2d_list_L14 = [];avg_2d_list_l14 = []; std_2d_list_l14 = []; rel_zero_val_diff_l14 = []; instant_val_diff_l14 = [];
        na_percent_2d_list_L15 = [];avg_2d_list_l15 = []; std_2d_list_l15 = []; rel_zero_val_diff_l15 = []; instant_val_diff_l15 = [];
        na_percent_2d_list_L16 = [];avg_2d_list_l16 = []; std_2d_list_l16 = []; rel_zero_val_diff_l16 = []; instant_val_diff_l16 = [];
        na_percent_2d_list_L17 = [];avg_2d_list_l17 = []; std_2d_list_l17 = []; rel_zero_val_diff_l17 = []; instant_val_diff_l17 = [];
        na_percent_2d_list_L18 = [];avg_2d_list_l18 = []; std_2d_list_l18 = []; rel_zero_val_diff_l18 = []; instant_val_diff_l18 = [];
        na_percent_2d_list_L19 = [];avg_2d_list_l19 = []; std_2d_list_l19 = []; rel_zero_val_diff_l19 = []; instant_val_diff_l19 = [];
        na_percent_2d_list_L20 = [];avg_2d_list_l20 = []; std_2d_list_l20 = []; rel_zero_val_diff_l20 = []; instant_val_diff_l20 = [];
        na_percent_2d_list_L21 = [];avg_2d_list_l21 = []; std_2d_list_l21 = []; rel_zero_val_diff_l21 = []; instant_val_diff_l21 = [];
        na_percent_2d_list_L22 = [];avg_2d_list_l22 = []; std_2d_list_l22 = []; rel_zero_val_diff_l22 = []; instant_val_diff_l22 = [];
        na_percent_2d_list_L23 = [];avg_2d_list_l23 = []; std_2d_list_l23 = []; rel_zero_val_diff_l23 = []; instant_val_diff_l23 = [];
        na_percent_2d_list_L24 = [];avg_2d_list_l24 = []; std_2d_list_l24 = []; rel_zero_val_diff_l24 = []; instant_val_diff_l24 = [];
        na_percent_2d_list_L25 = [];avg_2d_list_l25 = []; std_2d_list_l25 = []; rel_zero_val_diff_l25 = []; instant_val_diff_l25 = [];

        index_file = 0
        for k in timecount_for_id:

            def features_L1():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L1'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L1'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L1.append(na_percent)
                        avg_2d_list_l1.append(avg_val)
                        std_2d_list_l1.append(std_val)
                        rel_zero_val_diff_l1.append(rel_zero_val_diff)
                        instant_val_diff_l1.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L1.append(na_percent)
                        avg_2d_list_l1.append(sub_list[0])
                        std_2d_list_l1.append('0.0')
                        rel_zero_val_diff_l1.append('0.0')
                        instant_val_diff_l1.append('0.0')


                    m_index += 1
            def features_L2():
                ##feature L2##
                values_L1 =[]
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L2'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L2'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L2.append(na_percent)
                        avg_2d_list_l2.append(avg_val)
                        std_2d_list_l2.append(std_val)
                        rel_zero_val_diff_l2.append(rel_zero_val_diff)
                        instant_val_diff_l2.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L2.append(na_percent)
                        avg_2d_list_l2.append(sub_list[0])
                        std_2d_list_l2.append('0.0')
                        rel_zero_val_diff_l2.append('0.0')
                        instant_val_diff_l2.append('0.0')


                    m_index += 1
            def features_L3():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L3'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L3'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L3.append(na_percent)
                        avg_2d_list_l3.append(avg_val)
                        std_2d_list_l3.append(std_val)
                        rel_zero_val_diff_l3.append(rel_zero_val_diff)
                        instant_val_diff_l3.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L3.append(na_percent)
                        avg_2d_list_l3.append(sub_list[0])
                        std_2d_list_l3.append('0.0')
                        rel_zero_val_diff_l3.append('0.0')
                        instant_val_diff_l3.append('0.0')


                    m_index += 1
            def features_L4():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L4'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L4'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L4.append(na_percent)
                        avg_2d_list_l4.append(avg_val)
                        std_2d_list_l4.append(std_val)
                        rel_zero_val_diff_l4.append(rel_zero_val_diff)
                        instant_val_diff_l4.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L4.append(na_percent)
                        avg_2d_list_l4.append(sub_list[0])
                        std_2d_list_l4.append('0.0')
                        rel_zero_val_diff_l4.append('0.0')
                        instant_val_diff_l4.append('0.0')


                    m_index += 1
            def features_L5():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L5'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L5'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L5.append(na_percent)
                        avg_2d_list_l5.append(avg_val)
                        std_2d_list_l5.append(std_val)
                        rel_zero_val_diff_l5.append(rel_zero_val_diff)
                        instant_val_diff_l5.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L5.append(na_percent)
                        avg_2d_list_l5.append(sub_list[0])
                        std_2d_list_l5.append('0.0')
                        rel_zero_val_diff_l5.append('0.0')
                        instant_val_diff_l5.append('0.0')


                    m_index += 1
            def features_L6():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L6'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L6'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L6.append(na_percent)
                        avg_2d_list_l6.append(avg_val)
                        std_2d_list_l6.append(std_val)
                        rel_zero_val_diff_l6.append(rel_zero_val_diff)
                        instant_val_diff_l6.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L6.append(na_percent)
                        avg_2d_list_l6.append(sub_list[0])
                        std_2d_list_l6.append('0.0')
                        rel_zero_val_diff_l6.append('0.0')
                        instant_val_diff_l6.append('0.0')


                    m_index += 1
            def features_L7():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L7'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L7'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L7.append(na_percent)
                        avg_2d_list_l7.append(avg_val)
                        std_2d_list_l7.append(std_val)
                        rel_zero_val_diff_l7.append(rel_zero_val_diff)
                        instant_val_diff_l7.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L7.append(na_percent)
                        avg_2d_list_l7.append(sub_list[0])
                        std_2d_list_l7.append('0.0')
                        rel_zero_val_diff_l7.append('0.0')
                        instant_val_diff_l7.append('0.0')


                    m_index += 1
            def features_L8():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L8'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L8'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L8.append(na_percent)
                        avg_2d_list_l8.append(avg_val)
                        std_2d_list_l8.append(std_val)
                        rel_zero_val_diff_l8.append(rel_zero_val_diff)
                        instant_val_diff_l8.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L8.append(na_percent)
                        avg_2d_list_l8.append(sub_list[0])
                        std_2d_list_l8.append('0.0')
                        rel_zero_val_diff_l8.append('0.0')
                        instant_val_diff_l8.append('0.0')


                    m_index += 1
            def features_L9():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L9'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L9'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L9.append(na_percent)
                        avg_2d_list_l9.append(avg_val)
                        std_2d_list_l9.append(std_val)
                        rel_zero_val_diff_l9.append(rel_zero_val_diff)
                        instant_val_diff_l9.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L9.append(na_percent)
                        avg_2d_list_l9.append(sub_list[0])
                        std_2d_list_l9.append('0.0')
                        rel_zero_val_diff_l9.append('0.0')
                        instant_val_diff_l9.append('0.0')


                    m_index += 1
            def features_L10():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L10'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L10'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L10.append(na_percent)
                        avg_2d_list_l10.append(avg_val)
                        std_2d_list_l10.append(std_val)
                        rel_zero_val_diff_l10.append(rel_zero_val_diff)
                        instant_val_diff_l10.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L10.append(na_percent)
                        avg_2d_list_l10.append(sub_list[0])
                        std_2d_list_l10.append('0.0')
                        rel_zero_val_diff_l10.append('0.0')
                        instant_val_diff_l10.append('0.0')


                    m_index += 1
            def features_L11():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L11'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L11'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L11.append(na_percent)
                        avg_2d_list_l11.append(avg_val)
                        std_2d_list_l11.append(std_val)
                        rel_zero_val_diff_l11.append(rel_zero_val_diff)
                        instant_val_diff_l11.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L11.append(na_percent)
                        avg_2d_list_l11.append(sub_list[0])
                        std_2d_list_l11.append('0.0')
                        rel_zero_val_diff_l11.append('0.0')
                        instant_val_diff_l11.append('0.0')


                    m_index += 1
            def features_L12():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L12'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L12'][l])
                m_index = 0

                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L12.append(na_percent)
                        avg_2d_list_l12.append(avg_val)
                        std_2d_list_l12.append(std_val)
                        rel_zero_val_diff_l12.append(rel_zero_val_diff)
                        instant_val_diff_l12.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L12.append(na_percent)
                        avg_2d_list_l12.append(sub_list[0])
                        std_2d_list_l12.append('0.0')
                        rel_zero_val_diff_l12.append('0.0')
                        instant_val_diff_l12.append('0.0')


                    m_index += 1
            def features_L13():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L13'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L13'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L13.append(na_percent)
                        avg_2d_list_l13.append(avg_val)
                        std_2d_list_l13.append(std_val)
                        rel_zero_val_diff_l13.append(rel_zero_val_diff)
                        instant_val_diff_l13.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L13.append(na_percent)
                        avg_2d_list_l13.append(sub_list[0])
                        std_2d_list_l13.append('0.0')
                        rel_zero_val_diff_l13.append('0.0')
                        instant_val_diff_l13.append('0.0')


                    m_index += 1
            def features_L14():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L14'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L14'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L14.append(na_percent)
                        avg_2d_list_l14.append(avg_val)
                        std_2d_list_l14.append(std_val)
                        rel_zero_val_diff_l14.append(rel_zero_val_diff)
                        instant_val_diff_l14.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L14.append(na_percent)
                        avg_2d_list_l14.append(sub_list[0])
                        std_2d_list_l14.append('0.0')
                        rel_zero_val_diff_l14.append('0.0')
                        instant_val_diff_l14.append('0.0')


                    m_index += 1
            def features_L15():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L15'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L15'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L15.append(na_percent)
                        avg_2d_list_l15.append(avg_val)
                        std_2d_list_l15.append(std_val)
                        rel_zero_val_diff_l15.append(rel_zero_val_diff)
                        instant_val_diff_l15.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L15.append(na_percent)
                        avg_2d_list_l15.append(sub_list[0])
                        std_2d_list_l15.append('0.0')
                        rel_zero_val_diff_l15.append('0.0')
                        instant_val_diff_l15.append('0.0')


                    m_index += 1
            def features_L16():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L16'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L16'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L16.append(na_percent)
                        avg_2d_list_l16.append(avg_val)
                        std_2d_list_l16.append(std_val)
                        rel_zero_val_diff_l16.append(rel_zero_val_diff)
                        instant_val_diff_l16.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L16.append(na_percent)
                        avg_2d_list_l16.append(sub_list[0])
                        std_2d_list_l16.append('0.0')
                        rel_zero_val_diff_l16.append('0.0')
                        instant_val_diff_l16.append('0.0')


                    m_index += 1
            def features_L17():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L17'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L17'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L17.append(na_percent)
                        avg_2d_list_l17.append(avg_val)
                        std_2d_list_l17.append(std_val)
                        rel_zero_val_diff_l17.append(rel_zero_val_diff)
                        instant_val_diff_l17.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L17.append(na_percent)
                        avg_2d_list_l17.append(sub_list[0])
                        std_2d_list_l17.append('0.0')
                        rel_zero_val_diff_l17.append('0.0')
                        instant_val_diff_l17.append('0.0')


                    m_index += 1
            def features_L18():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L18'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L18'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L18.append(na_percent)
                        avg_2d_list_l18.append(avg_val)
                        std_2d_list_l18.append(std_val)
                        rel_zero_val_diff_l18.append(rel_zero_val_diff)
                        instant_val_diff_l18.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L18.append(na_percent)
                        avg_2d_list_l18.append(sub_list[0])
                        std_2d_list_l18.append('0.0')
                        rel_zero_val_diff_l18.append('0.0')
                        instant_val_diff_l18.append('0.0')


                    m_index += 1
            def features_L19():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L19'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L19'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L19.append(na_percent)
                        avg_2d_list_l19.append(avg_val)
                        std_2d_list_l19.append(std_val)
                        rel_zero_val_diff_l19.append(rel_zero_val_diff)
                        instant_val_diff_l19.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L19.append(na_percent)
                        avg_2d_list_l19.append(sub_list[0])
                        std_2d_list_l19.append('0.0')
                        rel_zero_val_diff_l19.append('0.0')
                        instant_val_diff_l19.append('0.0')


                    m_index += 1
            def features_L20():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L20'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L20'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L20.append(na_percent)
                        avg_2d_list_l20.append(avg_val)
                        std_2d_list_l20.append(std_val)
                        rel_zero_val_diff_l20.append(rel_zero_val_diff)
                        instant_val_diff_l20.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L20.append(na_percent)
                        avg_2d_list_l20.append(sub_list[0])
                        std_2d_list_l20.append('0.0')
                        rel_zero_val_diff_l20.append('0.0')
                        instant_val_diff_l20.append('0.0')


                    m_index += 1
            def features_L21():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L21'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L21'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L21.append(na_percent)
                        avg_2d_list_l21.append(avg_val)
                        std_2d_list_l21.append(std_val)
                        rel_zero_val_diff_l21.append(rel_zero_val_diff)
                        instant_val_diff_l21.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L21.append(na_percent)
                        avg_2d_list_l21.append(sub_list[0])
                        std_2d_list_l21.append('0.0')
                        rel_zero_val_diff_l21.append('0.0')
                        instant_val_diff_l21.append('0.0')


                    m_index += 1
            def features_L22():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L22'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L22'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L22.append(na_percent)
                        avg_2d_list_l22.append(avg_val)
                        std_2d_list_l22.append(std_val)
                        rel_zero_val_diff_l22.append(rel_zero_val_diff)
                        instant_val_diff_l22.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L22.append(na_percent)
                        avg_2d_list_l22.append(sub_list[0])
                        std_2d_list_l22.append('0.0')
                        rel_zero_val_diff_l22.append('0.0')
                        instant_val_diff_l22.append('0.0')


                    m_index += 1
            def features_L23():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L23'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L23'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L23.append(na_percent)
                        avg_2d_list_l23.append(avg_val)
                        std_2d_list_l23.append(std_val)
                        rel_zero_val_diff_l23.append(rel_zero_val_diff)
                        instant_val_diff_l23.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L23.append(na_percent)
                        avg_2d_list_l23.append(sub_list[0])
                        std_2d_list_l23.append('0.0')
                        rel_zero_val_diff_l23.append('0.0')
                        instant_val_diff_l23.append('0.0')


                    m_index += 1
            def features_L24():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L24'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L24'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L24.append(na_percent)
                        avg_2d_list_l24.append(avg_val)
                        std_2d_list_l24.append(std_val)
                        rel_zero_val_diff_l24.append(rel_zero_val_diff)
                        instant_val_diff_l24.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L24.append(na_percent)
                        avg_2d_list_l24.append(sub_list[0])
                        std_2d_list_l24.append('0.0')
                        rel_zero_val_diff_l24.append('0.0')
                        instant_val_diff_l24.append('0.0')


                    m_index += 1
            def features_L25():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_L25'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_L25'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_L25.append(na_percent)
                        avg_2d_list_l25.append(avg_val)
                        std_2d_list_l25.append(std_val)
                        rel_zero_val_diff_l25.append(rel_zero_val_diff)
                        instant_val_diff_l25.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_L25.append(na_percent)
                        avg_2d_list_l25.append(sub_list[0])
                        std_2d_list_l25.append('0.0')
                        rel_zero_val_diff_l25.append('0.0')
                        instant_val_diff_l25.append('0.0')


                    m_index += 1
            features_L1();features_L2();features_L3();features_L4();features_L5();features_L6();features_L7();features_L8();features_L9();features_L10();
            features_L11();features_L12();features_L13();features_L14();features_L15();features_L16();features_L17();features_L18();features_L19();features_L20();
            features_L21();features_L22();features_L23();features_L24();features_L25();
            index_file += k

        if feat_mode == 'train':
            X_feature_list.append(na_percent_2d_list_L1);X_feature_list.append(avg_2d_list_l1);X_feature_list.append(std_2d_list_l1);X_feature_list.append(rel_zero_val_diff_l1);X_feature_list.append(instant_val_diff_l1)
            X_feature_list.append(na_percent_2d_list_L2);X_feature_list.append(avg_2d_list_l2);X_feature_list.append(std_2d_list_l2);X_feature_list.append(rel_zero_val_diff_l2);X_feature_list.append(instant_val_diff_l2)
            X_feature_list.append(na_percent_2d_list_L3);X_feature_list.append(avg_2d_list_l3);X_feature_list.append(std_2d_list_l3);X_feature_list.append(rel_zero_val_diff_l3);X_feature_list.append(instant_val_diff_l3)
            X_feature_list.append(na_percent_2d_list_L4);X_feature_list.append(avg_2d_list_l4);X_feature_list.append(std_2d_list_l4);X_feature_list.append(rel_zero_val_diff_l4);X_feature_list.append(instant_val_diff_l4)
            X_feature_list.append(na_percent_2d_list_L5);X_feature_list.append(avg_2d_list_l5);X_feature_list.append(std_2d_list_l5);X_feature_list.append(rel_zero_val_diff_l5);X_feature_list.append(instant_val_diff_l5)
            X_feature_list.append(na_percent_2d_list_L6);X_feature_list.append(avg_2d_list_l6);X_feature_list.append(std_2d_list_l6);X_feature_list.append(rel_zero_val_diff_l6);X_feature_list.append(instant_val_diff_l6)
            X_feature_list.append(na_percent_2d_list_L7);X_feature_list.append(avg_2d_list_l7);X_feature_list.append(std_2d_list_l7);X_feature_list.append(rel_zero_val_diff_l7);X_feature_list.append(instant_val_diff_l7)
            X_feature_list.append(na_percent_2d_list_L8);X_feature_list.append(avg_2d_list_l8);X_feature_list.append(std_2d_list_l8);X_feature_list.append(rel_zero_val_diff_l8);X_feature_list.append(instant_val_diff_l8)
            X_feature_list.append(na_percent_2d_list_L9);X_feature_list.append(avg_2d_list_l9);X_feature_list.append(std_2d_list_l9);X_feature_list.append(rel_zero_val_diff_l9);X_feature_list.append(instant_val_diff_l9)
            X_feature_list.append(na_percent_2d_list_L10);X_feature_list.append(avg_2d_list_l10);X_feature_list.append(std_2d_list_l10);X_feature_list.append(rel_zero_val_diff_l10);X_feature_list.append(instant_val_diff_l10)
            X_feature_list.append(na_percent_2d_list_L11);X_feature_list.append(avg_2d_list_l11);X_feature_list.append(std_2d_list_l11);X_feature_list.append(rel_zero_val_diff_l11);X_feature_list.append(instant_val_diff_l11)
            X_feature_list.append(na_percent_2d_list_L12);X_feature_list.append(avg_2d_list_l12);X_feature_list.append(std_2d_list_l12);X_feature_list.append(rel_zero_val_diff_l12);X_feature_list.append(instant_val_diff_l12)
            X_feature_list.append(na_percent_2d_list_L13);X_feature_list.append(avg_2d_list_l13);X_feature_list.append(std_2d_list_l13);X_feature_list.append(rel_zero_val_diff_l13);X_feature_list.append(instant_val_diff_l13)
            X_feature_list.append(na_percent_2d_list_L14);X_feature_list.append(avg_2d_list_l14);X_feature_list.append(std_2d_list_l14);X_feature_list.append(rel_zero_val_diff_l14);X_feature_list.append(instant_val_diff_l14)
            X_feature_list.append(na_percent_2d_list_L15);X_feature_list.append(avg_2d_list_l15);X_feature_list.append(std_2d_list_l15);X_feature_list.append(rel_zero_val_diff_l15);X_feature_list.append(instant_val_diff_l15)
            X_feature_list.append(na_percent_2d_list_L16);X_feature_list.append(avg_2d_list_l16);X_feature_list.append(std_2d_list_l16);X_feature_list.append(rel_zero_val_diff_l16);X_feature_list.append(instant_val_diff_l16)
            X_feature_list.append(na_percent_2d_list_L17);X_feature_list.append(avg_2d_list_l17);X_feature_list.append(std_2d_list_l17);X_feature_list.append(rel_zero_val_diff_l17);X_feature_list.append(instant_val_diff_l17)
            X_feature_list.append(na_percent_2d_list_L18);X_feature_list.append(avg_2d_list_l18);X_feature_list.append(std_2d_list_l18);X_feature_list.append(rel_zero_val_diff_l18);X_feature_list.append(instant_val_diff_l18)
            X_feature_list.append(na_percent_2d_list_L19);X_feature_list.append(avg_2d_list_l19);X_feature_list.append(std_2d_list_l19);X_feature_list.append(rel_zero_val_diff_l19);X_feature_list.append(instant_val_diff_l19)
            X_feature_list.append(na_percent_2d_list_L20);X_feature_list.append(avg_2d_list_l20);X_feature_list.append(std_2d_list_l20);X_feature_list.append(rel_zero_val_diff_l20);X_feature_list.append(instant_val_diff_l20)
            X_feature_list.append(na_percent_2d_list_L21);X_feature_list.append(avg_2d_list_l21);X_feature_list.append(std_2d_list_l21);X_feature_list.append(rel_zero_val_diff_l21);X_feature_list.append(instant_val_diff_l21)
            X_feature_list.append(na_percent_2d_list_L22);X_feature_list.append(avg_2d_list_l22);X_feature_list.append(std_2d_list_l22);X_feature_list.append(rel_zero_val_diff_l22);X_feature_list.append(instant_val_diff_l22)
            X_feature_list.append(na_percent_2d_list_L23);X_feature_list.append(avg_2d_list_l23);X_feature_list.append(std_2d_list_l23);X_feature_list.append(rel_zero_val_diff_l23);X_feature_list.append(instant_val_diff_l23)
            X_feature_list.append(na_percent_2d_list_L24);X_feature_list.append(avg_2d_list_l24);X_feature_list.append(std_2d_list_l24);X_feature_list.append(rel_zero_val_diff_l24);X_feature_list.append(instant_val_diff_l24)
            X_feature_list.append(na_percent_2d_list_L25);X_feature_list.append(avg_2d_list_l25);X_feature_list.append(std_2d_list_l25);X_feature_list.append(rel_zero_val_diff_l25);X_feature_list.append(instant_val_diff_l25)
            print '%s, Length. Timelab appended. Training.' % len(rel_zero_val_diff_l1)
        elif feat_mode == 'test':
            X_feature_list_test.append(na_percent_2d_list_L1);X_feature_list_test.append(avg_2d_list_l1);X_feature_list_test.append(std_2d_list_l1);X_feature_list_test.append(rel_zero_val_diff_l1);X_feature_list_test.append(instant_val_diff_l1)
            X_feature_list_test.append(na_percent_2d_list_L2);X_feature_list_test.append(avg_2d_list_l2);X_feature_list_test.append(std_2d_list_l2);X_feature_list_test.append(rel_zero_val_diff_l2);X_feature_list_test.append(instant_val_diff_l2)
            X_feature_list_test.append(na_percent_2d_list_L3);X_feature_list_test.append(avg_2d_list_l3);X_feature_list_test.append(std_2d_list_l3);X_feature_list_test.append(rel_zero_val_diff_l3);X_feature_list_test.append(instant_val_diff_l3)
            X_feature_list_test.append(na_percent_2d_list_L4);X_feature_list_test.append(avg_2d_list_l4);X_feature_list_test.append(std_2d_list_l4);X_feature_list_test.append(rel_zero_val_diff_l4);X_feature_list_test.append(instant_val_diff_l4)
            X_feature_list_test.append(na_percent_2d_list_L5);X_feature_list_test.append(avg_2d_list_l5);X_feature_list_test.append(std_2d_list_l5);X_feature_list_test.append(rel_zero_val_diff_l5);X_feature_list_test.append(instant_val_diff_l5)
            X_feature_list_test.append(na_percent_2d_list_L6);X_feature_list_test.append(avg_2d_list_l6);X_feature_list_test.append(std_2d_list_l6);X_feature_list_test.append(rel_zero_val_diff_l6);X_feature_list_test.append(instant_val_diff_l6)
            X_feature_list_test.append(na_percent_2d_list_L7);X_feature_list_test.append(avg_2d_list_l7);X_feature_list_test.append(std_2d_list_l7);X_feature_list_test.append(rel_zero_val_diff_l7);X_feature_list_test.append(instant_val_diff_l7)
            X_feature_list_test.append(na_percent_2d_list_L8);X_feature_list_test.append(avg_2d_list_l8);X_feature_list_test.append(std_2d_list_l8);X_feature_list_test.append(rel_zero_val_diff_l8);X_feature_list_test.append(instant_val_diff_l8)
            X_feature_list_test.append(na_percent_2d_list_L9);X_feature_list_test.append(avg_2d_list_l9);X_feature_list_test.append(std_2d_list_l9);X_feature_list_test.append(rel_zero_val_diff_l9);X_feature_list_test.append(instant_val_diff_l9)
            X_feature_list_test.append(na_percent_2d_list_L10);X_feature_list_test.append(avg_2d_list_l10);X_feature_list_test.append(std_2d_list_l10);X_feature_list_test.append(rel_zero_val_diff_l10);X_feature_list_test.append(instant_val_diff_l10)
            X_feature_list_test.append(na_percent_2d_list_L11);X_feature_list_test.append(avg_2d_list_l11);X_feature_list_test.append(std_2d_list_l11);X_feature_list_test.append(rel_zero_val_diff_l11);X_feature_list_test.append(instant_val_diff_l11)
            X_feature_list_test.append(na_percent_2d_list_L12);X_feature_list_test.append(avg_2d_list_l12);X_feature_list_test.append(std_2d_list_l12);X_feature_list_test.append(rel_zero_val_diff_l12);X_feature_list_test.append(instant_val_diff_l12)
            X_feature_list_test.append(na_percent_2d_list_L13);X_feature_list_test.append(avg_2d_list_l13);X_feature_list_test.append(std_2d_list_l13);X_feature_list_test.append(rel_zero_val_diff_l13);X_feature_list_test.append(instant_val_diff_l13)
            X_feature_list_test.append(na_percent_2d_list_L14);X_feature_list_test.append(avg_2d_list_l14);X_feature_list_test.append(std_2d_list_l14);X_feature_list_test.append(rel_zero_val_diff_l14);X_feature_list_test.append(instant_val_diff_l14)
            X_feature_list_test.append(na_percent_2d_list_L15);X_feature_list_test.append(avg_2d_list_l15);X_feature_list_test.append(std_2d_list_l15);X_feature_list_test.append(rel_zero_val_diff_l15);X_feature_list_test.append(instant_val_diff_l15)
            X_feature_list_test.append(na_percent_2d_list_L16);X_feature_list_test.append(avg_2d_list_l16);X_feature_list_test.append(std_2d_list_l16);X_feature_list_test.append(rel_zero_val_diff_l16);X_feature_list_test.append(instant_val_diff_l16)
            X_feature_list_test.append(na_percent_2d_list_L17);X_feature_list_test.append(avg_2d_list_l17);X_feature_list_test.append(std_2d_list_l17);X_feature_list_test.append(rel_zero_val_diff_l17);X_feature_list_test.append(instant_val_diff_l17)
            X_feature_list_test.append(na_percent_2d_list_L18);X_feature_list_test.append(avg_2d_list_l18);X_feature_list_test.append(std_2d_list_l18);X_feature_list_test.append(rel_zero_val_diff_l18);X_feature_list_test.append(instant_val_diff_l18)
            X_feature_list_test.append(na_percent_2d_list_L19);X_feature_list_test.append(avg_2d_list_l19);X_feature_list_test.append(std_2d_list_l19);X_feature_list_test.append(rel_zero_val_diff_l19);X_feature_list_test.append(instant_val_diff_l19)
            X_feature_list_test.append(na_percent_2d_list_L20);X_feature_list_test.append(avg_2d_list_l20);X_feature_list_test.append(std_2d_list_l20);X_feature_list_test.append(rel_zero_val_diff_l20);X_feature_list_test.append(instant_val_diff_l20)
            X_feature_list_test.append(na_percent_2d_list_L21);X_feature_list_test.append(avg_2d_list_l21);X_feature_list_test.append(std_2d_list_l21);X_feature_list_test.append(rel_zero_val_diff_l21);X_feature_list_test.append(instant_val_diff_l21)
            X_feature_list_test.append(na_percent_2d_list_L22);X_feature_list_test.append(avg_2d_list_l22);X_feature_list_test.append(std_2d_list_l22);X_feature_list_test.append(rel_zero_val_diff_l22);X_feature_list_test.append(instant_val_diff_l22)
            X_feature_list_test.append(na_percent_2d_list_L23);X_feature_list_test.append(avg_2d_list_l23);X_feature_list_test.append(std_2d_list_l23);X_feature_list_test.append(rel_zero_val_diff_l23);X_feature_list_test.append(instant_val_diff_l23)
            X_feature_list_test.append(na_percent_2d_list_L24);X_feature_list_test.append(avg_2d_list_l24);X_feature_list_test.append(std_2d_list_l24);X_feature_list_test.append(rel_zero_val_diff_l24);X_feature_list_test.append(instant_val_diff_l24)
            X_feature_list_test.append(na_percent_2d_list_L25);X_feature_list_test.append(avg_2d_list_l25);X_feature_list_test.append(std_2d_list_l25);X_feature_list_test.append(rel_zero_val_diff_l25);X_feature_list_test.append(instant_val_diff_l25)
            print '%s, Length. Timelab appended. Testing.' % len(rel_zero_val_diff_l1)
    def count_feat_timevital(self):

        if feat_mode == 'train':
            timecount_for_id = timeCount()
        elif feat_mode == 'test':
            timecount_for_id = timeCount_test()

        na_percent_2d_list_T1 = [];avg_2d_list_t1 = []; std_2d_list_t1 = []; rel_zero_val_diff_t1 = []; instant_val_diff_t1 = [];
        na_percent_2d_list_T2 = [];avg_2d_list_t2 = []; std_2d_list_t2 = []; rel_zero_val_diff_t2 = []; instant_val_diff_t2 = [];
        na_percent_2d_list_T3 = [];avg_2d_list_t3 = []; std_2d_list_t3 = []; rel_zero_val_diff_t3 = []; instant_val_diff_t3 = [];
        na_percent_2d_list_T4 = [];avg_2d_list_t4 = []; std_2d_list_t4 = []; rel_zero_val_diff_t4 = []; instant_val_diff_t4 = [];
        na_percent_2d_list_T5 = [];avg_2d_list_t5 = []; std_2d_list_t5 = []; rel_zero_val_diff_t5 = []; instant_val_diff_t5 = [];
        na_percent_2d_list_T6 = [];avg_2d_list_t6 = []; std_2d_list_t6 = []; rel_zero_val_diff_t6 = []; instant_val_diff_t6 = [];

        index_file = 0
        for k in timecount_for_id:

            def features_T1():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_T1'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_T1'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_T1.append(na_percent)
                        avg_2d_list_t1.append(avg_val)
                        std_2d_list_t1.append(std_val)
                        rel_zero_val_diff_t1.append(rel_zero_val_diff)
                        instant_val_diff_t1.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_T1.append(na_percent)
                        avg_2d_list_t1.append(sub_list[0])
                        std_2d_list_t1.append('0.0')
                        rel_zero_val_diff_t1.append('0.0')
                        instant_val_diff_t1.append('0.0')


                    m_index += 1
            def features_T2():
                ##feature L2##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_T2'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_T2'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_T2.append(na_percent)
                        avg_2d_list_t2.append(avg_val)
                        std_2d_list_t2.append(std_val)
                        rel_zero_val_diff_t2.append(rel_zero_val_diff)
                        instant_val_diff_t2.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_T2.append(na_percent)
                        avg_2d_list_t2.append(sub_list[0])
                        std_2d_list_t2.append('0.0')
                        rel_zero_val_diff_t2.append('0.0')
                        instant_val_diff_t2.append('0.0')


                    m_index += 1
            def features_T3():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_T3'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_T3'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_T3.append(na_percent)
                        avg_2d_list_t3.append(avg_val)
                        std_2d_list_t3.append(std_val)
                        rel_zero_val_diff_t3.append(rel_zero_val_diff)
                        instant_val_diff_t3.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_T3.append(na_percent)
                        avg_2d_list_t3.append(sub_list[0])
                        std_2d_list_t3.append('0.0')
                        rel_zero_val_diff_t3.append('0.0')
                        instant_val_diff_t3.append('0.0')


                    m_index += 1
            def features_T4():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_T4'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_T4'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_T4.append(na_percent)
                        avg_2d_list_t4.append(avg_val)
                        std_2d_list_t4.append(std_val)
                        rel_zero_val_diff_t4.append(rel_zero_val_diff)
                        instant_val_diff_t4.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_T4.append(na_percent)
                        avg_2d_list_t4.append(sub_list[0])
                        std_2d_list_t4.append('0.0')
                        rel_zero_val_diff_t4.append('0.0')
                        instant_val_diff_t4.append('0.0')


                    m_index += 1
            def features_T5():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_T5'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_T5'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_T5.append(na_percent)
                        avg_2d_list_t5.append(avg_val)
                        std_2d_list_t5.append(std_val)
                        rel_zero_val_diff_t5.append(rel_zero_val_diff)
                        instant_val_diff_t5.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_T5.append(na_percent)
                        avg_2d_list_t5.append(sub_list[0])
                        std_2d_list_t5.append('0.0')
                        rel_zero_val_diff_t5.append('0.0')
                        instant_val_diff_t5.append('0.0')


                    m_index += 1
            def features_T6():
                ##feature L1##
                values_L1 = []
                for l in range(index_file,index_file+k):
                    if feat_mode == 'train':
                        values_L1.append(gd.main_frame['feat_T6'][l])
                    elif feat_mode == 'test':
                        values_L1.append(gd_test.main_frame_test['feat_T6'][l])
                m_index = 0
                for m in values_L1:
                    index_lower = m_index
                    #indexes to be used for making the features.
                    if not index_lower==0:
                        sub_list = []
                        for u in range(0,index_lower+1):
                            sub_list.append(values_L1[u])
                        ##NA_percent##
                        na_count = 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                na_count += 1
                        na_percent = float(na_count)/float(len(sub_list))

                        ##Avg value,max,min,std ##
                        sub_list_index= 0
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                sub_list[sub_list_index] = 0
                            else:
                                sub_list[sub_list_index] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[sub_list_index])[0])
                            sub_list_index += 1
                        rel_zero_val_diff = sub_list[len(sub_list)-1] - sub_list[0]
                        instant_val_diff = sub_list[len(sub_list)-1] - sub_list[len(sub_list) - 2]
                        avg_val = float(np.mean(sub_list))
                        std_val = float(np.std(sub_list))

                        na_percent_2d_list_T6.append(na_percent)
                        avg_2d_list_t6.append(avg_val)
                        std_2d_list_t6.append(std_val)
                        rel_zero_val_diff_t6.append(rel_zero_val_diff)
                        instant_val_diff_t6.append(instant_val_diff)

                    else:
                        sub_list = []
                        count_na = 0
                        sub_list.append(values_L1[index_lower])
                        for ll in sub_list:
                            if ll== 'NA' or ll.startswith('NA'):
                                count_na += 1
                                sub_list[0] = 0
                            else:
                                sub_list[0] = float(re.findall(r'[0-9]+.[0-9]+',sub_list[0])[0])

                        #binary na_percent##
                        na_percent = float(count_na)/1
                        na_percent_2d_list_T6.append(na_percent)
                        avg_2d_list_t6.append(sub_list[0])
                        std_2d_list_t6.append('0.0')
                        rel_zero_val_diff_t6.append('0.0')
                        instant_val_diff_t6.append('0.0')


                    m_index += 1
            features_T1();features_T2();features_T3();features_T4();features_T5();features_T6()
            index_file += k

        if feat_mode == 'train':
            X_feature_list.append(na_percent_2d_list_T1);X_feature_list.append(avg_2d_list_t1);X_feature_list.append(std_2d_list_t1);X_feature_list.append(rel_zero_val_diff_t1);X_feature_list.append(instant_val_diff_t1)
            X_feature_list.append(na_percent_2d_list_T2);X_feature_list.append(avg_2d_list_t2);X_feature_list.append(std_2d_list_t2);X_feature_list.append(rel_zero_val_diff_t2);X_feature_list.append(instant_val_diff_t2)
            X_feature_list.append(na_percent_2d_list_T3);X_feature_list.append(avg_2d_list_t3);X_feature_list.append(std_2d_list_t3);X_feature_list.append(rel_zero_val_diff_t3);X_feature_list.append(instant_val_diff_t3)
            X_feature_list.append(na_percent_2d_list_T4);X_feature_list.append(avg_2d_list_t4);X_feature_list.append(std_2d_list_t4);X_feature_list.append(rel_zero_val_diff_t4);X_feature_list.append(instant_val_diff_t4)
            X_feature_list.append(na_percent_2d_list_T5);X_feature_list.append(avg_2d_list_t5);X_feature_list.append(std_2d_list_t5);X_feature_list.append(rel_zero_val_diff_t5);X_feature_list.append(instant_val_diff_t5)
            X_feature_list.append(na_percent_2d_list_T6);X_feature_list.append(avg_2d_list_t6);X_feature_list.append(std_2d_list_t6);X_feature_list.append(rel_zero_val_diff_t6);X_feature_list.append(instant_val_diff_t6)
            print '%s, Length. Timevital appended. Training.' % len(rel_zero_val_diff_t1)
        elif feat_mode == 'test':
            X_feature_list_test.append(na_percent_2d_list_T1);X_feature_list_test.append(avg_2d_list_t1);X_feature_list_test.append(std_2d_list_t1);X_feature_list_test.append(rel_zero_val_diff_t1);X_feature_list_test.append(instant_val_diff_t1)
            X_feature_list_test.append(na_percent_2d_list_T2);X_feature_list_test.append(avg_2d_list_t2);X_feature_list_test.append(std_2d_list_t2);X_feature_list_test.append(rel_zero_val_diff_t2);X_feature_list_test.append(instant_val_diff_t2)
            X_feature_list_test.append(na_percent_2d_list_T3);X_feature_list_test.append(avg_2d_list_t3);X_feature_list_test.append(std_2d_list_t3);X_feature_list_test.append(rel_zero_val_diff_t3);X_feature_list_test.append(instant_val_diff_t3)
            X_feature_list_test.append(na_percent_2d_list_T4);X_feature_list_test.append(avg_2d_list_t4);X_feature_list_test.append(std_2d_list_t4);X_feature_list_test.append(rel_zero_val_diff_t4);X_feature_list_test.append(instant_val_diff_t4)
            X_feature_list_test.append(na_percent_2d_list_T5);X_feature_list_test.append(avg_2d_list_t5);X_feature_list_test.append(std_2d_list_t5);X_feature_list_test.append(rel_zero_val_diff_t5);X_feature_list_test.append(instant_val_diff_t5)
            X_feature_list_test.append(na_percent_2d_list_T6);X_feature_list_test.append(avg_2d_list_t6);X_feature_list_test.append(std_2d_list_t6);X_feature_list_test.append(rel_zero_val_diff_t6);X_feature_list_test.append(instant_val_diff_t6)
            print '%s, Length. Timevital appended. Testing.' % len(rel_zero_val_diff_t1)

    def add_age(self):
        age_list = []
        if feat_mode == 'train':
            id = add_id()
        elif feat_mode == 'test':
            id = add_id_test()
            id_index_test = 4793
        id_index = 1
        age_index = 0
        for i in id:
            if feat_mode == 'train':
                if i == id_index:
                    age_list.append(gd.age_frame['age'].ix[age_index])
                else:
                    id_index += 1
                    age_index += 1
                    age_list.append(gd.age_frame['age'].ix[age_index])
            elif feat_mode == 'test':
                if i == id_index_test:
                    age_list.append(gd_test.age_frame_test['age'].ix[age_index])
                else:
                    id_index_test += 1
                    age_index += 1
                    age_list.append(gd_test.age_frame_test['age'].ix[age_index])
        assert len(age_list)== len(id), 'Error in appending age'
        if feat_mode == 'train':
            X_feature_list.append(age_list)
        elif feat_mode == 'test':
            X_feature_list_test.append(age_list)
        print 'Age appended in X_feature_list/X_feature_list_test.'

    def add_label(self):
        id_index = 1
        label_index = 0
        label_list = []
        id = add_id()
        for i in id:
            if i == id_index:
                label_list.append(gd.age_frame['label'].ix[label_index])
            else:
                id_index += 1
                label_index += 1
                label_list.append(gd.age_frame['label'].ix[label_index])
        assert len(label_list)== len(id)
        y_label_list.append(label_list)
        print 'Label appended in y_label_list for training.'



gd = getData()
gd.readData("id_time_labs_train.csv", mode = 'train')
gd.readVitals("id_time_vitals_train.csv", mode = 'train')
gd.readAge("id_age_train.csv", mode = 'train')
gd.readLabel("id_label_train.csv")
gd_test = getData()
gd_test.readData(sys.argv[2], mode = 'test')
gd_test.readVitals(sys.argv[1], mode = 'test')
gd_test.readAge(sys.argv[3], mode = 'test')


def add_id():
    id_list= []
    for i in gd.main_frame['id']:
        id_list.append(int(i))
    return id_list

def add_id_test():
    id_list= []
    for i in gd_test.main_frame_test['id']:
        id_list.append(int(i))
    return id_list

## min,max, patient ID number ##
id = add_id()
id = list(set(id))
MIN_PATIENT_ID_NUMBER = min(id)
MAX_PATIENT_ID_NUMBER = max(id)
##---##

print 'Getting ICU values'
icu_values_test = []
for i in gd_test.main_frame_test['icu']:
    icu_values_test.append(int(i))
    

def timeCount():
    timecount_for_id = []
    id = add_id()
    id_index = 1
    timecount = 0
    for i in id:
        if i == id_index:
            timecount += 1
        else:
            timecount_for_id.append(timecount)
            timecount = 1
            id_index += 1
    timecount_for_id.append(timecount)
    return timecount_for_id

def timeCount_test():
    timecount_for_id = []
    id = add_id_test()
    id_index = 1
    timecount = 0
    for i in id:
        if i == id_index:
            timecount += 1
        else:
            timecount_for_id.append(timecount)
            timecount = 1
            id_index += 1
    timecount_for_id.append(timecount)
    return timecount_for_id

X_feature_list = []
y_label_list = []

X_feature_list_test = []




##Adding features to X_feature_list##
print 'Calling features for mode = Train'
feat_mode = 'train'
feat = features()
feat.count_feat_timelab()
feat.count_feat_timevital()
feat.add_age()
feat.add_label()
assert len(y_label_list[0])==628437, 'Error in y_label_list'

print 'Calling features for mode = Test'
feat_mode = 'test'
feat_test = features()
feat_test.count_feat_timelab()
feat_test.count_feat_timevital()
feat_test.add_age()

###---###
# X_train = np.array(X_feature_list)
# X_train = np.transpose(X_train)
# y= np.array(y_label_list[0])
# X_test = np.array(X_feature_list_test)
# X_test = np.transpose(X_test)
# print len(X_feature_list)
# print len(y_label_list)
# print X_train.shape
# print y.shape
def final_run():
    print 'Setting model with %s training features.' % len(X_feature_list)
    X_train = np.array(X_feature_list)
    X_train = np.transpose(X_train)
    y_train= np.array(y_label_list[0])
    print 'Training model.'
    clf = RandomForestClassifier(n_estimators=10,n_jobs=-1)
    clf.fit(X_train,y_train)
    print 'Setting testing parameters.'
    X_test = np.array(X_feature_list_test)
    X_test = np.transpose(X_test)

    test_df = df()
    test_final_index = []
    id_df = []
    time_df = []
    gd_test_ix = 0
    for k in gd_test.main_frame_test['icu']:
        if int(k)==1:
            id_df.append(int(gd_test.main_frame_test['id'][gd_test_ix]))
            time_df.append(int(gd_test.main_frame_test['time'][gd_test_ix]))
            test_final_index.append(gd_test_ix)
        gd_test_ix += 1
    print 'Testing.'
    X_test = X_test[test_final_index]
    y_test = clf.predict(X_test)

    print 'Generating output.csv'
    test_df['id'] = id_df
    test_df['time'] = time_df
    test_df['y_pred'] = y_test
    test_df.to_csv('output.csv', header=False, index=False)

# def cross_validate(X,y, n=4):
#     kf = KFold(len(X_feature_list[0]), n_folds=n, shuffle=True)
#     print 'Setting up KFold.'
#     for train,test in kf:
#         # clf = SVC(C=13)
#         # ids = add_id()
#         #
#         # train_index = []
#         # test_index = []
#         # print 'Getting valid indices.'
#         # print len(train),len(test)
#         # train_value_for_watching = 0
#         # for k in test:
#         #     index = 0
#         #     for j in ids:
#         #         if j==k:
#         #             test_index.append(index)
#         #         else:
#         #             train_index.append(index)
#         #         index += 1
#         #
#         #     print '%s'%train_value_for_watching
#         #
#         #     train_value_for_watching += 1
#         #
#         # # for k in test:
#         # #     index = 0
#         # #     for j in ids:
#         # #         if j==k:
#         # #             test_index.append(index)
#         # #         index += 1
#
#         #indices to predict on
#         test_final_index = []
#         for k in test:
#             if icu_values[k]==1:
#                 test_final_index.append(k)
#
#         # print 'Finding Predicted IDs index.'
#         # ##Predicted IDs##
#         # predicted_id = []
#         # for kk in test_final_index:
#         #     predicted_id.append(gd.main_frame['id'][kk])
#         #
#
#         print 'Setting up model.'
#         X_train,y_train = X[train],y[train]
#         X_test, y_test = X[test_final_index],y[test_final_index]
#
#         # final_dataframe = df()
#         # final_dataframe['id'] = predicted_id
#         # print "Training. SVC"
#         # clf.fit(X_train,y_train)
#         # print "Testing. SVC"
#         # y_pred = clf.predict(X_test)
#         # print classification_report(y_test,y_pred)
#         # final_dataframe['y_pred'] = y_pred
#
#
#         #
#         # def timeCount_new():
#         #     timecount_for_id = []
#         #     id = []
#         #     for j in final_dataframe['id']:
#         #         id.append(int(j))
#         #     id_index = 1
#         #     timecount = 0
#         #     for i in id:
#         #         if i == id_index:
#         #             timecount += 1
#         #         else:
#         #             timecount_for_id.append(timecount)
#         #             timecount = 1
#         #             id_index += 1
#         #     timecount_for_id.append(timecount)
#         #     return timecount_for_id
#         #
#         #
#         # y_test_id = []
#         # y_pred_id = []
#         #
#         #
#         # timecount_for_id_new = timeCount_new()
#         # index_cv = 0
#         # for k in timecount_for_id_new:
#         #     values_y_pred = []
#         #     for l in range(index_cv,index_cv+k):
#         #         values_y_pred.append(final_dataframe['y_pred'][l])
#         #
#         #     y_test_id.append(values_y_test[0])
#         #
#         #     print 'Checking predicted sequence'
#         #
#         #     count_zero = 0
#         #     count_one = 0
#         #     for l in values_y_pred:
#         #         if l == 0:
#         #             count_zero += 1
#         #         elif l == 1:
#         #             count_one += 1
#         #         else:
#         #             assert False,'Error in predicted values.'
#         #     if count_one >= 1:
#         #         y_pred_id.append(1)
#         #         with open('output.csv','a') as fll:
#         #             fll.write('%s,1' % final_dataframe['id'][index_cv] + "\n")
#         #     elif count_one == 0:
#         #         y_pred_id.append(0)
#         #         with open('output.csv','a') as fll:
#         #             fll.write('%s,0' % final_dataframe['id'][index_cv] + "\n")
#         #
#         #     index_cv += k
#         #
#         # print classification_report(y_test_id,y_pred_id)
#         # clf_lr = LinearRegression()
#         # print "Training. LinearRegression"
#         # clf_lr.fit(X_train,y_train)
#         # print "Testing. LinearRegression"
#         # y_pred = clf_lr.predict(X_test)
#         # print classification_report(y_test,y_pred)
#         #
#         clf_rf = RandomForestClassifier(n_estimators=10,n_jobs=-1)
#         print "Training. RF"
#         clf_rf.fit(X_train,y_train)
#         print "Testing. RF"
#         y_pred = clf_rf.predict(X_test)
#
#         print classification_report(y_test,y_pred)

if __name__ == '__main__':
    final_run()