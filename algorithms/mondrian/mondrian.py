# coding:utf-8
"""
main module of mondrian
"""

# Implemented by Qiyuan Gong
# qiyuangong@gmail.com
# 2014-09-11

# @InProceedings{LeFevre2006,
#   Title = {Mondrian Multidimensional K-Anonymity},
#   Author = {LeFevre, Kristen and DeWitt, David J. and Ramakrishnan, Raghu},
#   Booktitle = {ICDE '06: Proceedings of the 22nd International Conference on Data Engineering},
#   Year = {2006},
#   Address = {Washington, DC, USA},
#   Pages = {25},
#   Publisher = {IEEE Computer Society},
#   Doi = {http://dx.doi.org/10.1109/ICDE.2006.101},
#   ISBN = {0-7695-2570-9},
# }

# !/usr/bin/env python
# coding=utf-8

import pdb
import time
import copy

from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI
from pyod.models.ocsvm import OCSVM
from pyod.models.sos import SOS

from .utils import cmp_value, value, merge_qi_value
from functools import cmp_to_key
from pyod.models.lof import LOF
from pyod.utils.example import visualize

# warning all these variables should be re-inited, if
# you want to run mondrian with different parameters
from ..utils.generalization.hierarchy_utilities import create_gen_hierarchy

DEBUG = False


class Partition(object):
    """
    Class for Group (or EC), which is used to keep records
    self.member: records in group
    self.low: lower point, use index to avoid negative values
    self.high: higher point, use index to avoid negative values
    self.allow: show if partition can be split on this QI
    """

    def __init__(self, data, low, high, QI_LEN):
        """
        split_tuple = (index, low, high)
        """
        self.low = list(low)
        self.high = list(high)
        self.member = data[:]
        self.allow = [1] * QI_LEN

    def add_record(self, record, dim):
        """
        add one record to member
        """
        self.member.append(record)

    def add_multiple_record(self, records, dim):
        """
        add multiple records (list) to partition
        """
        for record in records:
            self.add_record(record, dim)

    def __len__(self):
        """
        return number of records
        """
        return len(self.member)


ALLRES = []


class Mondrian:
    DEBUG = False

    def __init__(self, GL_K=0, RESULT=None, QI_RANGE=None, QI_DICT=None, QI_ORDER=None, outs=None):
        __DEBUG = False
        self.GL_K = GL_K
        self.QI_LEN = 10

        if RESULT is None:
            self.RESULT = []
        if QI_RANGE is None:
            self.QI_RANGE = []
        if QI_DICT is None:
            self.QI_DICT = []
        if QI_ORDER is None:
            self.QI_ORDER = []
        if outs is None:
            self.outs = []

    # global GL_K, RESULT, QI_LEN, QI_DICT, QI_RANGE, QI_ORDER

    def get_normalized_width(self, partition, index):
        if (DEBUG): print("get_normalized_width çagrıldı: " + "index:" + str(index))
        if DEBUG: print("gelen partition allow:")
        if DEBUG: print(partition.allow)
        """
        return Normalized width of partition
        similar to NCP
        """
        d_order = self.QI_ORDER[index]
        width = value(d_order[partition.high[index]]) - value(d_order[partition.low[index]])
        if width == self.QI_RANGE[index]:
            return 1

        if DEBUG: print("gelen genişlik")
        if DEBUG: print(width * 1.0 / self.QI_RANGE[index])
        return width * 1.0 / self.QI_RANGE[index]

    def choose_dimension(self, partition):
        """
        choose dim with largest norm_width from all attributes.
        This function can be upgraded with other distance function.
        """

        if DEBUG: print("choose_dimension cagrıldı")
        if DEBUG: print("gelen partition:")
        if DEBUG: print(partition.member)
        max_width = -1
        max_dim = -1
        for dim in range(self.QI_LEN):
            if partition.allow[dim] == 0:  # ??
                continue
            norm_width = self.get_normalized_width(partition, dim)
            if norm_width > max_width:
                max_width = norm_width
                max_dim = dim
        if max_width > 1:
            pdb.set_trace()
        return max_dim

    def frequency_set(self, partition, dim):
        """
        get the frequency_set of partition on dim
        """
        frequency = {}
        for record in partition.member:
            try:
                frequency[record[dim]] += 1
            except KeyError:
                frequency[record[dim]] = 1
        return frequency

    def find_median(self, partition, dim):
        """
        find the middle of the partition, return split_val
        """
        # use frequency set to get median
        frequency = self.frequency_set(partition, dim)
        split_val = ''
        next_val = ''
        value_list = list(frequency.keys())
        value_list.sort(key=cmp_to_key(cmp_value))
        total = sum(frequency.values())
        middle = total // 2
        if middle < self.GL_K or len(value_list) <= 1:
            try:
                return '', '', value_list[0], value_list[-1]
            except IndexError:
                return '', '', '', ''
        index = 0
        split_index = 0
        for i, qi_value in enumerate(value_list):
            index += frequency[qi_value]
            if index >= middle:
                split_val = qi_value
                split_index = i
                break
        else:
            print("Error: cannot find split_val")
        try:
            next_val = value_list[split_index + 1]
        except IndexError:
            # there is a frequency value in partition
            # which can be handle by mid_set
            # e.g.[1, 2, 3, 4, 4, 4, 4]
            next_val = split_val
        return (split_val, next_val, value_list[0], value_list[-1])

    def anonymize_strict(self, partition):
        """
        recursively partition groups until not allowable
        """

        allow_count = sum(partition.allow)
        # print("allow count = ")
        if DEBUG: print(allow_count)
        # only run allow_count times
        if allow_count == 0:
            self.RESULT.append(partition)
            return
        for index in range(allow_count):  # sütun sayısı kadar dönüyor
            # choose attrubite from domain
            dim = self.choose_dimension(partition)
            if dim == -1:
                print("Error: dim=-1")
                pdb.set_trace()
            (split_val, next_val, low, high) = self.find_median(partition, dim)
            # Update parent low and high
            if low != '':
                partition.low[dim] = self.QI_DICT[dim][low]
                partition.high[dim] = self.QI_DICT[dim][high]
            if split_val == '' or split_val == next_val:
                # cannot split
                partition.allow[dim] = 0
                continue
            # split the group from median
            mean = self.QI_DICT[dim][split_val]
            lhs_high = partition.high[:]
            rhs_low = partition.low[:]
            lhs_high[dim] = mean
            rhs_low[dim] = self.QI_DICT[dim][next_val]
            lhs = Partition([], partition.low, lhs_high, self.QI_LEN)
            rhs = Partition([], rhs_low, partition.high, self.QI_LEN)
            for record in partition.member:
                pos = self.QI_DICT[dim][record[dim]]
                if pos <= mean:
                    # lhs = [low, mean]
                    lhs.add_record(record, dim)
                else:
                    # rhs = (mean, high]
                    rhs.add_record(record, dim)
            # check is lhs and rhs satisfy k-anonymity
            if len(lhs) < self.GL_K or len(rhs) < self.GL_K:
                partition.allow[dim] = 0
                continue
            # anonymize sub-partition
            self.anonymize_strict(lhs)
            self.anonymize_strict(rhs)
            return
        self.RESULT.append(partition)

    def anonymize_relaxed(self, partition):
        """
        recursively partition groups until not allowable
        """
        if sum(partition.allow) == 0:
            # can not split
            self.RESULT.append(partition)
            return
        # choose attribute from domain
        dim = self.choose_dimension(partition)
        if dim == -1:
            print("Error: dim=-1")
            pdb.set_trace()
        # use frequency set to get median
        (split_val, next_val, low, high) = self.find_median(partition, dim)
        # Update parent low and high
        if low != '':
            partition.low[dim] = self.QI_DICT[dim][low]
            partition.high[dim] = self.QI_DICT[dim][high]
        if split_val == '':
            # cannot split
            partition.allow[dim] = 0
            self.anonymize_relaxed(partition)
            return
        # split the group from median
        mean = self.QI_DICT[dim][split_val]
        lhs_high = partition.high[:]
        rhs_low = partition.low[:]
        lhs_high[dim] = mean
        rhs_low[dim] = self.QI_DICT[dim][next_val]
        lhs = Partition([], partition.low, lhs_high, self.QI_LEN)
        rhs = Partition([], rhs_low, partition.high, self.QI_LEN)
        mid_set = []
        for record in partition.member:
            pos = self.QI_DICT[dim][record[dim]]
            if pos < mean:
                # lhs = [low, mean)
                lhs.add_record(record, dim)
            elif pos > mean:
                # rhs = (mean, high]
                rhs.add_record(record, dim)
            else:
                # mid_set keep the means
                mid_set.append(record)
        # handle records in the middle
        # these records will be divided evenly
        # between lhs and rhs, such that
        # |lhs| = |rhs| (+1 if total size is odd)
        half_size = len(partition) // 2
        for i in range(half_size - len(lhs)):
            record = mid_set.pop()
            lhs.add_record(record, dim)
        if len(mid_set) > 0:
            rhs.low[dim] = mean
            rhs.add_multiple_record(mid_set, dim)
        # It's not necessary now.
        # if len(lhs) < self.GL_K or len(rhs) < self.GL_K:
        #     print "Error: split failure"
        # anonymize sub-partition
        self.anonymize_relaxed(lhs)
        self.anonymize_relaxed(rhs)

    def init(self, data, k, QI_num=-1):
        """
        reset global variables
        """

        print("init çağırıldı.")

        if QI_num <= 0:
            self.QI_LEN = len(data[0]) - 1
            if DEBUG: print("QI_LEN(SUTUN SAYISI BU):")
            if DEBUG: print(self.QI_LEN)
        else:
            if DEBUG: print("QI_LEN(SUTUN SAYISI BU):")
            if DEBUG: print(self.QI_LEN)
            self.QI_LEN = QI_num
        self.GL_K = k
        self.RESULT = []
        # static values
        self.QI_DICT = []
        self.QI_ORDER = []
        self.QI_RANGE = []
        att_values = []
        for i in range(self.QI_LEN):
            att_values.append(set())
            self.QI_DICT.append(dict())

        for record in data:
            for i in range(self.QI_LEN):
                att_values[i].add(record[i])
        if DEBUG: print(
            "burda içinde sözlükler olan diziye verileri aktarıyor. her sütunda  ozamana kadar gelmiş tüm değerleri kaydediyor.")
        if DEBUG: print("ATT_VALUES:")
        if DEBUG: print(att_values)

        for i in range(self.QI_LEN):
            value_list = list(att_values[i])  # sözlükten liste çevirme  2 boyutlu array oldu

            value_list.sort(key=cmp_to_key(cmp_value))
            if DEBUG: print(value_list)
            self.QI_RANGE.append(value(value_list[-1]) - value(value_list[0]))
            self.QI_ORDER.append(list(value_list))
            for index, qi_value in enumerate(value_list):
                self.QI_DICT[i][qi_value] = index
        if DEBUG: print("QI_DICT(veriler ve indisleri küçükten büyüğe sırasını tutuyor):")
        if DEBUG: print(self.QI_DICT)
        if DEBUG: print("init tamamlandı")

    def outlier(self, data):
        out_index = 0

        for part in data:

            clf = COF(contamination=0.5, n_neighbors=2)

            if part.member.__len__() < 2:
                return

            clf.fit(self.prepare_for_ouitler(part.member))
            if DEBUG: print("outlier:")
            if DEBUG: print(clf.labels_)

            willdelete = []
            for i in range(clf.labels_.__len__()):

                if (clf.labels_[i] == 1):
                    willdelete.append(part.member[i])
                # if (clf.labels_[i] == 0):
                #     part.member[i] = self.reverse_for_ouitler(part.member[i])

            for i in willdelete:
                self.outs.append(i)
                part.member.remove(i)



            out_index += 1

    def prepare_for_ouitler(self, data):

        copy2 = copy.deepcopy(data)

        for row in copy2:
            del row[6]
            del row[6]
            del row[6]
            del row[6]
            del row[6]
            del row[6]
            del row[6]
            del row[6]
            del row[6]


        return copy2

    def reverse_for_ouitler(self, data):
        i = data.copy()
        i[8] = str(i[8])
        if i[9] == 0:
            i[9] = '<=50K'
        else:
            i[9] = '>50K'
        return i

    def reverse_for_all(self, data):
        copy = data.copy()
        for i in copy:
            i[8] = str(i[8])
            if i[9] == 0:
                i[9] = '<=50K'
            else:
                i[9] = '>50K'
        return copy

    def mondrian(self, data, k, relax=False, QI_num=-1, iter=1):

        global ALLRES
        """
        Main function of mondrian, return result in tuple (result, (ncp, rtime)).
        data: dataset in 2-dimensional array.
        k: k parameter for k-anonymity
        QI_num: Default -1, which exclude the last column. Othewise, [0, 1,..., QI_num - 1]
                will be anonymized, [QI_num,...] will be excluded.
        relax: determine use strict or relaxed mondrian,
        Both mondrians split partition with binary split.
        In strict mondrian, lhs and rhs have not intersection.
        But in relaxed mondrian, lhs may be have intersection with rhs.
        """

        if data.__len__() < 1:
            return None, None

        if DEBUG: print("data:")
        if DEBUG: print(data)
        self.init(data, k, QI_num)
        result = []
        data_size = len(data)
        if DEBUG: print("data_size:")
        if DEBUG: print(data_size)

        low = [0] * self.QI_LEN
        high = [(len(t) - 1) for t in self.QI_ORDER]
        if DEBUG: print("low:")
        if DEBUG: print(low)
        if DEBUG: print("high(max veri indisi gibi duruyor):")
        if DEBUG: print(high)
        whole_partition = Partition(data, low, high, self.QI_LEN)
        if DEBUG: print("partition tüm veriyi kapsıyor bu noktada")

        # begin mondrian
        start_time = time.time()
        if relax:
            # relax model
            print("relax")
            self.anonymize_relaxed(whole_partition)
        else:
            # strict model
            self.anonymize_strict(whole_partition)

        print("sonuçlar hesaplanıyor")
        rtime = float(time.time() - start_time)
        # generalization result and
        # evaluation information loss
        ncp = 0.0
        dp = 0.0
        if DEBUG: print("partition sayısı =")
        if DEBUG: print(self.RESULT.__len__())



        self.outlier(self.RESULT)

        ALLRES=ALLRES+self.RESULT

        generated = Mondrian()
        print("outs::")
        print(self.outs)
        result2, eval_result2 = generated.mondrian(copy.deepcopy(self.outs),  k, False, QI_num,iter+1)


        if result2 != None:
            result =  result +result2



        if(not iter == 1):
            dp2, ncp2 = self.genellestir(dp, ncp, result)



            if eval_result2 != None:
                dp3, ncp3 = eval_result2

                if True:
                    from decimal import Decimal

                    ncp2 /= self.QI_LEN
                    ncp2 /= data_size
                    print("Discernability Penalty=%.2E" % Decimal(str(dp2)))
                    print("size of partitions=%d" % len(self.RESULT))
                    print("K=%d" % k)
                    print("NCP = %.2f %%" % ncp2)
                    print("*"*30)

                return (result, (ncp2 + ncp3, dp2 + dp3))
            else:
                return (result,(ncp2, dp2))






        dp, ncp = self.genellestir(dp, ncp, result)
        dp4, ncp4 = eval_result2
        dp = dp + dp4

        if DEBUG: print(self.RESULT[0].member)

        ncp /= self.QI_LEN
        ncp /= data_size
        if True:
            from decimal import Decimal
            print("Discernability Penalty=%.2E" % Decimal(str(dp)))
            print("size of partitions=%d" % len(self.RESULT))
            print("K=%d" % k)
            print("NCP = %.2f %%" % ncp)
        return (result, (ncp, rtime))

    def genellestir(self,dp, ncp, result):
        ncp = 0.0
        dp = 0.0
        for partition in self.RESULT:
            rncp = 0.0
            for index in range(self.QI_LEN):
                rncp += self.get_normalized_width(partition, index)  # 0.4-1 arası geliyor hepsini toplayacak
            rncp *= len(partition)
            ncp += rncp
            dp += len(partition) ** 2
            for record in partition.member[:]:
                for index in range(self.QI_LEN):  # tüm veriyi geziyor burda
                    record[index] = merge_qi_value(self.QI_ORDER[index][partition.low[index]],
                                                   self.QI_ORDER[index][partition.high[index]])
                result.append(record)

        print("genelleştirden gelen sonuç:")
        print(dp)
        print(ncp)

        return dp, ncp
