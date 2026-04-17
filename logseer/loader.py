import os
import sys
import re
import numpy as np


class Loader:

    def __init__(self):
        self.stored_data = []
        self.nb_errors = 0

    def progress(self, i, end_val, bar_length=50):
        percent = float(i) / end_val
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write(f"\rProcessing... [{hashes + spaces}] {int(percent * 100)}% ")
        sys.stdout.flush()

    def loadfiles(self, DATA_DIR, fromid=0, toid=6000):
        print('*** Loading Files ***')
        efiles, sfiles = [], []
        for name in sorted(os.listdir(DATA_DIR)):
            path = os.path.join(DATA_DIR, name)
            if os.path.isdir(path):
                if name == 'error':
                    tmpfiles = efiles
                elif name == 'success':
                    tmpfiles = sfiles
                else:
                    continue

                endval = len(os.listdir(path)) - 1
                for i, dname in enumerate(sorted(os.listdir(path))):
                    if dname.find('a') != -1:
                        continue
                    if int(dname) < fromid or int(dname) > toid:
                        continue
                    self.progress(i, endval)
                    dpath = os.path.join(path, dname)
                    dfiles = []
                    for fname in sorted(os.listdir(dpath)):
                        fpath = os.path.join(dpath, fname)
                        with open(fpath, encoding='utf-8') as f:
                            dfiles.append(f.read())
                    tmpfiles.append([dfiles, dname])
                print()

        return sfiles, efiles

    def gen_text_label(self, files, label_id, numchar=3000, multiple=1):
        print('*** Generating Data ***')
        text_label = []
        endval = len(files) - 1
        num_k = 0
        num_n = 0
        for i, item in enumerate(files):
            filegroup = item[0]
            dname = item[1]
            t = []
            self.progress(i, endval)
            for file in filegroup:
                start_index = len(file) - numchar
                index = file.find('\n', start_index)
                if index == -1:
                    index = start_index
                str_to_add = file[index:]

                index = file.find('Kernel of Type:', 0, 200)
                ktype = ''
                if index != -1:
                    num_k = num_k + 1
                    if file.find('UBE', index+16, index+26) != -1:
                        ktype = 'UBE'
                    elif file.find('SECURITY', index+16, index+26) != -1:
                        ktype = 'SECURITY'
                    elif file.find('WORK FLOW', index+16, index+26) != -1:
                        ktype = 'WORKFLOW'
                    elif file.find('CALLOBJ', index+16, index+26) != -1:
                        ktype = 'CALLOBJ'
                    elif file.find('METADATA', index+16, index+26) != -1:
                        ktype = 'METADATA'
                    elif file.find('MANAGEMENT', index+16, index+26) != -1:
                        ktype = 'MANAGEMENT'
                    elif file.find('TEXTSEARCH', index+16, index+26) != -1:
                        ktype = 'TEXTSEARCH'
                    elif file.find('SAW', index+16, index+26) != -1:
                        ktype = 'SAW'
                    elif file.find('PACKAGE', index+16, index+26) != -1:
                        ktype = 'PACKAGE'
                    elif file.find('QUEUE', index+16, index+26) != -1:
                        ktype = 'QUEUE'
                    elif file.find('XML', index+16, index+26) != -1:
                        ktype = 'XML'
                elif file.find('jdenet_n> registered', 0, 200) != -1:
                    num_n = num_n + 1
                    ktype = 'JDENET'

                t.append(str_to_add + ' ***' + ktype + 'LOG*** ')
            for rep in range(multiple):
                np.random.shuffle(t)
                text_label.append([''.join(t), label_id, dname])
                if label_id == 1:
                    self.nb_errors = self.nb_errors + 1
        print()
        print('Found %s kernel logs.' % num_k)
        print('Found %s jdenet logs.' % num_n)
        self.clean(text_label)

        return text_label

    def clean(self, text_label):
        print('*** Cleaning Data ***')
        endval = len(text_label) - 1
        for i, item in enumerate(text_label):
            self.progress(i, endval)
            t = re.sub(r'[\n\r\t]', ' ', item[0])
            t = re.sub(r'[(),]', ' ', t)
            # Date-time
            t = re.sub(r'(^|\s)[A-Z][a-z][a-z] [A-Z][a-z][a-z][ ]{1,2}[0-9]{1,2} [0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]+(\s|$)', ' ', t)
            t = re.sub(r'(^|\s)[A-Z][a-z][a-z][ ]{1,2}[0-9]{1,2} [0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]+(\s|$)', ' ', t)
            t = re.sub(r'(^|\s)[A-Z][a-z][a-z] [A-Z][a-z][a-z][ ]{1,2}[0-9]{1,2} [0-9][0-9]:[0-9][0-9]:[0-9][0-9](\s|$)', ' ', t)
            t = re.sub(r'(^|\s)[0-9]+:[0-9]+:[0-9]+\.[0-9]+(\s|$)', ' ', t)
            # Directory name
            t = re.sub(r'/slot/[A-Za-z0-9/]+/', '', t)
            # IP address
            t = re.sub(r'=[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+[,]*(\s|$)', '=ip ', t)
            t = re.sub(r'<[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+>', '<ip>', t)
            # JDE message IDs
            t = re.sub(r'(^|\s)msgId[=\-][0-9]+[\.]*(\s|$)', 'msgId=num ', t)
            t = re.sub(r'(^|\s)msgPort[=\-][0-9]+[\.]*(\s|$)', 'msgPort=num ', t)
            t = re.sub(r'(^|\s)reqKrnl[=\-][0-9]+[\.]*(\s|$)', 'reqKrnl=num ', t)
            t = re.sub(r'(^|\s)reqKrnl[=\-][0-9]+[\.]*(\s|$)', 'resKrnl=num ', t)
            t = re.sub(r'(^|\s)reqNet[=\-][0-9]+[\.]*(\s|$)', 'reqNet=num ', t)
            t = re.sub(r'(^|\s)resNet[=\-][0-9]+[\.]*(\s|$)', 'resNet=num ', t)
            t = re.sub(r'(^|\s)maxrows=[0-9]+(\s|$)', ')maxrows=num  ', t)
            t = re.sub(r'(^|\s)fetched=[0-9]+(\s|$)', 'fetched=num  ', t)
            t = re.sub(r'<Krnl[0-9]+ReqQ>', '<Krnl_num_ReqQ>', t)
            t = re.sub(r'(^|\s)conn=[0-9a-f]{8}(\s|$)', 'conn=hex  ', t)
            t = re.sub(r'(^|\s)requ=[0-9a-f]{8}(\s|$)', 'requ=hex  ', t)
            # Network errors
            t = re.sub(r'le_net_error [0-9][0-9]:<.+?> <.*?>', 'le_net_error num:', t)
            # Other numbers
            t = re.sub(r'(^|\s)[0-9]+[\.:,/-]*(\s|$)', ' num ', t)
            t = re.sub(r'(^|\s)[0-9]+[/][0-9\-]+(\s|$)', ' num ', t)
            t = re.sub(r'(^|\s)0x[0-9a-f]{8}[\.:,/]*(\s|$)', ' 0xhex  ', t)
            t = re.sub(r'(^|\s)[0-9a-f]{8}[\.:,/]*(\s|$)', ' hex  ', t)
            t = re.sub(r'=[0-9a-f]{8}[\.:,/)]*(\s|$)', '=hex ', t)
            t = re.sub(r'(^|\s)[0-9]+(\s|$)', ' num ', t)
            t = re.sub(r'(^|\s)[-=\.](\s|$)', ' ', t)
            t = re.sub(r'\s+', ' ', t)
            item[0] = t
        print()

    def load(self, DATA_DIR, numchar=3000, toid=6000):
        sfiles, efiles = self.loadfiles(DATA_DIR, toid=toid)
        etext_label = self.gen_text_label(efiles, 1, numchar=numchar, multiple=1)
        stext_label = self.gen_text_label(sfiles, 0, numchar=numchar, multiple=1)
        self.stored_data = stext_label + etext_label

    def getdata(self, DATA_DIR, TEST_ERRORNUM=10, SUCCESS_LOG_RATIO=99, SUCCESS_LOG_RATIO_TEST=12.4, force_reload=False, numchar=3000, toid=6000):
        if len(self.stored_data) == 0 or force_reload:
            self.load(DATA_DIR, numchar=numchar, toid=toid)

        np.random.shuffle(self.stored_data)
        nb_test_success = int(TEST_ERRORNUM * SUCCESS_LOG_RATIO_TEST)
        nb_success = int(self.nb_errors * SUCCESS_LOG_RATIO)
        nb_test_error = TEST_ERRORNUM
        texts, labels, test_texts, test_labels = [], [], [], []
        text_labels = []
        test_text_labels = []
        all_test_list = []

        for j in self.stored_data:
            if j[1] == 1:
                if nb_test_error > 0 and j[2] not in all_test_list:
                    test_text_labels.append(j)
                    all_test_list.append(j[2])
                    nb_test_error = nb_test_error - 1
                    continue
                else:
                    text_labels.append(j)
                    continue
            else:
                if nb_test_success > 0 and j[2] not in all_test_list:
                    test_text_labels.append(j)
                    nb_test_success = nb_test_success - 1
                    all_test_list.append(j[2])
                    continue
                elif nb_success > 0:
                    text_labels.append(j)
                    nb_success = nb_success - 1

        for vrow in test_text_labels:
            for row in list(text_labels):
                if vrow[2] == row[2]:
                    text_labels.remove(row)

        with open('datalist.log', 'a', encoding='utf-8') as f:
            for row in test_text_labels:
                test_texts.append(row[0])
                test_labels.append(row[1])
                f.write(row[2] + '\n')
            for row in text_labels:
                texts.append(row[0])
                labels.append(row[1])
                f.write('train' + row[2] + '\n')

        return texts, labels, test_texts, test_labels
