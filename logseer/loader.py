import os
import sys
import re
import numpy as np


class Loader:
    """Generic log loader. Subclass and override extract_file_tag() and
    clean_domain() to add system-specific behaviour."""

    def __init__(self):
        self.stored_data = []
        self.nb_errors = 0

    def progress(self, i, end_val, bar_length=50):
        percent = float(i) / end_val if end_val > 0 else 1.0
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write(f"\rProcessing... [{hashes + spaces}] {int(percent * 100)}% ")
        sys.stdout.flush()

    def loadfiles(self, DATA_DIR, fromid=0, toid=6000):
        print('*** Loading Files ***')
        success_file_groups, error_file_groups = [], []
        for name in sorted(os.listdir(DATA_DIR)):
            if name.startswith('.'):
                continue
            path = os.path.join(DATA_DIR, name)
            if os.path.isdir(path):
                if name == 'error':
                    file_groups = error_file_groups
                elif name == 'success':
                    file_groups = success_file_groups
                else:
                    continue

                subdirs = [d for d in sorted(os.listdir(path)) if not d.startswith('.')]
                endval = len(subdirs) - 1
                for i, directory_name in enumerate(subdirs):
                    # fromid and toid only works when directory names are integers
                    try:
                        dir_id = int(directory_name)
                        if dir_id < fromid or dir_id > toid:
                            continue
                    except ValueError:
                        pass

                    self.progress(i, endval)
                    dpath = os.path.join(path, directory_name)
                    files = []
                    for fname in sorted(os.listdir(dpath)):
                        if fname.startswith('.'):
                            continue
                        fpath = os.path.join(dpath, fname)
                        with open(fpath, encoding='utf-8') as f:
                            files.append(f.read())
                    file_groups.append([files, directory_name])
                print()

        return success_file_groups, error_file_groups

    def extract_file_tag(self, file):
        """Return a domain-specific tag string for a single log file.
        Override in subclasses to inject process type or other metadata tags.
        """
        return ''

    def gen_texts(self, file_groups, numchar=3000):
        """Generate cleaned texts from file groups. Returns [[text, group_name], ...]."""
        print('*** Generating Data ***')
        texts = []
        endval = len(file_groups) - 1
        for i, item in enumerate(file_groups):
            files = item[0]
            group_name = item[1]
            t = []
            self.progress(i, endval)
            for file in files:
                start_index = len(file) - numchar
                index = file.find('\n', start_index)
                if index == -1:
                    index = start_index
                str_to_add = file[index:]
                tag = self.extract_file_tag(file)
                t.append(str_to_add + (' ***' + tag + 'LOG*** ' if tag else ' '))
            np.random.shuffle(t)
            texts.append([''.join(t), group_name])
        print()
        self.clean(texts)
        return texts

    def gen_labeled_texts(self, file_groups, label_id, numchar=3000):
        texts = self.gen_texts(file_groups, numchar=numchar)
        return [[text, label_id, group_name] for text, group_name in texts]

    def clean_domain(self, t):
        """Apply domain-specific text normalization. Override in subclasses."""
        return t

    def clean(self, text_label):
        print('*** Cleaning Data ***')
        endval = len(text_label) - 1
        for i, item in enumerate(text_label):
            self.progress(i, endval)
            t = re.sub(r'[\n\r\t]', ' ', item[0])
            t = re.sub(r'[(),]', ' ', t)
            # Date-
            t = re.sub(r'(^|\s)[0-9]+[/][0-9]+[/][0-9]+(\s|$)', ' num ', t)
            t = re.sub(r'(^|\s)[A-Z][a-z][a-z] [A-Z][a-z][a-z][ ]{1,2}[0-9]{1,2} [0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]+(\s|$)', ' ', t)
            t = re.sub(r'(^|\s)[A-Z][a-z][a-z][ ]{1,2}[0-9]{1,2} [0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9]+(\s|$)', ' ', t)
            t = re.sub(r'(^|\s)[A-Z][a-z][a-z] [A-Z][a-z][a-z][ ]{1,2}[0-9]{1,2} [0-9][0-9]:[0-9][0-9]:[0-9][0-9](\s|$)', ' ', t)
            t = re.sub(r'(^|\s)[0-9]+:[0-9]+:[0-9]+\.[0-9]+(\s|$)', ' ', t)
            # IP address
            t = re.sub(r'=[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+[,]*(\s|$)', '=ip ', t)
            t = re.sub(r'<[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+>', '<ip>', t)
            # Domain-specific patterns
            t = self.clean_domain(t)
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
        self.nb_errors = 0
        success_file_groups, error_file_groups = self.loadfiles(DATA_DIR, toid=toid)
        error_labeled_texts = self.gen_labeled_texts(error_file_groups, 1, numchar=numchar)
        success_labeled_texts = self.gen_labeled_texts(success_file_groups, 0, numchar=numchar)
        self.nb_errors = len(error_labeled_texts)
        self.stored_data = success_labeled_texts + error_labeled_texts

    def getdata(self, DATA_DIR, TEST_ERRORNUM=10, SUCCESS_LOG_RATIO=99, SUCCESS_LOG_RATIO_TEST=12.4,
                force_reload=False, numchar=3000, toid=6000):
        if len(self.stored_data) == 0 or force_reload:
            self.load(DATA_DIR, numchar=numchar, toid=toid)

        np.random.shuffle(self.stored_data)
        nb_test_success = int(TEST_ERRORNUM * SUCCESS_LOG_RATIO_TEST)
        nb_success = int(self.nb_errors * SUCCESS_LOG_RATIO)
        nb_test_error = TEST_ERRORNUM
        texts, labels, test_texts, test_labels = [], [], [], []
        labeled_texts = []
        test_labeled_texts = []

        for labeled_text in self.stored_data:
            if labeled_text[1] == 1:
                if nb_test_error > 0:
                    test_labeled_texts.append(labeled_text)
                    nb_test_error -= 1
                    continue
                else:
                    labeled_texts.append(labeled_text)
                    continue
            else:
                if nb_test_success > 0:
                    test_labeled_texts.append(labeled_text)
                    nb_test_success -= 1
                    continue
                elif nb_success > 0:
                    labeled_texts.append(labeled_text)
                    nb_success -= 1

        test_ids = {row[2] for row in test_labeled_texts}
        labeled_texts = [row for row in labeled_texts if row[2] not in test_ids]

        with open('datalist.log', 'a', encoding='utf-8') as f:
            for row in test_labeled_texts:
                test_texts.append(row[0])
                test_labels.append(row[1])
                f.write(row[2] + '\n')
            for row in labeled_texts:
                texts.append(row[0])
                labels.append(row[1])
                f.write('train' + row[2] + '\n')

        return texts, labels, test_texts, test_labels
