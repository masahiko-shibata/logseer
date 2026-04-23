import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class Tester:

    def __init__(self):
        self.stored = []

    def genresult(self, name, y, pred, heatmap):
        results = []
        cm = confusion_matrix(y, pred)

        total = len(pred)
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[0][0]

        res = precision_recall_fscore_support(y, pred, zero_division=0)
        results.append([res[0][1], res[1][1], res[2][1], total, tp, fp, fn, tn, name])

        df_cols = ['precision', 'recall', 'f1_score', 'total_samples', 'TP', 'FP', 'FN', 'TN', 'model']
        result_df = pd.DataFrame(results, columns=df_cols)

        result_text = result_df.to_string(columns=df_cols, index=False)
        print()
        print(result_text)

        filename = name + '.log'
        content = str(tp) + '\t' + str(fp) + '\t' + str(tn) + '\t' + str(fn) + '\n'
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(content)

        if heatmap:
            plt.figure()
            df_cm = pd.DataFrame(cm, index=['Success', 'Failed'])
            df_cm.columns = ['Success', 'Failed']
            ax = plt.axes()
            sns.heatmap(df_cm, annot=True, fmt="d", linewidths=.5, ax=ax)
            ax.set_title(name)

        return result_text

    def testModel(self, model, x_test, y_test, fnames=None, heatmap=True,
                  threshold=0.5):
        result = model.predict(x_test)
        if issubclass(type(result[0]), np.ndarray):
            probs = result.flatten().tolist()
            predictions = [1 if p >= threshold else 0 for p in probs]
        else:
            probs = list(model.predict_proba(x_test)[:, 1])
            predictions = [1 if p >= threshold else 0 for p in probs]

        if fnames:
            print()
            print('Prediction of ' + model.name)
            print()
            for i, x in enumerate(predictions):
                prediction = 'Success' if x == 1 else 'Failure'
                print(prediction, y_test[i], fnames[i], result[i])

        added = False
        for row in self.stored:
            if row[0] == model.name:
                row[1] = row[1] + y_test
                row[2] = row[2] + predictions
                row[3] = row[3] + probs
                added = True
        if self.stored == [] or not added:
            self.stored.append([model.name, y_test, predictions, probs])

        self.genresult(model.name, y_test, predictions, heatmap=False)

    def total(self, heatmap=False):
        result_text = ''
        print()
        print('### Total Result ###')
        for row in self.stored:
            tmp_text = self.genresult(row[0], row[1], row[2], heatmap=heatmap)
            tmp_text = "\t".join(tmp_text.split())
            result_text = result_text + tmp_text + '\n'
        with open('result.txt', 'w', encoding='utf-8') as f:
            f.write(result_text)

    def dump_proba(self, path='proba.csv'):
        """Write all stored probabilities to a CSV for threshold analysis.
        Columns: model, label, proba
        """
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'label', 'proba'])
            for row in self.stored:
                name, labels, _, probs = row[0], row[1], row[2], row[3]
                for label, proba in zip(labels, probs):
                    writer.writerow([name, label, proba])
        print(f'Probabilities written to {path}')

