"""
Threshold tuning utility.

Usage:
    python tune_threshold.py                         # reads proba.csv
    python tune_threshold.py --proba proba.csv
    python tune_threshold.py --model xgbModel        # single model only
    python tune_threshold.py --min-recall 0.4        # filter to rows where recall >= threshold
"""
import argparse
import csv
import numpy as np


def sweep(labels, probs, thresholds):
    rows = []
    total_pos = sum(labels)
    for t in thresholds:
        pred = [1 if p >= t else 0 for p in probs]
        tp = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 1)
        fp = sum(1 for y, p in zip(labels, pred) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(labels, pred) if y == 1 and p == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / total_pos if total_pos > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append((t, tp, fp, fn, precision, recall, f1))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Threshold tuning from proba.csv')
    parser.add_argument('--proba',       default='proba.csv', help='Path to proba.csv')
    parser.add_argument('--model',       default=None,        help='Filter to a specific model name')
    parser.add_argument('--min-recall',  type=float,          default=0.0,
                        help='Only show rows where recall >= this value')
    parser.add_argument('--step',        type=float,          default=0.05, help='Threshold step size')
    args = parser.parse_args()

    # Load
    data = {}
    with open(args.proba, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name  = row['model']
            label = int(row['label'])
            proba = float(row['proba'])
            if args.model and name != args.model:
                continue
            if name not in data:
                data[name] = ([], [])
            data[name][0].append(label)
            data[name][1].append(proba)

    if not data:
        print('No data found. Check --proba path or --model name.')
        return

    thresholds = np.arange(0.05, 1.0, args.step)

    for name, (labels, probs) in sorted(data.items()):
        print()
        print(f'=== {name}  (n={len(labels)}, errors={sum(labels)}) ===')
        print(f'  Prob stats: mean={np.mean(probs):.3f}  median={np.median(probs):.3f}'
              f'  p10={np.percentile(probs, 10):.3f}  p90={np.percentile(probs, 90):.3f}')
        print()
        header = f'  {"threshold":>9}  {"TP":>5}  {"FP":>5}  {"FN":>5}  {"precision":>9}  {"recall":>6}  {"F1":>6}'
        print(header)
        print('  ' + '-' * (len(header) - 2))

        rows = sweep(labels, probs, thresholds)
        best_f1 = max(rows, key=lambda r: r[6])

        for t, tp, fp, fn, precision, recall, f1 in rows:
            if recall < args.min_recall:
                continue
            marker = ' <-- best F1' if t == best_f1[0] else ''
            print(f'  {t:>9.3f}  {tp:>5}  {fp:>5}  {fn:>5}  {precision:>9.3f}  {recall:>6.3f}  {f1:>6.3f}{marker}')

        print()
        print(f'  Suggested threshold: {best_f1[0]:.3f}  '
              f'(precision={best_f1[4]:.3f}  recall={best_f1[5]:.3f}  F1={best_f1[6]:.3f})')


if __name__ == '__main__':
    main()

