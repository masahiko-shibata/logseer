import argparse
import os
import sys
import pickle
import yaml
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.dirname(__file__))
from logseer.seer import Seer, OUTCOME_RESTART, OUTCOME_ALERT


def main():
    parser = argparse.ArgumentParser(description='LogSeer ensemble prediction on new logs')
    parser.add_argument('input', help='Directory of log sets (each subdirectory = one set)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--nn-model',     default=None,    help='Path to NN model (overrides config)')
    parser.add_argument('--xgb-model',    default=None,    help='Path to XGB model (overrides config)')
    parser.add_argument('--tokenizer',    default=None,    help='Path to tokenizer pickle (overrides config)')
    parser.add_argument('--nn-threshold', type=float, default=None)
    parser.add_argument('--xgb-threshold', type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    nn_model_path  = args.nn_model       or cfg.get('model_save_path',   'logseer.keras')
    xgb_model_path = args.xgb_model     or cfg.get('xgb_path',          'xgb.pkl')
    tokenizer_path = args.tokenizer      or cfg.get('tokenizer_path',    'tokenizer.pickle')
    nn_threshold   = args.nn_threshold   if args.nn_threshold  is not None else cfg.get('nn_threshold',      0.5)
    xgb_threshold  = args.xgb_threshold  if args.xgb_threshold is not None else cfg.get('sklearn_threshold', 0.5)
    numchar        = cfg.get('numchar',           3000)
    max_seq_len    = cfg.get('max_sequence_length', 26000)

    print(f'Tokenizer : {tokenizer_path}')
    print(f'NN model  : {nn_model_path}')
    print(f'XGB model : {xgb_model_path}')
    print(f'Thresholds: NN={nn_threshold:.2f}  XGB={xgb_threshold:.2f}')

    nn_model = load_model(nn_model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    xgb_model = None
    if os.path.exists(xgb_model_path):
        with open(xgb_model_path, 'rb') as f:
            xgb_model = pickle.load(f)
    else:
        print(f'Warning: XGB model not found at {xgb_model_path}, running NN only')

    seer = Seer(nn_model, tokenizer, xgb_model,
                numchar=numchar, max_sequence_length=max_seq_len,
                nn_threshold=nn_threshold, xgb_threshold=xgb_threshold)

    results = seer.predict(args.input)

    if not results:
        print('No processable log sets found.')
        sys.exit(1)

    has_xgb   = results[0]['xgb_prob'] is not None
    single    = len(results) == 1
    w = max(len(r['name']) for r in results)
    w = max(w, 8)
    print()
    if has_xgb:
        hdr = f'  {"Set":<{w}}  {"NN_prob":>8}  {"XGB_prob":>8}  {"OR":>5}  {"AND":>5}' + ('  Note' if single else '')
        print(hdr)
        print('  ' + '-' * len(hdr))
        for r in results:
            line = (f'  {r["name"]:<{w}}  {r["nn_prob"]:>8.4f}  {r["xgb_prob"]:>8.4f}'
                    f'  {"ERR" if r["or_pred"] else "ok":>5}  {"ERR" if r["and_pred"] else "ok":>5}')
            line += f'  {r["outcome"]}'
            print(line)
    else:
        hdr = f'  {"Set":<{w}}  {"NN_prob":>8}  {"NN":>5}' + ('  Note' if single else '')
        print(hdr)
        print('  ' + '-' * len(hdr))
        for r in results:
            line = f'  {r["name"]:<{w}}  {r["nn_prob"]:>8.4f}  {"ERR" if r["nn_pred"] else "ok":>5}'
            line += f'  {r["outcome"]}'
            print(line)

    print()
    any_restart = any(r['outcome'] == OUTCOME_RESTART for r in results)
    any_alert   = any(r['outcome'] == OUTCOME_ALERT   for r in results)

    if single:
        if any_restart:
            print('OUTCOME: RESTART  (AND ensemble triggered — hold deployment)')
            sys.exit(2)
        elif any_alert:
            print('OUTCOME: ALERT    (OR ensemble triggered — monitor closely)')
            sys.exit(1)
        else:
            print('OUTCOME: OK')
            sys.exit(0)

if __name__ == '__main__':
    main()
