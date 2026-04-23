import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from logseer import run_training


def main():
    parser = argparse.ArgumentParser(
        description='Train LogSeer models',
        epilog='Any config key can be overridden as key=value, e.g. epochs=10 max_nb_words=100',
    )
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML file')
    parser.add_argument('overrides', nargs='*', help='key=value overrides for config')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for override in args.overrides:
        if '=' not in override:
            print(f'Warning: ignoring malformed override {override!r} (expected key=value)')
            continue
        key, _, raw_value = override.partition('=')
        # preserve type: try int, float, bool, then leave as string
        for cast in (int, float):
            try:
                value = cast(raw_value)
                break
            except ValueError:
                pass
        else:
            if raw_value.lower() in ('true', 'yes'):
                value = True
            elif raw_value.lower() in ('false', 'no'):
                value = False
            elif key in ('sklearn_models',) and ',' in raw_value:
                value = [m.strip() for m in raw_value.split(',')]
            else:
                value = raw_value
        cfg[key] = value
        print(f'Override: {key} = {value!r}')

    run_training(
        cfg['data_dir'],
        max_nb_words=cfg.get('max_nb_words', 24000),
        max_sequence_length=cfg.get('max_sequence_length', 26000),
        embedding_dim=cfg.get('embedding_dim', 32),
        validation_split=cfg.get('validation_split', 0.1),
        validate_on_test_data=cfg.get('validate_on_test_data', False),
        epochs=cfg.get('epochs', 60),
        batch_size=cfg.get('batch_size', 32),
        model_save_path=cfg.get('model_save_path', 'logseer.keras'),
        tokenizer_path=cfg.get('tokenizer_path', 'tokenizer.pickle'),
        embedding_layer_type=cfg.get('embedding_layer', 'vanilla'),
        model_name=cfg.get('model_name', 'LogCNNLite'),
        repetition=cfg.get('repetition', 100),
        nn_error_weight=cfg.get('nn_error_weight', 2),
        sklearn_models=cfg.get('sklearn_models', ['xgb']),
        ensemble_model=cfg.get('ensemble_model', None),
        sklearn_weight=cfg.get('sklearn_weight', 6),
        learning_rate=cfg.get('learning_rate', 0.0003),
        max_loss=cfg.get('max_loss', 0.7),
        retrain=cfg.get('retrain', False),
        numchar=cfg.get('numchar', 3000),
        toid=cfg.get('toid', 6000),
        checkpoint_type=cfg.get('checkpoint_type', 'best_f1'),
        start_from_epoch=cfg.get('start_from_epoch', 0),
        es_start_from_epoch=cfg.get('es_start_from_epoch', 0),
        use_early_stopping=cfg.get('use_early_stopping', False),
        patience=cfg.get('patience', None),
        monitor=cfg.get('monitor', 'val_f1'),
        mode=cfg.get('mode', 'max'),
        restore_best_weights=cfg.get('restore_best_weights', False),
        test_nn=cfg.get('test_nn', True),
        nn_threshold=cfg.get('nn_threshold', 0.5),
        sklearn_threshold=cfg.get('sklearn_threshold', 0.5),
        success_log_ratio=cfg.get('success_log_ratio', 99),
        success_log_ratio_test=cfg.get('success_log_ratio_test', 12.4),
        dump_proba=cfg.get('dump_proba', False),
    )


if __name__ == '__main__':
    main()