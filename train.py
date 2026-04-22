import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from logseer import run_training


def main():
    parser = argparse.ArgumentParser(description='Train LogSeer models')
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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
        error_weight=cfg.get('error_weight', 6),
        nn_error_weight=cfg.get('nn_error_weight', 2),
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
        test_xgb=cfg.get('test_xgb', True),
        test_svm=cfg.get('test_svm', False),
        test_rf=cfg.get('test_rf', False),
        success_log_ratio=cfg.get('success_log_ratio', 99),
        success_log_ratio_test=cfg.get('success_log_ratio_test', 12.4),
    )


if __name__ == '__main__':
    main()