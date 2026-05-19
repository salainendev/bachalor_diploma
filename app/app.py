from pathlib import Path
import argparse
import pickle

import torch

from app_functions import get_real_observations, get_xuv_and_he_prediction, get_msw_prediction, read_yaml_to_dict, load_model_with_architecture

def load_all_from_config(config_path):
    """Загружает все скейлеры и модели, используя read_yaml_to_dict."""
    config = read_yaml_to_dict(config_path)

    weights_dir = Path(config['paths']['inference_weights_dir'])
    models_cfg = config['models']

    # Скейлеры
    scaler_xuv = pickle.load(open(weights_dir / models_cfg['scaler_xuv'], 'rb'))
    scaler_he  = pickle.load(open(weights_dir / models_cfg['scaler_he'],   'rb'))
    scaler_msw = pickle.load(open(weights_dir / models_cfg['scaler_msw'],  'rb'))

    # Модели
    model_xuv = load_model_with_architecture(weights_dir / models_cfg['model_xuv'])
    model_he  = load_model_with_architecture(weights_dir / models_cfg['model_he'])
    model_msw = load_model_with_architecture(weights_dir / models_cfg['model_msw'])
    model_h2a = load_model_with_architecture(weights_dir / models_cfg['model_h2a'])

    # Скоринговая модель – может лежать в текущей папке или внутри inference_weights_dir
    scoring_path = Path(models_cfg['model_scoring'])
    if not scoring_path.exists():
        scoring_path = weights_dir / models_cfg['model_scoring']
    model_scoring = load_model_with_architecture(scoring_path, device='cpu')

    return {
        'scaler_xuv': scaler_xuv,
        'scaler_he': scaler_he,
        'scaler_msw': scaler_msw,
        'model_xuv': model_xuv,
        'model_he': model_he,
        'model_msw': model_msw,
        'model_h2a': model_h2a,
        'model_scoring': model_scoring,
    }


def parse_args():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Predict planet parameters from spectra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python predict.py --config config.yaml --data path_to_dat_folder --output results.txt
        """
    )

    # Обязательные аргументы
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Путь к YAML конфигу с путями к моделям"
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Путь к файлу с данными (.pkl или .json)"
    )

    # Опциональные аргументы
    parser.add_argument(
        "--output", "-o",
        default="results.txt",
        required=True,
        help="Файл для сохранения результатов (по умолчанию: results.json)"
    )
    

    return parser.parse_args()


def main():
    args = parse_args()
    answer = get_real_observations(path_to_folder=args.data)
    models = load_all_from_config(config_path=args.config)
    scaler_xuv = models['scaler_xuv']
    scaler_he = models['scaler_he']
    scaler_msw = models['scaler_msw']
    model_xuv = models['model_xuv']
    model_he = models['model_he']
    model_msw = models['model_msw']
    model_h2a = models['model_h2a']
    model_scoring = models['model_scoring']
    with open(str(args.output), "w", encoding='utf-8') as file:
        for pname, p_info in answer.items():
            top5, scores = get_xuv_and_he_prediction(p_info['FullAbs_interp'], model_xuv, scaler_xuv, model_he, scaler_he, model_scoring)
            msw = get_msw_prediction(p_info["FullAbs_interp"], model_msw, scaler_msw)
            model_h2a.to('cpu')
            h2a = model_h2a.predict(torch.from_numpy(p_info["FullAbs_interp"]).float().unsqueeze(0), threshold=0.5)
            print(f"========================== Planet - {pname} ============================")
            file.write(f"========================== Planet - {pname} ============================\n")
            print(" ")
            file.write(" \n")
            print(f" Msw = {msw}, H2a = {h2a}")
            file.write(f" Msw = {msw}, H2a = {h2a}\n")
            for score, params in top5:
                print(f"Score: {score:.4f}, Params: He/H={params[0]:.4f}, XUVInt={int(params[1])}")
                file.write(f"Score: {score:.4f}, Params: He/H={params[0]:.4f}, XUVInt={int(params[1])}\n")
            print(" ")
            file.write(" \n")
            print(" ")
            file.write(" \n")

if __name__ == "__main__":
    main()
