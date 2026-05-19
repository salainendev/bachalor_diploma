from pathlib import Path
from typing import Tuple, Union
import dill
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import yaml
from pathlib import Path

def read_yaml_to_dict(file_path):
    """
    Читает YAML-файл и возвращает его содержимое в виде словаря.

    Args:
        file_path (str или Path): Путь к YAML-файлу.

    Returns:
        dict: Данные из YAML-файла.

    Raises:
        FileNotFoundError: Если файл не существует.
        yaml.YAMLError: Если файл имеет неверный синтаксис.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)  # safe_load предотвращает выполнение произвольного кода
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")

    return data if data is not None else {}  # Пустой файл -> пустой словарь

def read_txt_to_dataframe_v2(file_path, column_names=None):
    """
    Альтернативная версия с использованием read_csv и регулярного выражения
    """
    
    if column_names is None:
        column_names = ['VV', 'FullAbs', 'CI']
    elif len(column_names) != 3:
        raise ValueError("Должно быть ровно 3 имени столбцов")
    
    try:
        # Используем регулярное выражение для разделителя: 
        # пробел или табуляция, один или более раз
        df = pd.read_csv(
            file_path,
            sep=r'[ \t]+',  # регулярное выражение для пробелов и табуляций
            header=None,
            names=column_names,
            engine='python',  # python engine поддерживает regex разделители
            encoding='utf-8',
            skip_blank_lines=True
        )
        
        # Очистка строковых значений
        if df['CI'].isna().all(): df['CI'] = df['CI'].fillna(0)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # print(f"Файл успешно прочитан")
        # print(f"Размер DataFrame: {df.shape}")
        
        return df
        
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {e}")
    
def get_real_observations(path_to_folder:str | Path):
    path_to_folder = Path(path_to_folder)
    answer = {}
    for file in list(path_to_folder.rglob("*.txt")): 
        df_Wasp52 = read_txt_to_dataframe_v2(str(file))
        print(file)
        # Удаляем строки с NaN (если были ошибки преобразования)
        df_Wasp52_clean = df_Wasp52.dropna()

        df_Wasp52_clean["VV"] = (
            df_Wasp52_clean["VV"]
            .astype(str)                      # гарантируем строковый тип
            .str.replace(',', '.')            # заменяем запятые на точки
            .str.replace(' ', '')             # удаляем пробелы если есть
        )
        df_Wasp52_clean["FullAbs"] = (
            df_Wasp52_clean["FullAbs"]
            .astype(str)                      # гарантируем строковый тип
            .str.replace(',', '.')            # заменяем запятые на точки
            .str.replace(' ', '')             # удаляем пробелы если есть
        )
        # Теперь можно преобразовывать в numpy float32
        VV = df_Wasp52_clean["VV"].to_numpy(np.float32)
        X = df_Wasp52_clean["FullAbs"].to_numpy(np.float32)
        pname = Path(file).stem
        if pname!="Wasp-121b":
            X /= 100
        VV /= 10.07
        
        # Исходные данные (возрастающие x)
        x_orig = VV
        y_orig = X   # последний отрезок резко вверх

        # Запросы: внутри и вне диапазона
        x_query = np.linspace(-5,5,101)
        
        # --- Вариант 1: линейная интерполяция с экстраполяцией по последнему отрезку ---
        f_linear_ext = interp1d(x_orig, y_orig,
                                kind='linear',
                                fill_value='extrapolate')   # ключевой параметр
    
        # print(df_Wasp52)
        y_linear_ext = f_linear_ext(x_query)

        

        # --- Вариант 2: более безопасная экстраполяция (постоянные значения на краях) ---
        f_const_ext = interp1d(x_orig, y_orig,
                            kind='linear',
                            fill_value=(y_orig[0], y_orig[-1]),   # задать явно
                            bounds_error=False)

        y_const_ext = f_const_ext(x_query)
       

        answer[pname] = {
            "V_orig":VV,
            "FullAbs_orig":X,
            "V_interp":x_query,
            "FullAbs_interp":y_const_ext
        }
    return answer


def normalize_spectrum(S, method='area'):
    """
    S: (B, 101) или (101,)
    method: 'area' (интеграл=1) или 'max' (амплитуда=1)
    """
    if method == 'area':
        return S / (torch.sum(torch.abs(S), dim=-1, keepdim=True) + 1e-8)
    elif method == 'max':
        return S / (torch.max(torch.abs(S), dim=-1, keepdim=True).values + 1e-8)
    return S

def get_validation_calibrated_top5(scores, threshold=0.8, n=5):
    """
    Простой вариант: penalize за отклонение от медианы валидации.
    Предпочитает scores, похожие на те, что модель реально выдаёт на валидации.
    """
    val_median = threshold
    
    calibrated_scores = {}
    for score, params in scores.items():
        # Близость к медиане валидации — это хорошо
        # Отклонение вверх (переобучение) штрафуется сильнее, чем вниз
        if score > val_median:
            penalty = (score - val_median) * 2.0  # сильный штраф за переобучение
        else:
            penalty = (val_median - score) * 0.5  # слабый штраф за недообучение
        
        calibrated = score - penalty
        calibrated_scores[calibrated] = params
    
    sorted_calibrated = sorted(calibrated_scores.items(), key=lambda x: x[0], reverse=True)
    return sorted_calibrated[:n], calibrated_scores

def compute_pair_score(model, spectrum, param_a, param_b, device=None):
    """
    Вычисляет cosine similarity между эмбеддингами спектра и параметров.
    
    Args:
        model: SpectralParamTwoTower
        spectrum: torch.Tensor [seq_len] или [1, seq_len]
        param_a, param_b: float или torch.Tensor [1]
    
    Returns:
        score: float
    """
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    model.eval()
    
    if isinstance(spectrum, np.ndarray):
        spectrum = torch.from_numpy(spectrum).float()
    if isinstance(param_a, np.ndarray):
        param_a = torch.from_numpy(param_a).float()
    if isinstance(param_b, np.ndarray):
        param_b = torch.from_numpy(param_b).float()
    # Подготовка спектра
    if spectrum.ndim == 1:
        spectrum = spectrum.unsqueeze(0)  # [1, seq_len]
    spectrum = normalize_spectrum(spectrum).to(device)
    # Подготовка параметров
    if isinstance(param_a, (float, int)):
        param_a = torch.tensor([param_a], device=device)
    if isinstance(param_b, (float, int)):
        param_b = torch.tensor([param_b], device=device)
    
    # Инференс + cosine similarity
    with torch.no_grad():
        z_spec, z_param, _,_ = model(spectrum, param_a, param_b)
        # z_spec = F.normalize(z_spec, p=2, dim=-1)
        # z_param = F.normalize(z_param, p=2, dim=-1)
        score = F.cosine_similarity(z_spec, z_param).item()
    
    return score

def get_validation_calibrated_top5(scores, threshold=0.8, n=5):
    """
    Простой вариант: penalize за отклонение от медианы валидации.
    Предпочитает scores, похожие на те, что модель реально выдаёт на валидации.
    """
    val_median = threshold
    
    calibrated_scores = {}
    for score, params in scores.items():
        # Близость к медиане валидации — это хорошо
        # Отклонение вверх (переобучение) штрафуется сильнее, чем вниз
        if score > val_median:
            penalty = (score - val_median) * 2  # сильный штраф за переобучение
        else:
            penalty = (val_median - score) * 0.5  # слабый штраф за недообучение
        
        calibrated = score - penalty
        calibrated_scores[calibrated] = params
    
    sorted_calibrated = sorted(calibrated_scores.items(), key=lambda x: x[0], reverse=True)
    return sorted_calibrated[:n], calibrated_scores



def get_xuv_and_he_prediction(spectra, model_xuv, scaler_xuv, model_he, scaler_he, model_scoring):
    y_const_ext = spectra
    xuv_candidates, xuv_candidates_denormalized = run_chronos_multimodal_inference(model=model_xuv,
                                    scaler=scaler_xuv,
                                    spectra=y_const_ext,
                                    device='cpu')

    he_candidates, he_candidates_denormalized = run_chronos_multimodal_inference(model=model_he,
                                    scaler=scaler_he,
                                    spectra=y_const_ext,
                                    device='cpu')

    grid_i, grid_j = torch.meshgrid(torch.arange(len(he_candidates.squeeze(0))), torch.arange(len(xuv_candidates.squeeze(0))), indexing='ij')

    # Индексируем тензоры по сетке

    combinations = torch.stack([he_candidates.squeeze(0)[grid_i.flatten()], xuv_candidates.squeeze(0)[grid_j.flatten()]], dim=1)
    # combinations.shape
    scores = {}
    for pair in combinations:
        score = compute_pair_score(model_scoring, y_const_ext, pair[0], pair[1], device='cpu')
        ans = []
        for pred, scaler in zip((pair[0], pair[1]), (scaler_he, scaler_xuv)):
            
            preds_flat = pred.reshape(-1, 1).numpy()  # [batch * num_heads, output_dim]
            preds_flat_denorm = scaler.inverse_transform(preds_flat)
            # preds_denorm = torch.from_numpy(preds_flat_denorm).float()
            ans.append(preds_flat_denorm[0][0])
        scores[score] = ans
        # print(ans, score)

    sorted_items, calibrated_scores = get_validation_calibrated_top5(scores,0.66, n=5)
    # sorted_items = sorted(scores.items(), key=lambda x: x[0], reverse=True)
    top5 = sorted_items[:7]
    

    # for score, params in top5:
    #     print(f"Score: {score:.4f}, Params: He/H={params[0]:.6f}, XUVInt={params[1]:.6f}")
    return top5, scores
@torch.no_grad
def get_msw_prediction(spectra, model_msw, scaler_msw, device="cpu"):
    if device is None:
        device = next(model_msw.parameters()).device
        if device == torch.device('cpu') and torch.cuda.is_available():
            device = 'cuda'
    
    # Переносим модель на нужное устройство
    model_msw = model_msw.to(device)
    model_msw.eval()
    # ВАЖНО: Сохраняем ссылку на токенизатор и его boundaries
    tokenizer = None
    original_boundaries = None
    if hasattr(model_msw, 'tokenizer'):
        tokenizer = model_msw.tokenizer
        if hasattr(tokenizer, 'boundaries'):
            original_boundaries = tokenizer.boundaries
            # Убеждаемся, что boundaries на CPU
            tokenizer.boundaries = original_boundaries.cpu()
    
    # 1. Нормализация входа
    if isinstance(spectra, np.ndarray):
        spectra = torch.from_numpy(spectra).float()
    if spectra.ndim == 1:
        spectra = spectra.unsqueeze(0)  # [1, seq_len]
        
    # 2. Перенос на устройство модели
    spectra = spectra.to(device)
    msw_raw = model_msw(spectra)
    msw_raw.reshape(-1, 1).numpy()
    msw_log = scaler_msw.inverse_transform(msw_raw)
    msw = np.expm1(msw_log)
    return msw

def run_chronos_multimodal_inference(
    model: torch.nn.Module,
    scaler,
    spectra: Union[np.ndarray, torch.Tensor],
    device: str = None,
    return_conf: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Инференс MultiHeadChronosRegressor.
    Возвращает предсказания и оценки уверенности для ВСЕХ голов.
    
    Args:
        model: модель в режиме eval()
        scaler: обученный StandardScaler для денормализации предсказаний
        spectra: np.array или torch.Tensor формы [batch, seq_len] или [seq_len]
        device: 'cuda'/'cpu'. Если None, берётся из весов модели
        return_conf: если False, возвращает только preds
        
    Returns:
        preds: torch.Tensor [batch, num_heads, output_dim] (денормализованные)
        confs: torch.Tensor [batch, num_heads] (если return_conf=True)
    """
    if device is None:
        device = next(model.parameters()).device
        if device == torch.device('cpu') and torch.cuda.is_available():
            device = 'cuda'
    
    # Переносим модель на нужное устройство
    model = model.to(device)
    
    # ВАЖНО: Сохраняем ссылку на токенизатор и его boundaries
    tokenizer = None
    original_boundaries = None
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
        if hasattr(tokenizer, 'boundaries'):
            original_boundaries = tokenizer.boundaries
            # Убеждаемся, что boundaries на CPU
            tokenizer.boundaries = original_boundaries.cpu()
    
    # 1. Нормализация входа
    if isinstance(spectra, np.ndarray):
        spectra = torch.from_numpy(spectra).float()
    if spectra.ndim == 1:
        spectra = spectra.unsqueeze(0)  # [1, seq_len]
        
    # 2. Перенос на устройство модели
    spectra = spectra.to(device)
    
    # 3. Инференс
    model.eval()
    with torch.no_grad():
        preds = model(spectra)
    
    # Восстанавливаем boundaries после инференса (на всякий случай)
    if original_boundaries is not None:
        tokenizer.boundaries = original_boundaries
        
    # 4. Возврат на CPU и денормализация
    preds = preds.cpu()
    
    
    # Денормализация preds
    batch_size, num_heads, output_dim = preds.shape
    preds_flat = preds.reshape(-1, output_dim).numpy()  # [batch * num_heads, output_dim]
    preds_flat_denorm = scaler.inverse_transform(preds_flat)
    preds_denorm = torch.from_numpy(preds_flat_denorm).reshape(batch_size, num_heads, output_dim).float()
    
    return preds, preds_denorm

def load_model_with_architecture(filepath='complete_model.pkl', device='cuda'):
    """
    Загружает модель ПОЛНОСТЬЮ, даже если класс не определён в текущем скоупе
    """
    with open(filepath, 'rb') as f:
        model = torch.load(f, map_location=device, pickle_module=dill)
    model.eval()
    print(f"✅ Модель со всей архитектурой загружена из {filepath}")
    return model

