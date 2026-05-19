from pathlib import Path

import numpy as np

def parse_parameters(file_path):
    """
    Читает файл parameters.txt и возвращает словарь с нужными параметрами
    """
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                value = ' '.join(parts[1:])  # на случай, если значение содержит пробелы
                params[key] = value
    # Извлекаем нужные параметры
    result = {}
    for param in ['PName', 'Msw', 'XUVInt', 'H2a', 'Helium']:
        result[param] = params.get(param)
    return result

def process_files_in_folder_with_params(root_folder):
    """
    Рекурсивно проходит по папке, читает параметры и файлы с данными
    Возвращает список словарей, где каждый словарь — это один файл
    """
    dataset = []
    root = Path(root_folder)
    for data_file_path in root.rglob('*[.txt,.dat]'):
        if data_file_path.name == 'parameters.txt':
            continue  # Пропускаем файл параметров

        # Ищем файл parameters.txt в той же папке
        param_file_path = data_file_path.parent / 'parameters.txt'

        if param_file_path.exists():
            params = parse_parameters(param_file_path)
            # print(f"Файл: {data_file_path}")
            # print(f"Параметры: {params}")
        else:
            # print(f"Файл parameters.txt не найден для {data_file_path}")
            params = {}

        # Читаем V и FullAbs из data файла
        try:
            V, FullAbs = np.loadtxt(data_file_path, usecols=(0, 1), unpack=True, skiprows=1)
            entry = {
                'V': V,
                'FullAbs': FullAbs,
                'PName': params.get('PName'),
                'Msw': params.get('Msw'),
                'XUVInt': params.get('XUVInt'),
                'H2a': params.get('H2a'),
                'Helium': params.get('Helium'),
                # 'File': str(data_file_path)  # можно убрать, если не нужно
            }
            dataset.append(entry)
        except Exception as e:
            print(f"Ошибка при загрузке данных из {data_file_path}: {e}")
        # print("-" * 50)

    return dataset

def pad_or_truncate(seq, target_len=101, pad_value=0.0):
    seq = np.array(seq, dtype=np.float32)
    L = len(seq)

    if L < target_len:
        total_pad = target_len - L
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad

        left = np.full(left_pad, pad_value, dtype=np.float32)
        right = np.full(right_pad, pad_value, dtype=np.float32)

        return np.concatenate([left, seq, right])

    else:
        return seq