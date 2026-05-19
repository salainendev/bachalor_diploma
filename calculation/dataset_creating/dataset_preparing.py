from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
# import matplotlib.pyplot as plt
import random
import shutil

PARAMS_HELIUM = np.unique(np.round(((np.trunc(np.geomspace(0.01, 0.3, 50)*1000) / 1000) + 0.000022), 6)).tolist()
PARAMS_h2a = [1, 0]
PARAMS_XUV = range(1, 51, 2)
# PARAMS_MSW = [1e11, 2e11, 3e11, 4e11, 5e11, 6e11, 7e11, 8e11, 9e11, 1e12, 2e12, 3e12, 4e12, 5e12, 8e12, 1e13, 2e13, 5e13]

@dataclass
class AbsorptionData:
    """Структура для данных из absorbtion.dat"""
    time: np.ndarray
    full_abs: np.ndarray
    res_part: np.ndarray
    thermal: np.ndarray


@dataclass
class Parameters:
    """Структура для параметров из parameters.txt"""
    geometry: int
    rmax: float
    pname: str
    xuv_int: int
    rtherm: float
    rbase: float
    tem_base: float
    pbase: float
    max_flux: float
    tem_min: float
    tem_max: float
    h2a: int
    helium: float
    he_m2s: float
    ewoo: float
    vsw: float
    rsw: float
    msw: str
    tmax: float
    dt_ris: float
    dt_save: float
    tstart: float

@dataclass
class ExperimentData:
    """Общая структура данных эксперимента"""
    absorption: AbsorptionData
    parameters: Parameters
    folder_path: Path

@dataclass(frozen=True)
class TargetParamsSample:
    xuv: int
    msw: str
    h2a: int
    helium: float

def generate_sample(
    dataset: set[TargetParamsSample],
    xuv: int | None = None,
    msw: str | None = None,
    h2a: int | None = None,
    helium: float | None = None,
    max_attempts: int = 10000
) -> TargetParamsSample:
    PARAMS_MSW = generate_msw()
    
    for _ in range(max_attempts):
        current_xuv = xuv if xuv is not None else random.choice(PARAMS_XUV)
        current_msw = msw if msw is not None else random.choice(list(PARAMS_MSW))
        current_h2a = h2a if h2a is not None else random.choice(PARAMS_h2a)
        current_helium = helium if helium is not None else float(random.choice(PARAMS_HELIUM))

        new = TargetParamsSample(
            xuv=current_xuv,
            msw=current_msw,
            h2a=current_h2a,
            helium=current_helium
        )

        if new not in dataset:
            return new

    raise RuntimeError(f"Failed to generate unique sample after {max_attempts} attempts")

def generate_samples_set(dataset: set[TargetParamsSample], num: int, xuv: int | None = None, msw: str | None = None, h2a: int | None = None, helium: float | None = None) -> set[TargetParamsSample]:
    res = set()
    count = 0
    for _ in range(num):
        sample = generate_sample(dataset=dataset,
                                 xuv=xuv,
                                 msw=msw,
                                 h2a=h2a,
                                 helium=helium)
        if sample not in dataset and sample not in res:
            res.add(sample)
            count+=1
            # print(sample)
    return res

def update_parameters(input_file, output_file, new_values):
    """
    Обновляет числовые значения параметров в файле.

    :param input_file: путь к исходному файлу
    :param output_file: путь к обновлённому файлу (может совпадать с input_file)
    :param new_values: словарь вида {'Параметр': новое_значение}
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or ' ' not in stripped:
            # Пропускаем пустые строки, комментарии или строки без пробела
            updated_lines.append(line)
            continue

        parts = stripped.split(maxsplit=1)
        key = parts[0]
        if key in new_values:
            # Заменяем значение
            new_val = new_values[key]
            # Сохраняем формат: ключ и значение через пробел
            updated_line = f"{key} {new_val}\n"
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    return output_file


def generate_sample_folder(path_to_calculation_utils: Path, path_to_output_folder: Path, params: TargetParamsSample):
    path_to_output_folder.mkdir(parents=True, exist_ok=True)

    path_to_exo3d = path_to_calculation_utils / 'exo3d.exe'
    path_to_grid = path_to_calculation_utils / 'gridPars.txt'
    path_to_mpi = path_to_calculation_utils / 'mpi.bat'
    path_to_probes = path_to_calculation_utils / 'probes.txt'
    path_to_parameters = path_to_calculation_utils / 'parameters.txt'
    
    new_values = {
        "XUVInt": params.xuv,
        "H2a" : params.h2a,
        "Helium": params.helium,
        "Msw" : params.msw,
        "Tmax": 120
    }

    path_to_parameters = update_parameters(path_to_parameters, path_to_output_folder / 'parameters.txt', new_values=new_values)

    shutil.copy(path_to_exo3d, path_to_output_folder)
    shutil.copy(path_to_grid, path_to_output_folder)
    shutil.copy(path_to_mpi, path_to_output_folder)
    shutil.copy(path_to_probes, path_to_output_folder)

def generate_samples_from_set(path_to_calculation_utils: Path, path_to_output_folder:Path, samples_set: set[TargetParamsSample]):
    samples_list= list(samples_set)
    for sample in samples_list:
        output_folder_name = f"XUV{sample.xuv}Msw{sample.msw}He{str(sample.helium).replace('.', 'p')[:-2]}H{sample.h2a}"
        path_to_output_folder_sample  = Path(path_to_output_folder) / Path(output_folder_name)
        generate_sample_folder(
            path_to_calculation_utils=Path(path_to_calculation_utils),
            path_to_output_folder=Path(path_to_output_folder_sample),
            params=sample
        )
        print(f"generated folder - {output_folder_name}")
def generate_msw_list(exponenta) -> List:
    match exponenta:

        case 11:
            mantisa = range(1,100)
        case 12:
            mantisa = range(1,100)
        case 13:
            mantisa = range(1,6)
    
    return [f'{i}e{exponenta}' for i in mantisa]

def generate_msw() -> set:

    res = generate_msw_list(11) + generate_msw_list(12) + generate_msw_list(13)
    return set(res)


def parse_absorption_file(file_path: Path) -> Optional[AbsorptionData]:
    """Парсит файл absorbtion.dat"""
    try:
        if not file_path.exists():
            print(f"Файл не найден: {file_path}")
            return None
            
        data = np.loadtxt(file_path, skiprows=1)
        if data.shape[1] < 4:
            print(f"Предупреждение: Файл {file_path} имеет недостаточно столбцов")
            return None
            
        return AbsorptionData(
            time=data[:, 0],
            full_abs=data[:, 1],
            res_part=data[:, 2],
            thermal=data[:, 3]
        )
    except Exception as e:
        print(f"Ошибка парсинга файла {file_path}: {e}")
        return None

def parse_parameters_file(file_path: Path) -> Optional[Parameters]:
    """Парсит файл parameters.txt"""
    try:
        if not file_path.exists():
            print(f"Файл не найден: {file_path}")
            return None
            
        params_dict = {}
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                key = parts[0].lower()
                value = ' '.join(parts[1:])
                
                # Преобразование числовых значений
                try:
                    if any(char in value for char in [ 'e', 'E']):
                        numeric_value = str(value)
                    elif '.' in value:
                        numeric_value = float(value)
                    else:
                        numeric_value = int(value)
                    params_dict[key] = numeric_value
                except ValueError:
                    params_dict[key] = value
        
        return Parameters(
            geometry=params_dict.get('geometry', 0),
            rmax=params_dict.get('rmax', 0.0),
            pname=params_dict.get('pname', ''),
            xuv_int=params_dict.get('xuvint', 0.0),
            rtherm=params_dict.get('rtherm', 0.0),
            rbase=params_dict.get('rbase', 0.0),
            tem_base=params_dict.get('tembase', 0.0),
            pbase=params_dict.get('pbase', 0.0),
            max_flux=params_dict.get('maxflux', 0.0),
            tem_min=params_dict.get('temmin', 0.0),
            tem_max=params_dict.get('temmax', 0.0),
            h2a=params_dict.get('h2a', 0.0),
            helium=params_dict.get('helium', 0.0),
            he_m2s=params_dict.get('hem2s', 0.0),
            ewoo=params_dict.get('ewoo', 0.0),
            vsw=params_dict.get('vsw', 0.0),
            rsw=params_dict.get('rsw', 0.0),
            msw=params_dict.get('msw', 0.0),
            tmax=params_dict.get('tmax', 0.0),
            dt_ris=params_dict.get('dtris', 0.0),
            dt_save=params_dict.get('dtsave', 0.0),
            tstart=params_dict.get('tstart', 0.0)
        )
        
    except Exception as e:
        print(f"Ошибка парсинга файла {file_path}: {e}")
        return None

def find_experiment_folders(root_dir: Path) -> List[Path]:
    """Находит все папки с файлами absorbtion.dat и parameters.txt"""
    experiment_folders = []
    
    # Рекурсивный поиск всех файлов absorbtion.dat
    
    for absorption_file in root_dir.rglob('Absorption.dat'):
        folder = absorption_file.parent
        
        parameters_file = folder / 'parameters.txt'
        
        if parameters_file.exists():
            experiment_folders.append(folder)
    
    return experiment_folders

def parse_experiments_from_folders(experiment_folders: List[Path]) -> List[ExperimentData]:
    """Парсит эксперименты из списка папок"""
    experiments = []
    
    for folder in experiment_folders:
        absorption_file = folder / 'Absorption.dat'
        parameters_file = folder / 'parameters.txt'
        
        absorption_data = parse_absorption_file(absorption_file)
        parameters_data = parse_parameters_file(parameters_file)
        
        if absorption_data and parameters_data:
            experiment = ExperimentData(
                absorption=absorption_data,
                parameters=parameters_data,
                folder_path=folder
            )
            experiments.append(experiment)
    
    return experiments

def find_and_parse_experiments(root_dir: Path) -> List[ExperimentData]:
    """
    Рекурсивно ищет и парсит все эксперименты в указанной директории
    """
    if not root_dir.exists():
        print(f"Директория не существует: {root_dir}")
        return []
    
    if not root_dir.is_dir():
        print(f"Указанный путь не является директорией: {root_dir}")
        return []
    
    print(f"Поиск экспериментов в: {root_dir.absolute()}")
    experiment_folders = find_experiment_folders(root_dir)
    
    print(f"Найдено папок с экспериментами: {len(experiment_folders)}")
    
    return parse_experiments_from_folders(experiment_folders)

def display_experiment_info(experiments: List[ExperimentData]) -> None:
    """Отображает информацию о найденных экспериментах"""
    if not experiments:
        print("Эксперименты не найдены")
        return
    
    print(f"\n{'='*50}")
    print(f"НАЙДЕНО ЭКСПЕРИМЕНТОВ: {len(experiments)}")
    print(f"{'='*50}")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\nЭксперимент #{i}:")
        print(f"  Папка: {exp.folder_path}")
        print(f"  Планета: {exp.parameters.pname}")
        print(f"  Точек данных: {len(exp.absorption.time)}")
        print(f"  доплеровская скорость: {exp.absorption.time[0]:.1f} - {exp.absorption.time[-1]:.1f}")
        print(f"  Rmax: {exp.parameters.rmax}")
        print(f"  Geometry: {exp.parameters.geometry}")

def parse_experiments_data(root:str|Path):
    """Основная функция программы"""
    try:
        root_input = Path(root)
        root_directory = Path(root_input).expanduser().resolve()
        
        experiments = find_and_parse_experiments(root_directory)
        display_experiment_info(experiments)
            
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")

    return experiments




def display_parameters_statistics(data :List[ExperimentData]):
    result = set()
    for element in data:
        h2a, helium, msw , xuv =  element.parameters.h2a, element.parameters.helium, element.parameters.msw, element.parameters.xuv_int
        identificator = TargetParamsSample(
            xuv=xuv,
            msw= msw,
            helium=helium,
            h2a=h2a
        )
        # identidicator = f"xuv{xuv}_msw{msw}_hel{helium}_h2a{h2a}"

        result.add(identificator)
    return result