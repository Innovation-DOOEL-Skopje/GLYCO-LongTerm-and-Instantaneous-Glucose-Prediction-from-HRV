import importlib, module_imports
importlib.reload(module_imports)
from module_imports import *



def create_initial_dirs_and_files(model_dir: Union[str, module_paths.Path]) -> Dict[str, module_paths.Path]:

    if type(model_dir) is str:
        model_dir_string = model_dir
    if typ(model_dir) is module_paths.Path:
        model_dir_string = model_dir.path_string

    # dirs
    results_dir = module_paths.Path(f'{model_dir_string}/Results_{module_time.ymd()}'); results_dir.create_dir(warn_exists = False)
    cv_results_dir = module_paths.Path(f'{model_dir_string}/CVResults_{module_time.ymd()}'); cv_results_dir.create_dir(warn_exists = False)
    exception_dir = module_paths.Path(f'{model_dir_string}/Exceptions'); exception_dir.create_dir(warn_exists = False)
    metadata_dir = module_paths.Path(f'{model_dir_string}/Metadata'); metadata_dir.create_dir(warn_exists = False)
    metadata_folds_dir = module_paths.Path(f'{model_dir_string}/Metadata\\Folds'); metadata_folds_dir.create_dir(warn_exists = False)
    metadata_search_space_dir = module_paths.Path(f'{model_dir_string}/Metadata/SearchSpace'); metadata_search_space_dir.create_dir(warn_exists = False)
    pretrained_models_dir = module_paths.Path(f'{model_dir_string}/ModelsData/PretrainedModels'); pretrained_models_dir.create_dir(warn_exists = False)

    # files
    exceptions_file_path = f'{exception_dir.path_string}/Exceptions_{SCRIPT_ID}.txt'

    dirs_and_files_dict = dict(
        model_dir = model_dir,
        fold_results_dir = results_dir,
        cv_results_dir = cv_results_dir,
        exceptions_dir = exceptions_dir,
        metadata_dir = metadata_dir,
        meatadata_folds_dir = metadata_folds_dir,
        metadata_search_space_dir = metadata_search_space_dir,
        pretrained_models_dir = pretrained_models_dir
    )

    return dirs_and_files_dict

