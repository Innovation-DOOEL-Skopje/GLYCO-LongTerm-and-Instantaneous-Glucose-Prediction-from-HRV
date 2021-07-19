import os
import sys
from datetime import datetime
import module_lists_strings
from typing import List, Union

#____________________________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________________________


class Path:

    path: str

    def __init__(self, path):
        self.path = path

    #____________________________________________________________________________________________________________________________________

    @property
    def to_string(self) -> str:
        return self.path

    #____________________________________________________________________________________________________________________________________

    def create_dir(self, warn_exists: bool = True) -> None:

        # create if does not exist
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # if exists
        else:

            if warn_exists:

                answer = str(input(f"""The directory {self.path} that you are trying to write in already exists.
                                       Some files may be overwritten.
                                       Are you sure you want to continue? [y/n]"""))

                if answer.lower() == 'y': pass
                else: sys.exit()

            else: pass
    #____________________________________________________________________________________________________________________________________

    @property
    def last_dir_or_file(self) -> str:
        """Get the last dir from path"""

        return self.path.split(os.path.sep)[-1]

    #____________________________________________________________________________________________________________________________________

    @property
    def descriptive_path_from_super_onwards(self) -> str:
        """Get parent/current string"""

        return '-'.join(self.path.split(os.path.sep)[-2:])

    #____________________________________________________________________________________________________________________________________

    def list_sorted(self, absolute: bool = False) -> List[str]:
        """Get list of alphanumerically sorted files in a directory with their absolute paths"""

        file_names = os.listdir(self.path)
        file_names = module_lists_strings.sortl_list_of_strings_alphanumerically(file_names)

        if absolute:
            file_names = [f'{self.path}\\{file_name_i}' for file_name_i in file_names]

        return file_names
#____________________________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________________________


def create_directory(directory: str, warn_exists: bool = True):

    if not os.path.exists(directory):

        os.makedirs(directory)

    else:

        if warn_exists:

            answer = str(input(f"""The directory {directory} that you are trying to write in already exists.
                                  Some files may be overwritten.
                                  Are you sure you want to continue? [y/n]"""))

            if answer.lower() == 'y': pass
            else: sys.exit()

        else: pass
        
    return directory

#____________________________________________________________________________________________________________________________________


def create_plots_dir_relative(description: str = '',
                              warn_exists: bool = True,
                              path_class: bool = False,
                              ) -> Union[str, Path]:
    """

    :param description:
    :param warn_exists:
    :param path_class: whether to return an instance of the Path class or a string

    :return:
    """

    current_date = datetime.now().strftime("%d.%m")
    plots_dir = f"Plots\\{current_date}_Plots_{description}"

    if not os.path.exists(plots_dir):

        os.makedirs(plots_dir)

    else:

        if warn_exists:

            answer = str(input(f"""The directory {plots_dir} that you are trying to write in already exists.
                                  Some files may be overwritten.
                                  Are you sure you want to continue? [y/n]"""))

            if answer.lower() == 'y': pass
            else: sys.exit()

        else: pass

    if path_class:
        return Path(plots_dir)
    else:
        return plots_dir

#____________________________________________________________________________________________________________________________________


def create_tables_dir_relative(description: str = '',
                               warn_exists: bool = True,
                               path_class: bool = False,
                               ) -> Union[str, Path]:
    """

    :param description:
    :param warn_exists:
    :param path_class: whether to return an instance of the Path class or a string

    :return:
    """

    current_date = datetime.now().strftime("%d.%m")
    tables_dir = f"Tables\\{current_date}_Tables_{description}"

    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)

    else:
        if warn_exists:
            answer = str(input(f"""The directory {tables_dir} that you are trying to write in already exists.
                                  Some files may be overwritten.
                                  Are you sure you want to continue? [y/n]"""))

            if answer.lower() == 'y': pass
            else: sys.exit()

        else: pass

    if path_class:
        return Path(tables_dir)
    else:
        return tables_dir

#____________________________________________________________________________________________________________________________________


def create_results_dir_relative(description: str = '',
                                warn_exists: bool = True,
                                path_class: bool = False,
                                ) -> Union[str, Path]:
    """

    :param description:
    :param warn_exists:
    :param path_class: whether to return an instance of the Path class or a string

    :return:
    """

    current_date = datetime.now().strftime("%d.%m")
    results_dir = f"Results\\{current_date}_Results_{description}"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    else:
        if warn_exists:
            answer = str(input(f"""The directory {results_dir} that you are trying to write in already exists.
                                  Some files may be overwritten.
                                  Are you sure you want to continue? [y/n]"""))

            if answer.lower() == 'y': pass
            else: sys.exit()

        else: pass

    if path_class:
        return Path(results_dir)
    else:
        return results_dir

#____________________________________________________________________________________________________________________________________


def create_cvresults_dir_relative(description: str = '',
                                  warn_exists: bool = True,
                                  path_class: bool = False,
                                  ) -> Union[str, Path]:
    """
    Create a CV_Results directory in the same folder as the script

    :param description: description to be appended to CV_Results_
    :param warn_exists: whether to warn if the directory exists and files may be overwritten
    :param path_class: whether to return an instance of the Path class or a string

    :return: string or Path
    """

    current_date = datetime.now().strftime("%d.%m")
    cv_results_dir = f"CV_Results\\{current_date}_CV_Results_{description}"

    if not os.path.exists(cv_results_dir):
        os.makedirs(cv_results_dir)

    else:
        if warn_exists:
            answer = str(input(f"""The directory {cv_results_dir} that you are trying to write in already exists.
                                  Some files may be overwritten.
                                  Are you sure you want to continue? [y/n]"""))

            if answer.lower() == 'y': pass
            else: sys.exit()

        else: pass

    if path_class:
        return Path(cv_results_dir)
    else:
        return cv_results_dir

#____________________________________________________________________________________________________________________________________


def create_metadata_dir_relative(description: str = '',
                                 warn_exists: bool = True,
                                 path_class: bool = False,
                                 ) -> Union[str, Path]:
    """
    Create a Model_Data directory in the same folder as the script

    :param description: description to be appended to Model_Data
    :param warn_exists: whether to warn if the directory exists and files may be overwritten
    :param path_class: whether to return an instance of the Path class or a string

    :return: string or Path
    """

    model_metadata_dir = f"Metadata"
    if description: model_metadata_dir += f"_{description}"

    if not os.path.exists(model_metadata_dir):
        os.makedirs(model_metadata_dir)

    else:
        if warn_exists:
            answer = str(input(f"""The directory {model_metadata_dir} that you are trying to write in already exists.
                                  Some files may be overwritten.
                                  Are you sure you want to continue? [y/n]"""))

            if answer.lower() == 'y': pass
            else: sys.exit()

        else: pass

    if path_class:
        return Path(model_metadata_dir)
    else:
        return model_metadata_dir

