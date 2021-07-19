import os
import pandas as pd
import module_debug
import importlib as implib
from typing import List, Dict, Tuple, Union
import module_dataset_analysis; implib.reload(module_dataset_analysis)
import module_io; implib.reload(module_io)

p = print

#____________________________________________________________________________________________________________________________________


def rename_hrv(df: pd.DataFrame
               ) -> pd.DataFrame:
    """
    :param df: df with standardized glyco features

    :return dataframe with renamed columns
    """

    # get column names
    column_names = df.columns.to_list()

    # rename columns
    for i in range(len(column_names)):
        column_names[i] = column_names[i].replace('(‰)', '')\
            .replace(' Prima', '-1').replace(' Secunda', '-2').replace(' Tertia', '-3')

    # assign column names
    df.columns = column_names

    return df

#____________________________________________________________________________________________________________________________________


def read_test_ecg_lt3c(source_dir: str,
                       intervals: List[str],
                       drop_frequency_domain: bool = True,
                       stop_after: Union[int, float] = float('inf'),
                       write_dir: str = None) -> Union[Tuple[dict, pd.DataFrame], None]:
    """
    NOTE: to be refactored to work with all datasets

    Parse files downloaded from test.ecg

    :param source_dir: dir of test.ecg files
    :param intervals: the intervals for which to parse the files, with corresponding time unit. ex: ['1h', '2h, ... '24h']
    :param drop_frequency_domain: if True, don't consider frequency domain parameters
    :param stop_after: stop after certain amount of patients if debugging or some other rechecking is needed
    :param write_dir: write directory

    :return: if no write directory is specified, the parsed dataframes are returned together with the samples and patients info
    """

    # initialize paths and files
    source = source_dir
    patients_classification_path = 'C:\\Users\\ilija\\Pycharm Projects\\GLYCO\\DATA\\Classification_of_Patients\\NewClasses__ND_GD_BD.csv'
    patients_classification = pd.read_csv(patients_classification_path)
    patients_hba1c_path = 'C:\\Users\\ilija\\Pycharm Projects\\GLYCO\\DATA\\Clinical_Records\\PatientID_Hba1C_06102020.csv'
    patients_hba1c = pd.read_csv(patients_hba1c_path)

    # print classes info
    module_debug.line()
    p('Total number of patients in the classification file:')
    p(patients_classification['Patient_ID'].nunique())
    p('Number of non-diabetic patients:')
    p(patients_classification[patients_classification['Class'] == 'ND']['Patient_ID'].nunique())
    p('Number of diabetic patients:')
    p(patients_classification[patients_classification['Class'] != 'ND']['Patient_ID'].nunique())
    p('Number of diabetic patients with good regulation:')
    p(patients_classification[patients_classification['Class'] == 'GD']['Patient_ID'].nunique())
    p('Number of diabetic patients with bad regulation:')
    p(patients_classification[patients_classification['Class'] == 'BD']['Patient_ID'].nunique())
    module_debug.line()

    # print patients info
    module_debug.line()
    p('Number of patients in source directory: ')
    p(len(os.listdir(source)))
    module_debug.line()

    # initialize inspection dict, that makes sure everything works as it should
    inspection_dict = dict(
            total_samples = 0,
            all_patients = [i[:-5] for i in os.listdir(source)]
    )

    for interval in intervals:
        inspection_dict[f'samples_{interval}'] = 0
        inspection_dict[f'samples_{interval}'] = 0
        inspection_dict[f'patients_without_{interval}'] = list()
        inspection_dict[f'patients_without_{interval}'] = list()
        inspection_dict[f'patients_with_{interval}'] = list()
        inspection_dict[f'patients_with_{interval}'] = list()

    # dict for final files
    final_datasets_dict = dict()

    # loop control
    is_initial_patient = True

    # iterate file patients
    for file_name in os.listdir(source):

        # parse patient's ID
        patient_id = file_name[:-5]
        p(patient_id)

        # read individual xlsx file
        patient_records = pd.read_excel(f'{source}\\{file_name}', skiprows = 4)

        patient_records.drop(labels = ['External Link'], axis = 1, inplace = True)

        # remove unwanted columns
        if drop_frequency_domain:
            patient_records.drop(labels = ['HF', 'LF',  'LF/HF', 'ULF', 'VLF'], axis = 1, inplace = True)

        # assign Patient_ID, hba1c and regulation
        patient_records['Patient_ID'] = patient_id
        patient_records['Class'] = patients_classification[patients_classification['Patient_ID'] == patient_id]['Class'].iloc[0]
        patient_records['HbA1C(%)'] = patients_hba1c[patients_hba1c['Patient_ID'] == patient_id]['HbA1C(%)'].iloc[0]

        # split the dataset by hour
        patient_split_datasets_dict = dict()
        records_splits = list()

        # filter samples for current interval
        for interval in intervals:
            patient_split_datasets_dict[f'{interval}'] = patient_records[patient_records["Dataset Name"].str.startswith(f"Period: {interval}")]
            records_splits.append(patient_split_datasets_dict[f'{interval}'])

        # update inspection for total samples
        inspection_dict['total_samples'] += patient_records.shape[0]

        for interval in intervals:

            # update inspection for interval samples
            inspection_dict[f'samples_{interval}'] += patient_split_datasets_dict[f'{interval}'].shape[0]

            # if df is empty, add patient to the ones without
            if patient_split_datasets_dict[f'{interval}'].shape[0] == 0: inspection_dict[f'patients_without_{interval}'].append(patient_id)
            else: inspection_dict[f'patients_with_{interval}'].append(patient_id)

        assert patient_records.shape[0] == sum([df_iter.shape[0] for df_iter in patient_split_datasets_dict.values()]), 'Incorrect split on intervals'

        # if split is good, don't need patient_records
        del patient_records

        # rearrange dataframes
        for records_split_iter in records_splits:

            # drop dataset name
            records_split_iter.drop(labels = ['Dataset Name'], axis = 1, inplace = True)

            # rearange columns
            for feature_to_move_end in ['Start Date', 'End Date']:
                records_split_iter.insert(records_split_iter.shape[1] - 1, feature_to_move_end, records_split_iter.pop(feature_to_move_end))

            for feature_to_move_start in ['HbA1C(%)', 'Class', 'Patient_ID']:
                records_split_iter.insert(0, feature_to_move_start, records_split_iter.pop(feature_to_move_start))

        # add to datasets that will be saved
        if is_initial_patient:
            for interval in intervals:
                final_datasets_dict[f'{interval}'] = patient_split_datasets_dict[f'{interval}']
                is_initial_patient = False
        else:
            for interval in intervals:
                final_datasets_dict[f'{interval}'] = final_datasets_dict[f'{interval}'].append(patient_split_datasets_dict[f'{interval}'],
                                                                                               ignore_index= True)

        # stopping after certain amount of patients
        stop_after -= 1
        if stop_after <= 0: break

    samples_list = list()
    for interval in intervals:
        samples_dict = module_dataset_analysis.quantitative_analysis(df = final_datasets_dict[f'{interval}'],
                                                                    dataset_name = interval,
                                                                    class_feature = 'Class',
                                                                    classes = ['ND', 'GD', 'BD'])
        samples_list.append(samples_dict)

    samples_df = pd.DataFrame(samples_list)

    total_samples_in_dfs = 0
    for interval in intervals:
        assert inspection_dict[f'samples_{interval}'] == final_datasets_dict[f'{interval}'].shape[0], 'Problem with number of samples in 12h dataset'
        total_samples_in_dfs += final_datasets_dict[f'{interval}'].shape[0]
    assert inspection_dict[f'total_samples'] == total_samples_in_dfs, 'Problem with number of samples in all datasets'

    module_debug.line()
    for interval in intervals:
        p(f'Patients without {interval} measurements ({inspection_dict[f"patients_without_{interval}"].__len__()}):')
        p(inspection_dict[f'patients_without_{interval}'])
        p(f'Patients with {interval} measurements ({inspection_dict[f"patients_with_{interval}"].__len__()}):')
        p(inspection_dict[f'patients_without_{interval}'])
        p(f'Patients with and without {interval} measurements:')
        p(inspection_dict[f"patients_with_{interval}"].__len__() + inspection_dict[f"patients_without_{interval}"].__len__())

    # module_debug.line()
    # p('All patients: ')
    # all_patients = [i[:-5] for i in os.listdir(source)]
    # print(set(all_patients).__len__())
    #

    if write_dir:
        for interval in intervals:
            module_io.to_excel(final_datasets_dict[f'{interval}'],
                              path = f'{write_dir}\\hrv_{interval}')

        module_io.to_excel(df = samples_list,
                          path = f'{write_dir}\\summary')
    else:
        return final_datasets_dict, samples_df





