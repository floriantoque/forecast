import pandas as pd
import wget
import time
import os
import zipfile
import numpy as np
import shutil

from argparse import ArgumentParser

from  tqdm import tqdm
import sys
sys.path.insert(0, '../../utils')
from utils_date import *



def download_file(url, filename_path):

    directory = os.path.dirname(filename_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    start_time = time.time()
    filename = wget.download(url ,filename_path)
    print("--- %s seconds ---" % (time.time() - start_time))

    return filename


def main(path_to_save, full_path_date, full_path_save):

    print('Load data from open data paris API and remove them')


    url_data = 'https://opendata.stif.info/explore/dataset/histo-validations/files/ff00c4d0d1ac37b823f5ba854a895d8e/download/'
    filename_data = download_file(url_data, path_to_save + 'data_2017S1.zip')
    zip_ref = zipfile.ZipFile(filename_data, 'r')
    zip_ref.extractall(os.path.dirname(filename_data) + '/')
    zip_ref.close()

    print(filename_data)
    os.remove(filename_data)
    os.remove(path_to_save + 'DATA PUBLIEES S1 2017/2017S1_NB_SURFACE.txt')
    os.remove(path_to_save + 'DATA PUBLIEES S1 2017/2017S1_PROFIL_SURFACE.txt')

    filename_profil = path_to_save + 'DATA PUBLIEES S1 2017/2017S1_PROFIL_FER.txt'
    filename_validations = path_to_save + 'DATA PUBLIEES S1 2017/2017S1_NB_FER.txt'

    df_validations = pd.read_csv(filename_validations, sep='\t')
    df_profil = pd.read_csv(filename_profil, sep='\t')

    shutil.rmtree(path_to_save + 'DATA PUBLIEES S1 2017/', ignore_errors=True)

    url_data = 'https://opendata.stif.info/explore/dataset/emplacement-des-gares-idf/download/?format=csv&timezone=America/New_York&use_labels_for_header=true'
    filename_data = download_file(url_data, path_to_save + 'station_information.csv')

    df_station = pd.read_csv(filename_data, sep=';')

    print(filename_data)
    os.remove(filename_data)


    print('Clean data')




    # Modif df_validations

    df_validations = df_validations[['JOUR', 'ID_REFA_LDA', 'NB_VALD', 'CODE_STIF_ARRET']]
    df_validations = df_validations.dropna()
    df_validations['NB_VALD'] = np.array([1 if i[0] == 'M' else i for i in df_validations['NB_VALD'].values]).astype(
        int)
    df_validations = df_validations[df_validations.ID_REFA_LDA != '?']
    df_validations['CODE_STIF_ARRET'] = df_validations['CODE_STIF_ARRET'].values.astype(str)
    df_validations.ID_REFA_LDA = df_validations.ID_REFA_LDA.values.astype(int).astype(str)
    df_validations['JOUR'] = [i.split('/')[2] + '-' + i.split('/')[1] + '-' + i.split('/')[0] for i in
                              df_validations['JOUR']]

    # Modif df_profil
    df_profil = df_profil[['TRNC_HORR_60', 'ID_REFA_LDA', 'CAT_JOUR', 'pourc_validations']]
    df_profil = df_profil.dropna()
    df_profil = df_profil[df_profil.TRNC_HORR_60 != 'ND']
    df_profil = df_profil[df_profil.ID_REFA_LDA != '?']
    df_profil.ID_REFA_LDA = df_profil.ID_REFA_LDA.values.astype(int).astype(str)
    df_profil['TRNC_HORR_60'] = df_profil['TRNC_HORR_60'].values.astype(str)
    df_profil = df_profil.drop_duplicates()

    # Modif df_station
    df_station['GARES_ID'] = df_station['GARES_ID'].values.astype(str)



    print('Create csv file with information about stations')

    dict_idrl_csa = dict(df_validations[['ID_REFA_LDA', 'CODE_STIF_ARRET']].drop_duplicates().values)
    dict_csa_idrl = dict(df_validations[['CODE_STIF_ARRET', 'ID_REFA_LDA']].drop_duplicates().values)

    df_info = df_station[['MODE_', 'LIGNE', 'GARES_ID', 'NOM_GARE']].set_index('GARES_ID').join(
        df_validations[['CODE_STIF_ARRET', 'ID_REFA_LDA']].drop_duplicates().set_index(
            ('CODE_STIF_ARRET'))).dropna().reset_index().drop_duplicates()
    df_info.to_csv(path_to_save + 'station_information.csv', index=False)


    print('Create dictionaries')

    # 1 - dict_idstation_typeofday_percentage with df_profil
    trnc_horr_60 = ['0H-1H', '1H-2H', '2H-3H', '3H-4H', '4H-5H', '5H-6H', '6H-7H', '7H-8H', '8H-9H', '9H-10H',
                    '10H-11H', '11H-12H', '12H-13H', '13H-14H', '14H-15H', '15H-16H', '16H-17H', '17H-18H',
                    '18H-19H', '19H-20H', '20H-21H', '21H-22H', '22H-23H', '23H-0H']

    time_60 = [i[11:] for i in
               build_timestamp_list("2000-01-01 00:00:00", "2000-01-01 23:00:00", time_step_second=60 * 60)]

    df = pd.pivot_table(df_profil, values=['pourc_validations'],
                        index=['ID_REFA_LDA', 'CAT_JOUR'],
                        columns='TRNC_HORR_60', fill_value=0)

    df.columns = df.columns.droplevel(0)
    df.columns.name = None
    df = df.reset_index()
    df = df[['ID_REFA_LDA', 'CAT_JOUR'] + trnc_horr_60]

    list_index_incomplete = df['ID_REFA_LDA'].drop_duplicates().values[
        df[['ID_REFA_LDA']].groupby('ID_REFA_LDA').size().values != 5]

    for i in list_index_incomplete:
        df = df[df['ID_REFA_LDA'] != i]

    dict_idstation_typeofday_percentage = dict([((i[0], i[1]), np.array(i[2:]).astype(float)) for i in df.values])

    # 2 - dict_idstation_day_sumvalidation with df_profil

    df = df_validations.groupby(['JOUR', 'ID_REFA_LDA']).sum().reset_index()
    df = df[['ID_REFA_LDA', 'JOUR', 'NB_VALD']]
    dict_idstation_day_sumvalidation = dict([((i[0], i[1]), i[2]) for i in df.values])

    # 3 - dict_day_category1
    df_exogenous = pd.read_csv(full_path_date)
    dict_day_category1 = dict([(i[:10], j) for i, j in df_exogenous[['Datetime', 'Category1']].values])





    print('Create and reshape dataframe datetime_values')


    id_stations_validations = np.array(sorted(set([i[0] for i in dict_idstation_day_sumvalidation.keys()])))
    id_stations_profil = np.array(sorted(set([i[0] for i in dict_idstation_typeofday_percentage.keys()])))
    id_stations = list(filter(lambda x: x in id_stations_validations, id_stations_profil))
    day_list = sorted(df_validations['JOUR'].drop_duplicates().tolist())

    dict_trnc_time = dict(zip(trnc_horr_60, time_60))

    values = [build_timestamp_list(day + " 00:00:00", day + " 23:00:00", time_step_second=60 * 60)
              for day in day_list]
    values = [item for sublist in values for item in sublist]
    df_datetime_values = pd.DataFrame(data=values, columns=['Datetime'])

    columns = ['Date'] + trnc_horr_60
    list_df = []
    value_incomplete = 0

    for ids in tqdm(id_stations):
        data = []
        for day in day_list:
            typeofday = dict_day_category1[day]
            try:
                v = dict_idstation_day_sumvalidation[(ids, day)] * dict_idstation_typeofday_percentage[(ids, typeofday)] / 100
            except:
                value_incomplete = +1
                v = np.zeros(24)
            data.append([day] + list(v))

        df = pd.DataFrame(data=data, columns=columns)

        df = df.set_index("Date").stack().reset_index()
        df.columns = ['Date', 'trnc_horr_60', ids]
        df['time'] = [dict_trnc_time[i] for i in df['trnc_horr_60'].values]
        df['Datetime'] = [i + " " + j for i, j in df[['Date', 'time']].values]
        list_df.append(df[['Datetime', ids]].set_index('Datetime'))

    print("Value incomplete {:.4f}%".format((value_incomplete / (len(id_stations) * len(day_list) * 1.)) * 100))

    df_datetime_values = df_datetime_values.set_index('Datetime').join(list_df).reset_index()


    print('Filter on stations of Metro line 1 and 7 for the test')

    df_station[['MODE_', 'LIGNE', 'GARES_ID', 'NOM_GARE']].set_index('GARES_ID')
    df_station_ = df_station[['MODE_', 'LIGNE', 'GARES_ID']].groupby(['MODE_', 'LIGNE'])['GARES_ID'].apply(
        lambda x: "%s" % ','.join(x)).reset_index()
    dict_modeligne_csa = dict(
        [((i[0], i[1]), i[2].split(',')) for i in df_station_[['MODE_', 'LIGNE', 'GARES_ID']].values])

    filter_modeligne = [('Metro', '1'), ('Metro', '7')]
    filter_station_id = []
    for i in filter_modeligne:

        for j in dict_modeligne_csa[i]:

            try:
                filter_station_id.append(dict_csa_idrl[j])
            except:
                print("Fail to link station with CodeStifArret {} with IdRefaLda".format(j))

    filter_station_id_ = []
    for i in filter_station_id:
        if i in id_stations:
            filter_station_id_.append(i)
        else:
            print("Fail to link station id {} with datetime_values file".format(i))

    df_datetime_values_filtered = df_datetime_values[['Datetime'] + filter_station_id_]
    df_datetime_values_filtered.to_csv(full_path_save, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--path_to_save", default="../../data/")
    parser.add_argument("-fpd", "--full_path_date", default="../../data/date_file_2013_2018_included_test.csv")
    parser.add_argument("-fps", "--full_path_save",
                        default="../../data/observation_file_2017-01-01_2017-06-30_included_test.csv")

    args = parser.parse_args()
    path_to_save = args.path_to_save
    full_path_date = args.full_path_date
    full_path_save = args.full_path_save

    main(path_to_save, full_path_date, full_path_save)

