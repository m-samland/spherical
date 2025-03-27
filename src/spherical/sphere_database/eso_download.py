import os
import urllib
from collections import OrderedDict
from glob import glob

import numpy as np
from astropy.io import fits
from astropy.table import Table, setdiff
from bs4 import BeautifulSoup
from tqdm import tqdm


def make_obs_program_id_table(table_of_observations):
    prog_ids = table_of_observations['OBS_PROG_ID']
    unique_ids = np.unique(prog_ids)
    table = Table()
    table.add_column(unique_ids)
    return table


def make_list_from_1d_table(table):
    lst = []
    for row in table:
        lst.append(row[0])
    return lst


def mask_all_containing_strings_from_list(table, list_of_strings):
    mask = np.zeros(len(table), dtype='bool')
    for idx in range(len(mask)):
        value = table[idx]
        if any(map(value.__contains__, list_of_strings)):
            mask[idx] = True
    return mask


def query_file(filename):
    url = "http://archive.eso.org/hdr?DpId={}".format(filename)
    html = urllib.request.urlopen(url)
    content = html.read().decode('utf-8', 'ignore')
    # ipsh()
    soup = BeautifulSoup(content, "lxml")
    header_text = soup.text

    return header_text


def header_text_to_dictionary(header_text):
    index_begin = header_text.index("SIMPLE")
    index_end = header_text.index("\nEND")
    header_text = header_text[index_begin:index_end]
    header_text = header_text.splitlines()
    header_text = [item.rsplit('/')[0] for item in header_text]
    header_text = [item.split('=', 1) for item in header_text]

    for idx, item in enumerate(header_text):
        header_text[idx][0] = item[0].strip()
        value_string = item[1].strip()
        # if len(value_string) == 1:
        #     if value_string == 'T':
        #         header_text[idx][1] = True
        #     elif value_string == 'F':
        #         header_text[idx][1] = False
        if any(c.isalpha() for c in value_string) or  \
                "-" in value_string[1:] or \
                value_string.count('.') > 1 or \
                "'" in value_string:
            header_text[idx][1] = value_string.replace("'", "").replace('"', "")
        elif '.' in value_string:
            header_text[idx][1] = float(value_string)
        else:
            try:
                header_text[idx][1] = int(value_string)
            except ValueError:
                pass

    return OrderedDict(header_text)


def write_fits(filename, outputdir, header_dictionary):
    header = fits.Header(header_dictionary)
    primary_hdu = fits.PrimaryHDU(header=header)
    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(
        os.path.join(outputdir, "{}.fits".format(filename)), overwrite=True)


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1'}
eso_filename_key = 'DP.ID'
eso_prog_id_key = 'ProgId'
# gto_observation_table_path = \
#     '/home/samland/science/python/projects/esorexer/tables/table_of_observations_gto.fits'
# eso_archive_table = '/home/samland/science/python/projects/esorexer/wdb_2014_to_2016_all.tsv'
# download_path = '/home/samland/science/python/projects/esorexer/download_demo/'

# gto_observation_table_path = \
# '/mnt/fhgfs/sphere_shared/automatic_reduction/esorexer/tables/tables/table_of_observations_gto.fits'
# eso_archive_table = \
# '/mnt/fhgfs/sphere_shared/automatic_reduction/non_gto/wdb_2014_to_2016_all.tsv'
# download_path = '/mnt/fhgfs/sphere_shared/automatic_reduction/non_gto/2015/'

# gto_observation_table = Table.read(gto_observation_table_path)
# gto_program_IDs = make_obs_program_id_table(gto_observation_table)
# gto_program_IDs = make_list_from_1d_table(gto_program_IDs)
# sphere_obs = Table.read(
#     eso_archive_table, format='ascii', delimiter='\t', fast_reader=False,
#     guess=False, header_start=1)

# mask_zimpol = sphere_obs['SEQ ARM'] == 'ZIMPOL'
# mask_ifs = sphere_obs['SEQ ARM'] == 'IFS'
# mask_gto = mask_all_containing_strings_from_list(
#     table=sphere_obs[eso_prog_id_key], list_of_strings=gto_program_IDs)
#
# mask = np.logical_or.reduce([mask_zimpol, mask_ifs, mask_gto])

# non_gto_files = sphere_obs[eso_filename_key][~mask]
#
# non_gto_files = non_gto_files[10:]

# list_of_headers = []
# failed_files = []
# number_of_files = len(non_gto_files)
# for idx, non_gto_file in enumerate(tqdm(non_gto_files)):
#     try:
#         header_text = query_file(non_gto_file)
#         header_dictionary = header_text_to_dictionary(header_text)
#         write_fits(non_gto_file, download_path, header_dictionary)
#         list_of_headers.append(header_dictionary)
#     except:
#         failed_files.append(non_gto_file)

# a = Table.read('table_of_files_non_gto_2014_to_2017.fits')
# shutter = []
# for f in a[0]:
#     if f['SHUTTER'] == 'T':
#         shutter.append(True)
#     else:
#         shutter.append(False)
#
#
# table_of_observations, table_of_targets = observation_table.create_observation_table(
#     table_of_files, table_of_targets)

# download_table = Table.read('/home/masa4294/science/database_sphere/irdis_arm_2020_07_07.csv')
# download_table = Table.read('/home/masa4294/science/database_sphere/ifs_arm_2020_07_07.csv')
# download_table = Table.read('/home/masa4294/science/database_sphere/irdis_calibrations_2020_11_11.csv')
# download_table = Table.read('/home/masa4294/science/database_sphere/ifs_calibrations_2020_11_05.csv')
# download_table = Table.read('/home/masa4294/science/database_sphere/irdis_2020_07_06_to_2021_07_19.csv')
# download_table = Table.read('/home/samland/database_sphere/irdis_2020_07_06_to_2021_07_19.csv')
download_table = Table.read('/home/samland/database_sphere/ifs_2020_07_06_to_2021_07_19.csv')
# existing_files = glob('/home/masa4294/science/python/projects/esorexer/opentime_headers/SPHER.*.fits')
# existing_files = glob('/home/masa4294/science/database_sphere/headers_irdis/SPHER.*.fits')
existing_files = glob(
    '/home/samland/database_sphere/headers_ifs/SPHER.*.fits')
# existing_files = glob(
#     '/home/masa4294/science/database_sphere/headers_irdis/SPHER.*.fits')
# existing_files = glob('/home/masa4294/science/database_sphere/headers_ifs/SPHER.*.fits')

# existing_files = existing_files + existing_files2

existing_names = []
for file in existing_files:
    file_name = os.path.split(file)[1][:-5]
    existing_names.append(file_name)
existing_file_table = Table({'DP.ID': existing_names})
download_list_new = setdiff(download_table, existing_file_table, keys=['DP.ID'])

# download_path = '/home/masa4294/science/database_sphere/headers_irdis/'
download_path = '/home/samland/database_sphere/headers_ifs/'
# download_path = '/home/masa4294/science/database_sphere/headers_irdis/'
# download_path = '/home/masa4294/science/database_sphere/headers_ifs/'

# list_of_headers = []
# failed_files = []
number_of_files = len(download_list_new)
for idx, file_id in enumerate(tqdm(download_list_new['DP.ID'])):
    try:
        header_text = query_file(file_id)
        header_dictionary = header_text_to_dictionary(header_text)
        write_fits(file_id, download_path, header_dictionary)
        # list_of_headers.append(header_dictionary)
    except:
        pass
        # failed_files.append(file_id)
