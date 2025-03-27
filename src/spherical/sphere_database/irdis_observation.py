import os
from collections import OrderedDict

from astropy import units as u
from astropy.table import Table, vstack
from astropy.time import Time, TimeDelta

# from trap.embed_shell import ipsh
from .database_utils import filter_table, find_nearest


class IRDIS_observation(object):
    def __init__(self, observation=None, science_files=None, calibration_table=None):
        self.observation = observation
        self._calibration = calibration_table
        self._science_files = science_files

        self._target_name = " ".join(
            self.observation['MAIN_ID'][0].split())
        self._target_name = self._target_name.replace(" ", "_")
        self._file_prefix = '{0}_{1}_{2}_'.format(
            self._target_name,
            self.observation['NIGHT_START'][0],
            self.observation['DB_FILTER'][0])

        self._object_path_structure = os.path.join(
            self._target_name,
            self.observation['DB_FILTER'][0],
            self.observation['NIGHT_START'][0])
        self.filter = self.observation['DB_FILTER'][0]

        self.frames = self._sort_by_category(science_files)
        self.frames['FLAT'] = self._get_flats()
        self.frames['DISTORTION'] = self._get_distortion()

        if not observation['WAFFLE_MODE'][0]:
            self.frames['CENTER_BEFORE'], self.frames['CENTER_AFTER'] = \
                self._center_before_after_coronagraphy()
        # There are three types of background frames:
        # SKY, DARK,BACKGROUND, and DARK
        # And two sets of exposure times and filters
        # SCIENCE and FLUX
        self.background = {}
        try:
            self.background['SCIENCE'] = self._get_darks(
                exposure_time=self.frames['CENTER']['EXPTIME'][-1],
                ND_filter=self.frames['CENTER']['ND_FILTER'][-1])
            self.frames['BG_SCIENCE'] = self._pick_dark_type(
                self.background['SCIENCE'])
        except:
            self.background['SCIENCE'] = None
            dtypes = [science_files.dtype[i].__str__() for i in range(
                len(science_files.colnames))]
            self.frames['BG_SCIENCE'] = Table(names=science_files.colnames, dtype=dtypes)
            print('No background found for science frames: {}'.format(self._file_prefix))

        try:
            self.background['FLUX'] = self._get_darks(
                exposure_time=self.frames['FLUX']['EXPTIME'][-1],
                ND_filter=self.frames['FLUX']['ND_FILTER'][-1])
            self.frames['BG_FLUX'] = self._pick_dark_type(
                self.background['FLUX'])
        except:
            self.background['FLUX'] = None
            dtypes = [science_files.dtype[i].__str__() for i in range(
                len(science_files.colnames))]
            self.frames['BG_FLUX'] = Table(names=science_files.colnames, dtype=dtypes)
            print('No background found for flux frames: {}'.format(self._file_prefix))
        self.all_frames = vstack(list(self.frames.values()))
        self.data_quality_flags = self._set_data_quality_flags()

    def __repr__(self):
        return "ID: {}\nDATE: {}\nFILTER: {}\n".format(
            self.observation['MAIN_ID'][0],
            self.observation['NIGHT_START'][0],
            self.observation['DB_FILTER'][0])

    def mjd_observation(self):
        middle_index = int(len(self._science_files) // 2.)
        return Time(
            self._science_files['MJD_OBS'][middle_index],
            format='mjd',
            scale='utc')

    def _center_before_after_coronagraphy(self):
        if self.observation['WAFFLE_MODE'][0] == False:
            beginning_of_coro_seq = self.frames['CORO']['MJD_OBS'][0]
            end_of_coro_seq = self.frames['CORO']['MJD_OBS'][-1]

            center_frames = self.frames['CENTER']
            center_before = center_frames[center_frames['MJD_OBS'] < beginning_of_coro_seq]
            center_after = center_frames[center_frames['MJD_OBS'] > end_of_coro_seq]
        return center_before, center_after

    def _closest_files_with_single_date(self, table, mjd):
        try:
            idx = find_nearest(table['MJD_OBS'], mjd)
            date = table[idx]['NIGHT_START']
            table = filter_table(table, {'NIGHT_START': date})
            return table
        except ValueError:
            return None

    def _sort_by_category(self, science_files):
        frames = {}
        frames['CORO'] = filter_table(science_files, {'DPR_TYPE': 'OBJECT'})
        frames['CENTER'] = filter_table(science_files, {'DPR_TYPE': 'OBJECT,CENTER'})
        frames['FLUX'] = filter_table(science_files, {'DPR_TYPE': 'OBJECT,FLUX'})

        return frames

    def _get_flats(self):
        flats = filter_table(
            self._calibration['FLAT'],
            {'DB_FILTER': self.observation['DB_FILTER']})
        flats = self._closest_files_with_single_date(
            flats, self.mjd_observation().value)

        return flats

    def _get_darks(self, exposure_time, ND_filter):
        # Should information on observation be read from table or files here?
        # Probably table...
        background_files = self._calibration['BACKGROUND']
        background = {}

        background['SKY'] = filter_table(
            table=self._science_files, condition_dictionary={
                'DPR_TYPE': 'SKY',
                'DB_FILTER': self.observation['DB_FILTER'][0],
                'ND_FILTER': ND_filter,
                'EXPTIME': exposure_time})

        background['BACKGROUND'] = filter_table(
            background_files, condition_dictionary={
                'DPR_TYPE': 'DARK,BACKGROUND',
                'DB_FILTER': self.observation['DB_FILTER'],
                'ND_FILTER': self.frames['CENTER']['ND_FILTER'][-1],
                'EXPTIME': exposure_time})

        background['BACKGROUND'] = \
            self._closest_files_with_single_date(
                background['BACKGROUND'],
                self.mjd_observation().value)

        background['DARK'] = filter_table(
            background_files, condition_dictionary={
                'DPR_TYPE': 'DARK',
                'EXPTIME': exposure_time})
        background['DARK'] = \
            self._closest_files_with_single_date(
                background['DARK'],
                self.mjd_observation().value)

        return background

    def _pick_dark_type(self, background_dictionary):
        background = background_dictionary
        if len(background['SKY']) > 0:
            return background['SKY']
        elif len(background['BACKGROUND']) > 0:
            return background['BACKGROUND']
        elif len(background['DARK']) > 0:
            return background['DARK']
        else:
            raise FileNotFoundError('No dark found.')

    def _get_distortion(self):
        distortion_frames = filter_table(
            self._calibration['DISTORTION'],
            {'DB_FILTER': self.observation['DB_FILTER']})
        try:
            idx_nearest = find_nearest(
                distortion_frames['MJD_OBS'], self.mjd_observation().value)
            distortion_frames = Table(distortion_frames[idx_nearest])
        except ValueError:
            pass
        return distortion_frames

    def _set_data_quality_flags(self):
        data_quality_flags = OrderedDict()
        try:
            data_quality_flags['BG_SCIENCE'] = \
                [self.frames['BG_SCIENCE']['DPR_TYPE'][0]]
            data_quality_flags['BG_SCIENCE_TIMEDIFF'] = \
                [(self.mjd_observation() - Time(self.frames['BG_SCIENCE']['MJD_OBS'][0], format='mjd')).value]
            data_quality_flags['BG_SCIENCE_WARNING'] = \
                [data_quality_flags['BG_SCIENCE_TIMEDIFF'][0] > 1]
        except:
            data_quality_flags['BG_SCIENCE'] = ['NA']
            data_quality_flags['BG_SCIENCE_TIMEDIFF'] = [TimeDelta(-10000 * u.day).value]
            data_quality_flags['BG_SCIENCE_WARNING'] = [True]

        try:
            data_quality_flags['BG_FLUX'] = \
                [self.frames['BG_FLUX']['DPR_TYPE'][0]]
            data_quality_flags['BG_FLUX_TIMEDIFF'] = \
                [(self.mjd_observation() - Time(self.frames['BG_FLUX']['MJD_OBS'][0], format='mjd')).value]
            data_quality_flags['BG_FLUX_WARNING'] = \
                [data_quality_flags['BG_FLUX_TIMEDIFF'][0] > 1]
        except:
            data_quality_flags['BG_FLUX'] = ['NA']
            data_quality_flags['BG_FLUX_TIMEDIFF'] = [TimeDelta(-10000 * u.day).value]
            data_quality_flags['BG_FLUX_WARNING'] = [True]

        data_quality_flags['FLAT_TIMEDIFF'] = \
            [(self.mjd_observation() - Time(self.frames['FLAT']['MJD_OBS'][0], format='mjd')).value]
        data_quality_flags['FLAT_WARNING'] = \
            [data_quality_flags['FLAT_TIMEDIFF'][0] > 1]

        try:
            data_quality_flags['DISTORTION_TIMEDIFF'] = \
                [(self.mjd_observation() - Time(self.frames['DISTORTION']['MJD_OBS'][0], format='mjd')).value]
            data_quality_flags['DISTORTION_WARNING'] = \
                [data_quality_flags['DISTORTION_TIMEDIFF'][0] > 20]
        except:
            data_quality_flags['DISTORTION_TIMEDIFF'] = [TimeDelta(-10000 * u.day).value]
            data_quality_flags['DISTORTION_WARNING'] = [True]

        return Table(data_quality_flags)

    def _make_sof_file_paths(self, reduction_directory):
        file_prefix = self._file_prefix
        sof_file_paths = {
            'BG_SCIENCE': os.path.join(
                reduction_directory, 'sof', file_prefix + 'dark_science.sof'),
            'BG_FLUX': os.path.join(
                reduction_directory, 'sof', file_prefix + 'dark_flux.sof'),
            'FLAT': os.path.join(
                reduction_directory, 'sof', file_prefix + 'flat.sof'),
            'DISTORTION': os.path.join(
                reduction_directory, 'sof', file_prefix + 'distortion.sof'),
            'CENTER': os.path.join(
                reduction_directory, 'sof', file_prefix + 'center.sof'),
            'SCIENCE': os.path.join(
                reduction_directory, 'sof', file_prefix + 'science.sof'),
            'FLUX': os.path.join(
                reduction_directory, 'sof', file_prefix + 'flux.sof')}

        return sof_file_paths

    def check_frames(self):
        keys = ['FLAT', 'BG_SCIENCE', 'BG_FLUX', 'CENTER', 'FLUX']
        for key in keys:
            if len(self.frames[key]) < 1:
                raise FileNotFoundError("No {} file for observation {}".format(
                    key, self.__repr__()))
        if not self.observation['WAFFLE_MODE'][0]:
            if len(self.frames['CORO']) < 1:
                raise FileNotFoundError("No coronagraphic file for observation {}".format(
                    key, ))

    def write_sofs(self, reduction_directory, static_calibration):
        sof_file_paths = self._make_sof_file_paths(reduction_directory)
        obs_band = self.observation['DB_FILTER'][0]
        if self.observation['WAFFLE_MODE'][0] == True:
            science_frames = self.frames['CENTER']
        else:
            science_frames = self.frames['CORO']

        if not os.path.exists(os.path.join(reduction_directory, 'sof')):
            os.makedirs(os.path.join(reduction_directory, 'sof'))

        f = open(sof_file_paths['BG_SCIENCE'], 'w')
        for filename in self.frames['BG_SCIENCE']['FILE']:
            f.writelines([filename, ' ', 'IRD_DARK_RAW', '\n'])
        f.close()

        f = open(sof_file_paths['BG_FLUX'], 'w')
        for filename in self.frames['BG_FLUX']['FILE']:
            f.writelines([filename, ' ', 'IRD_DARK_RAW', '\n'])
        f.close()

        # Make .sof for master flat, should always be table
        f = open(sof_file_paths['FLAT'], 'w')
        for filename in self.frames['FLAT']['FILE']:
            f.writelines([filename, ' ', 'IRD_FLAT_FIELD_RAW', '\n'])
        f.writelines([os.path.join(
            reduction_directory, 'dark/static_badpixels_science.fits'), ' ', 'IRD_STATIC_BADPIXELMAP'])
        f.close()

        # Make .sof for distortion map
        f = open(sof_file_paths['DISTORTION'], 'w')
        f.writelines([self.frames['DISTORTION']['FILE'][0], ' ', 'IRD_DISTORTION_MAP_RAW', '\n'])
        # TODO: Indclude dark here at some point?
        f.writelines([os.path.join(
            reduction_directory, 'flat/irdis_flat.fits'), ' ', 'IRD_FLAT_FIELD', '\n'])
        f.writelines([static_calibration['POINT_PATTERN'], ' ', 'IRD_POINT_PATTERN', '\n'])
        f.close()

        center_frames = self.frames['CENTER']
        if self.observation['WAFFLE_MODE'][0] == False:

            center_before = self.frames['CENTER_BEFORE']
            center_after = self.frames['CENTER_AFTER']
            number_center_before = len(self.frames['CENTER_BEFORE'])
            number_center_after = len(self.frames['CENTER_AFTER'])

            # Make .sof for centering
            for i in range(number_center_before):
                f = open(sof_file_paths['CENTER'] + '_before_{}'.format(i), 'w')
                f.writelines([center_before['FILE'][i], ' ', 'IRD_STAR_CENTER_WAFFLE_RAW', '\n'])  # ?
                # for filename in center_frames['File']:
                #    f.writelines([filename, ' ', 'IRD_STAR_CENTER_WAFFLE_RAW', '\n'])
                f.writelines([static_calibration['CENTERING_MASK'], ' ', 'IRD_STATIC_BADPIXELMAP', '\n'])
                f.writelines([os.path.join(reduction_directory, 'dark/master_dark_science.fits'),
                              ' ', 'IRD_MASTER_DARK', '\n'])
                f.writelines([os.path.join(reduction_directory, 'flat/irdis_flat.fits'), ' ', 'IRD_FLAT_FIELD', '\n'])
                f.close()

            for i in range(number_center_after):
                f = open(sof_file_paths['CENTER'] + '_after_{}'.format(i), 'w')
                f.writelines([center_after['FILE'][i], ' ', 'IRD_STAR_CENTER_WAFFLE_RAW', '\n'])  # ?
                # for filename in center_frames['File']:
                #    f.writelines([filename, ' ', 'IRD_STAR_CENTER_WAFFLE_RAW', '\n'])
                f.writelines([static_calibration['CENTERING_MASK'], ' ', 'IRD_STATIC_BADPIXELMAP', '\n'])
                f.writelines([os.path.join(reduction_directory, 'dark/master_dark_science.fits'),
                              ' ', 'IRD_MASTER_DARK', '\n'])
                f.writelines([os.path.join(reduction_directory, 'flat/irdis_flat.fits'), ' ', 'IRD_FLAT_FIELD', '\n'])
                f.close()

            # Make .sof for science_dbi science
            f = open(sof_file_paths['SCIENCE'], 'w')
            for filename in science_frames['FILE']:
                f.writelines([filename, ' ', 'IRD_SCIENCE_DBI_RAW', '\n'])
            # f.writelines([reduction_directory, 'dark/static_badpixels_science.fits', ' ', 'IRD_STATIC_BADPIXELMAP', '\n'])
            f.writelines([os.path.join(reduction_directory, 'dark/master_dark_science.fits'), ' ', 'IRD_MASTER_DARK', '\n'])
            f.writelines([os.path.join(reduction_directory, 'flat/irdis_flat.fits'), ' ', 'IRD_FLAT_FIELD', '\n'])
            f.writelines([os.path.join(reduction_directory, 'center/starcenter.fits'), ' ', 'IRD_STAR_CENTER', '\n'])
            f.writelines([static_calibration['FILTER_TABLE'][obs_band], ' ', 'IRD_FILTER_TABLE', '\n'])
            f.close()

        else:  # WAFFLE MODE ON / CHARACTERIZATION SEQUENCE
            # Centers do not need to be divided in before and after
            for i in range(len(self.frames['CENTER'])):
                f = open(sof_file_paths['CENTER'] + '_{}'.format(i), 'w')
                f.writelines([center_frames['FILE'][i], ' ', 'IRD_STAR_CENTER_WAFFLE_RAW', '\n'])  # ?
                f.writelines([static_calibration['CENTERING_MASK'], ' ', 'IRD_STATIC_BADPIXELMAP', '\n'])
                f.writelines([os.path.join(reduction_directory, 'dark/master_dark_science.fits'),
                              ' ', 'IRD_MASTER_DARK', '\n'])
                f.writelines([os.path.join(reduction_directory, 'flat/irdis_flat.fits'), ' ', 'IRD_FLAT_FIELD', '\n'])
                f.close()

                # CENTER FILES ARE SCIENCE FILES
                f = open(sof_file_paths['SCIENCE'], 'w')
                for filename in center_frames['FILE']:
                    f.writelines([filename, ' ', 'IRD_SCIENCE_DBI_RAW', '\n'])
                # f.writelines([os.path.join(reduction_directory, 'dark/static_badpixels_science.fits'), ' ', 'IRD_STATIC_BADPIXELMAP', '\n'])
                f.writelines([os.path.join(reduction_directory, 'dark/master_dark_science.fits'),
                              ' ', 'IRD_MASTER_DARK', '\n'])
                f.writelines([os.path.join(reduction_directory, 'flat/irdis_flat.fits'), ' ', 'IRD_FLAT_FIELD', '\n'])
                f.writelines([os.path.join(reduction_directory, 'center/starcenter.fits'), ' ', 'IRD_STAR_CENTER', '\n'])
                f.writelines([static_calibration['FILTER_TABLE'][obs_band], ' ', 'IRD_FILTER_TABLE', '\n'])
                f.close()

        # Make .sof for sciece_dbi flux
        f = open(sof_file_paths['FLUX'], 'w')
        for filename in self.frames['FLUX']['FILE']:
            f.writelines([filename, ' ', 'IRD_SCIENCE_DBI_RAW', '\n'])
        # f.writelines([reduction_directory, 'dark/static_badpixels_flux.fits', ' ', 'IRD_STATIC_BADPIXELMAP', '\n'])
        f.writelines([os.path.join(reduction_directory, 'dark/master_dark_flux.fits'), ' ', 'IRD_MASTER_DARK', '\n'])
        f.writelines([os.path.join(reduction_directory, 'flat/irdis_flat.fits'), ' ', 'IRD_FLAT_FIELD', '\n'])
        f.writelines([os.path.join(reduction_directory, 'center/starcenter.fits'), ' ', 'IRD_STAR_CENTER', '\n'])
        f.writelines([static_calibration['FILTER_TABLE'][obs_band], ' ', 'IRD_FILTER_TABLE', '\n'])
        f.close()

    def get_reduction_info(self, reduction_directory):
        reduction_info_path = os.path.join(reduction_directory,
                                           self._object_path_structure,
                                           'reduction_info.fits')
        try:
            reduction_info = Table.read(reduction_info_path)
        except FileNotFoundError:
            reduction_info = Table()
        return reduction_info
