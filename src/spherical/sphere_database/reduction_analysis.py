#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'J. Kemmer @ MPIA (Heidelberg, Germany)'

import gc
import glob
import itertools
import os

import corner
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from astropy.table import Table, unique, vstack
from astropy.visualization import (
    AsymmetricPercentileInterval,
    ImageNormalize,
    LinearStretch,
    ManualInterval,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.widgets import Button
from tqdm import tqdm

from spherical.sphere_database.database_utils import collect_detected_sources, remove_spaces

sns.set_context('paper')
sns.set(style="ticks", font_scale=1.4)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

def plot_images_side_by_side(left_images, right_images, names,
                             left_unit='Counts',
                             right_unit='SNR',
                             left_interval=AsymmetricPercentileInterval(2, 99.7),
                             right_interval=ManualInterval(-3, 8),
                             savename='point_source_targets'):
    """Function for plotting two images side by side for comparison."""
    stretch = LinearStretch()
    fig, ax = plt.subplots()
    plt.subplot(121)
    norm = ImageNormalize(left_images[0], interval=left_interval,
                          stretch=stretch)
    image = plt.imshow(left_images[0],
                       cmap='viridis',
                       interpolation='nearest',
                       origin='lower',
                       norm=norm)
    plt.axis('off')
    plt.colorbar(image, label=left_unit, orientation='horizontal')
    text = plt.text(-50, 250, names[0][0]+'\n'+names[0][1]+'\n'+names[0][2],
                    color='black')
    plt.subplot(122)
    norm = ImageNormalize(right_images[0], interval=right_interval,
                          stretch=stretch)
    snr = plt.imshow(right_images[0],
                     cmap='viridis',
                     interpolation='nearest',
                     origin='lower',
                     norm=norm)
    plt.axis('off')
    plt.colorbar(image, label=right_unit, orientation='horizontal')

    class Index(object):
        ind = 0
        selected_objects = Table(names=['MAIN_ID',
                                        'DB_FILTER',
                                        'DATE_SHORT'],
                                 dtype=('S24', 'S10', 'S10'))

        def next(self, event):
            self.ind += 1
            if self.ind >= len(left_images):
                print('End of list, starting from beginning')
                self.ind = 0
            i = self.ind
            norm = ImageNormalize(left_images[i],
                                  interval=left_interval,
                                  stretch=stretch)
            image.set_array(left_images[i])
            image.set_norm(norm)
            text.set_text(names[i][0]+'\n'+names[i][1]+'\n'+names[i][2])
            norm = ImageNormalize(right_images[i],
                                  interval=right_interval,
                                  stretch=stretch)
            snr.set_array(right_images[i])
            snr.set_norm(norm)
            fig.canvas.draw()

        def prev(self, event):
            self.ind -= 1
            if self.ind < 0:
                print('Jumping to last element in list')
                self.ind = len(left_images)-1
            i = self.ind
            norm = ImageNormalize(left_images[i],
                                  interval=left_interval,
                                  stretch=stretch)
            image.set_array(left_images[i])
            image.set_norm(norm)
            text.set_text(names[i][0]+'\n'+names[i][1]+'\n'+names[i][2])
            norm = ImageNormalize(right_images[i],
                                  interval=right_interval,
                                  stretch=stretch)
            snr.set_array(right_images[i])
            snr.set_norm(norm)
            fig.canvas.draw()

        def obvious_disk(self, event):
            i = self.ind
            self.selected_objects.add_row((names[i][0],
                                           names[i][1],
                                           names[i][2]))
            self.selected_objects.write('{}.fits'.format(savename),
                                        overwrite=True)
            print('Observation {} added to list'.format(names[i][0]))
            self.ind += 1
            if self.ind >= len(left_images):
                print('End of list, starting from beginning')
                self.ind = 0
            i = self.ind
            norm = ImageNormalize(left_images[i],
                                  interval=left_interval,
                                  stretch=stretch)
            image.set_array(left_images[i])
            image.set_norm(norm)
            text.set_text(names[i][0]+'\n'+names[i][1]+'\n'+names[i][2])
            norm = ImageNormalize(right_images[i],
                                  interval=right_interval,
                                  stretch=stretch)
            snr.set_array(right_images[i])
            snr.set_norm(norm)
            fig.canvas.draw()


    callback = Index()
    axprev = plt.axes([0.35, 0.02, 0.1, 0.075])
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    axprev._button = bprev

    axnext = plt.axes([0.46, 0.02, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    axnext._button = bnext

    axdisk_obvious = plt.axes([0.6, 0.02, 0.15, 0.075])
    bdisk_obvious = Button(axdisk_obvious, 'Add to \n list')
    bdisk_obvious.on_clicked(callback.obvious_disk)
    axdisk_obvious._button = bdisk_obvious
    plt.show('fig')


def get_flux_and_snr_paths(package_name, observation_path):
    """Function to get the filepath of the flux and snr images for
       a given package.
    """
    results_path = glob.glob(os.path.join(observation_path,
                                          'results_{}*/'.format(package_name.lower())))[0]

    if package_name.lower() == 'andromeda':
        image_path = glob.glob(os.path.join(results_path,
                                            'flux_*IRDIS_*_channel1.fits'))[0]
        snr_path = glob.glob(os.path.join(results_path,
                                          'snr_*IRDIS_*_channel1.fits'))[0]
    elif package_name.lower() == 'pyklip'or package_name == 'vip':
        image_path = glob.glob(os.path.join(results_path,
                                            'flux_IRDIS*.fits'))[0]
        snr_path = glob.glob(os.path.join(results_path,
                                          'snr_IRDIS*.fits'))[0]
    else:
        raise Exception('Unknown package name. Valid packages are:'
                        ' \'andromeda\' or \'pyklip\'')
    return image_path, snr_path

def get_snr_paths(package_name, observation_path):
    """Function to get the filepath snr images from ANDROMEDA and PyKLIP.
    """
    results_path = glob.glob(os.path.join(observation_path,
                                          'results_{}*/'.format(package_name)))[0]
    if package_name == 'andromeda':
        snr_path = glob.glob(os.path.join(results_path,
                                          'snr_norm_*IRDIS_*_channel1.fits'))[0]
    elif package_name == 'pyklip'or package_name == 'vip':
        snr_path = glob.glob(os.path.join(results_path,
                                          'snr_IRDIS*.fits'))[0]
    else:
        raise Exception('Unknown package name. Valid packages are:'
                        ' \'andromeda\', \'pyklip\' or  \'vip\'.')
    return snr_path


def read_in_flux_and_snr_images(observation_list, reduction_folder,
                                package_name):
    """Returns lists of all available flux and snr images for a given list
       of observations and a package. Additionally the MAIN_ID, DB_FILTER and
       DATE_SHORT are provided.
    """
    description_labels = []
    images = []
    snrs = []
    for observation in tqdm(observation_list):
        try:
            main_id = remove_spaces(observation['MAIN_ID'])
            db_filter = observation['DB_FILTER']
            date_short = observation['DATE_SHORT']
            observation_path = os.path.join(reduction_folder,
                                            main_id,
                                            db_filter,
                                            date_short,
                                            'converted_mastercubes/')
            image_path, snr_path = get_flux_and_snr_paths(package_name,
                                                          observation_path)
            image_hdu = fits.open(image_path)
            image = image_hdu[0].data
            image_hdu.close()
            if len(image.shape) == 3:
                image = image[0]
            images.append(image)
            snr_hdu = fits.open(snr_path)
            snr = snr_hdu[0].data
            snr_hdu.close()
            if len(snr.shape) == 3:
                snr = snr[0]
            snrs.append(snr)
            labels = [main_id, db_filter, date_short]
            description_labels.append(labels)
            del image_hdu
            del image
            del snr_hdu
            del snr
        except Exception as e:
            print('Failed reading images of {}  {}'
                  '  {} - {}'.format(main_id, db_filter, date_short, e))
            pass
    gc.collect()
    return images, snrs, description_labels


def read_in_andromeda_and_pyklip_snr_images(observation_list,
                                            reduction_folder):
    """Returns lists of all available snr images obtained with ANDROMEDA and
       PyKLIP for a given list of observations. Additionally the MAIN_ID,
       DB_FILTER and DATE_SHORT are provided.
    """
    description_labels = []
    snrs_py = []
    snrs_an = []
    for observation in tqdm(observation_list):
        try:
            main_id = remove_spaces(observation['MAIN_ID'])
            db_filter = observation['DB_FILTER']
            date_short = observation['DATE_SHORT']
            observation_paths = glob.glob(os.path.join(reduction_folder,
                                                       main_id,
                                                       db_filter,
                                                       date_short,
                                                       'converted_mastercubes/'))
            for path in observation_paths:
                snr_path_py = get_snr_paths('pyklip', path)
                snr_path_an = get_snr_paths('andromeda', path)
                image_hdu_py = fits.open(snr_path_py)
                image_py = image_hdu_py[0].data
                image_hdu_py.close()
                if len(image_py.shape) == 3:
                    image_py = image_py[0]
                snrs_py.append(image_py)
                image_hdu_an = fits.open(snr_path_an)
                image_an = image_hdu_an[0].data
                image_hdu_an.close()
                if len(image_an.shape) == 3:
                    image_an = image_an[0]
                snrs_an.append(image_an)
                labels = snr_path_py.split('/')[7:10]
                description_labels.append([labels[0], labels[1], labels[2]])
                del image_hdu_py
                del image_py
                del image_hdu_an
                del image_an
        except Exception as e:
            print(e)
            pass
    gc.collect()
    return snrs_py, snrs_an, np.array(description_labels)


# Plotting Scripts:

def plot_contrast_vs_separation(observation_list, reduction_folder,
                                package_names, save_plot=False, save_path='',
                                save_suffix=None):
    """ Plots the available measured contrasts vs. the sources separation for
        a given list of observations and pair of package names.
    """

    filter_list = ['H2', 'H3', 'K1', 'K2']
    colors = ['#ffa62b', '#464196']
    markers = ['o', 's']

    fig = plt.figure(figsize=(6.2, 4))
    plt.xlabel('Separation [mas]')
    plt.ylabel('Contrast')
    plt.yscale('log')
    plt.ylim(1e-7, 1e-2)
    for package_name, col, marker in zip(package_names, colors, markers):
        sources_in_plot = 0
        if package_name.lower() == 'andromeda':
            sigma = 3
        else:
            sigma = 1
        detected_sources = collect_detected_sources(observation_list,
                                                    reduction_folder,
                                                    package_name.lower(),
                                                    show=False)
        detected_sources = unique(detected_sources, keys='UNIQUE_ID')
        print(len(detected_sources))
        for observation in detected_sources:
            for filt in filter_list:
                if observation['contrast_{}-[1]'.format(filt)] \
                 and observation['sep-[mas]'] < 1544:
                    if package_name.lower() == 'andromeda':
                        if observation['flag_lobe_{}'.format(filt)] != 0 \
                         or observation['flag_pos_{}'.format(filt)] != 0 \
                         or observation['flag_flux_{}'.format(filt)] != 0:
                            continue
                    sources_in_plot += 1
                    plt.scatter(observation['sep-[mas]'],
                                observation['contrast_{}-[1]'.format(filt)],
                                c=col, marker=marker)
        print('The plot contains {} {} measurements'.format(sources_in_plot,
                                                            package_name.lower()))
    legend = [Line2D([0], [0], marker='o', color='w',
                     markerfacecolor='#ffa62b',
                     markersize=15, label=package_names[0]),
              Line2D([0], [0], marker='s', color='w',
                     markerfacecolor='#464196',
                     markersize=15, label=package_names[1])]
    plt.legend(handles=legend, loc='best')
    if save_plot:
        plt.savefig(os.path.join(save_path,
                                 'contrast_vs_separation_'
                                 '{}.pdf'.format(save_suffix)),
                    bbox_inches='tight', overwrite=True)
    plt.show()


def plot_flux_vs_flux(observation_list, reduction_folder, package_names,
                      save_plot=False, save_path='', save_suffix=None):
    """ Plots the available measured contrasts for two given packages
        against each other for a given list of observations.
    """
    detected_sources_a = collect_detected_sources(observation_list,
                                                  reduction_folder,
                                                  package_names[0].lower(),
                                                  show=False)
    detected_sources_b = collect_detected_sources(observation_list,
                                                  reduction_folder,
                                                  package_names[1].lower(),
                                                  show=False)
    grouped_a = detected_sources_a.group_by('MAIN_ID')
    grouped_b = detected_sources_b.group_by('MAIN_ID')
    fig = plt.figure(figsize=(12.4, 8))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 3, 3],
                           height_ratios=[3, 3, 1])
    res_a_ax = plt.subplot(gs[2, 1:])
    res_b_ax = plt.subplot(gs[:2, 0])
    contrast_ax = plt.subplot(gs[:2, 1:], sharex=res_a_ax, sharey=res_b_ax)
    contrast_ax.tick_params(axis='both',
                            which='both',
                            labelbottom='off',
                            labelright='off',
                            labelleft='off')
    res_a_ax.set_xlabel('Contrast measured by {}'.format(package_names[0]))
    res_b_ax.set_ylabel('Contrast measured by {}'.format(package_names[1]))
    res_b_ax.set_xlabel('Difference from \n equivalence [%]',
                        rotation=45, position=(0, 0))
    for tick in res_b_ax.get_xticklabels():
        tick.set_rotation(90)
    res_a_ax.set_xlim(1e-7, 1e-3)
    res_b_ax.set_ylim(1e-7, 1e-3)
    res_a_ax.set_xscale('log')
    res_b_ax.set_yscale('log')

    lin = np.logspace(-7, -3)
    contrast_ax.plot(lin, lin, 'k--')
    res_a_ax.axhline(0, c='k', ls='--')
    res_b_ax.axvline(0, c='k', ls='--')
    sources_in_plot = 0

    if package_names[0].lower() == 'andromeda':
        sigma_a = 3
    else:
        sigma_a = 1
    if package_names[1].lower() == 'andromeda':
        sigma_b = 3
    else:
        sigma_b = 1
    for key in grouped_a.groups.keys:
        source_keys_a = grouped_a[grouped_a['MAIN_ID'] == key['MAIN_ID']]
        source_keys_b = grouped_b[grouped_b['MAIN_ID'] == key['MAIN_ID']]
        for source_a, source_b in itertools.product(source_keys_a,
                                                    source_keys_b):
            if np.allclose(np.array([source_a['offset_x-[px]'],
                                     source_a['offset_y-[px]']]),
                           np.array([source_b['offset_x-[px]'],
                                     source_b['offset_y-[px]']]),
                           atol=3):
                filter_list = ['H2', 'H3', 'K1', 'K2']
                colors = ['#4984b8', '#ffa62b', '#464196', '#990147']
                for filt, col in zip(filter_list, colors):
                    if source_a['contrast_{}-[1]'.format(filt)] \
                     and source_b['contrast_{}-[1]'.format(filt)]\
                     and source_a['err_contrast_{}-[1]'.format(filt)]/source_a['contrast_{}-[1]'.format(filt)] < 0.5 \
                     and source_b['err_contrast_{}-[1]'.format(filt)]/source_b['contrast_{}-[1]'.format(filt)] < 0.5 \
                     and source_a['contrast_{}-[1]'.format(filt)] < 1e-3 \
                     and source_b['contrast_{}-[1]'.format(filt)] < 1e-3:
                        if package_names[0].lower() == 'andromeda':
                            if source_a['flag_lobe_{}'.format(filt)] != 0 \
                             or source_a['flag_pos_{}'.format(filt)] != 0 \
                             or source_a['flag_flux_{}'.format(filt)] != 0:
                                continue
                        if package_names[1].lower() == 'andromeda':
                            if source_b['flag_lobe_{}'.format(filt)] != 0 \
                             or source_b['flag_pos_{}'.format(filt)] != 0 \
                             or source_b['flag_flux_{}'.format(filt)] != 0:
                                continue

                        sources_in_plot += 1
                        contrast_ax.errorbar(source_a['contrast_{}-[1]'.format(filt)],
                                             source_b['contrast_{}-[1]'.format(filt)],
                                             xerr=source_a['err_contrast_{}-[1]'.format(filt)]/sigma_a,
                                             yerr=source_b['err_contrast_{}-[1]'.format(filt)]/sigma_b,
                                             c=col,
                                             markersize=30)
                        res_a_ax.scatter(source_a['contrast_{}-[1]'.format(filt)],
                                         100*(source_a['contrast_{}-[1]'.format(filt)]-source_b['contrast_{}-[1]'.format(filt)])/source_a['contrast_{}-[1]'.format(filt)],
                                         c=col, s=40)
                        res_b_ax.scatter(100*(source_b['contrast_{}-[1]'.format(filt)]-source_a['contrast_{}-[1]'.format(filt)])/source_b['contrast_{}-[1]'.format(filt)],
                                         source_b['contrast_{}-[1]'.format(filt)],
                                         c=col, s=40)
    legend = [Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#4984b8', markeredgewidth=2,
                     markersize=15, label='H2'),
              Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#ffa62b', markeredgewidth=2,
                     markersize=15, label='H3'),
              Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#464196', markeredgewidth=2,
                     markersize=15, label='K1'),
              Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#990147', markeredgewidth=2,
                     markersize=15, label='K2')]
    contrast_ax.legend(handles=legend, loc='best')
    start, end = res_b_ax.get_xlim()
    res_a_ax.yaxis.set_ticks(np.arange(start, end, 50))
    print('The plot contains {} measurements'.format(sources_in_plot))
    if save_plot:
        plt.savefig(os.path.join(save_path,
                                 'contrast_vs_contrast_'
                                 '{}-{}_{}.pdf'.format(package_names[0],
                                                       package_names[1],
                                                       save_suffix)),
                    bbox_inches='tight', overwrite=True)
    plt.show()


def plot_snr_vs_snr(observation_list, reduction_folder, package_names,
                    save_plot=False, save_path='', save_suffix=None):
    """ Plots the available measured snr for two given packages
        against each other for a given list of observations.
    """
    detected_sources_a = collect_detected_sources(observation_list,
                                                  reduction_folder,
                                                  package_names[0].lower(),
                                                  show=False)
    detected_sources_b = collect_detected_sources(observation_list,
                                                  reduction_folder,
                                                  package_names[1].lower(),
                                                  show=False)
    grouped_a = detected_sources_a.group_by('MAIN_ID')
    grouped_b = detected_sources_b.group_by('MAIN_ID')
    fig = plt.figure(figsize=(12.4, 8))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 3, 3],
                           height_ratios=[3, 3, 1])
    res_a_ax = plt.subplot(gs[2, 1:])
    res_b_ax = plt.subplot(gs[:2, 0])
    snr_ax = plt.subplot(gs[:2, 1:], sharex=res_a_ax, sharey=res_b_ax)
    snr_ax.tick_params(axis='both',
                            which='both',
                            labelbottom='off',
                            labelright='off',
                            labelleft='off')
    res_a_ax.set_xlabel('SNR measured by {}'.format(package_names[0]))
    res_b_ax.set_ylabel('SNR measured by {}'.format(package_names[1]))
    res_b_ax.set_xlabel('Difference from \n equivalence [%]',
                        rotation=45, position=(0, 0))
    for tick in res_b_ax.get_xticklabels():
        tick.set_rotation(90)
    res_a_ax.set_xlim(5, 20)
    res_b_ax.set_ylim(5, 20)
    # res_a_ax.set_xscale('log')
    # res_b_ax.set_yscale('log')

    lin = np.linspace(5, 30)
    snr_ax.plot(lin, lin, 'k--')
    res_a_ax.axhline(0, c='k', ls='--')
    res_b_ax.axvline(0, c='k', ls='--')
    sources_in_plot = 0

    if package_names[0].lower() == 'andromeda':
        sigma_a = 3
    else:
        sigma_a = 1
    if package_names[1].lower() == 'andromeda':
        sigma_b = 3
    else:
        sigma_b = 1
    for key in grouped_a.groups.keys:
        source_keys_a = grouped_a[grouped_a['MAIN_ID'] == key['MAIN_ID']]
        source_keys_b = grouped_b[grouped_b['MAIN_ID'] == key['MAIN_ID']]
        for source_a, source_b in itertools.product(source_keys_a,
                                                    source_keys_b):
            if np.allclose(np.array([source_a['offset_x-[px]'],
                                     source_a['offset_y-[px]']]),
                           np.array([source_b['offset_x-[px]'],
                                     source_b['offset_y-[px]']]),
                           atol=3):
                filter_list = ['H2', 'H3', 'K1', 'K2']
                colors = ['#4984b8', '#ffa62b', '#464196', '#990147']
                for filt, col in zip(filter_list, colors):
                    if source_a['contrast_{}-[1]'.format(filt)] \
                     and source_b['contrast_{}-[1]'.format(filt)]\
                     and source_a['err_contrast_{}-[1]'.format(filt)]/source_a['contrast_{}-[1]'.format(filt)] < 0.5 \
                     and source_b['err_contrast_{}-[1]'.format(filt)]/source_b['contrast_{}-[1]'.format(filt)] < 0.5 \
                     and source_a['contrast_{}-[1]'.format(filt)] < 1e-3 \
                     and source_b['contrast_{}-[1]'.format(filt)] < 1e-3:
                        if package_names[0].lower() == 'andromeda':
                            if source_a['flag_lobe_{}'.format(filt)] != 0 \
                             or source_a['flag_pos_{}'.format(filt)] != 0 \
                             or source_a['flag_flux_{}'.format(filt)] != 0:
                                continue
                        if package_names[1].lower() == 'andromeda':
                            if source_b['flag_lobe_{}'.format(filt)] != 0 \
                             or source_b['flag_pos_{}'.format(filt)] != 0 \
                             or source_b['flag_flux_{}'.format(filt)] != 0:
                                continue

                        sources_in_plot += 1
                        snr_ax.scatter(source_a['SNR-[1]'],
                                       source_b['SNR-[1]'], c=col)
                        res_a_ax.scatter(source_a['SNR-[1]'],
                                         100*(source_a['SNR-[1]']-source_b['SNR-[1]'])/source_a['SNR-[1]'],
                                         c=col)
                        res_b_ax.scatter(100*(source_b['SNR-[1]']-source_a['SNR-[1]'])/source_b['SNR-[1]'],
                                         source_b['SNR-[1]'],
                                         c=col)
    legend = [Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#4984b8', markeredgewidth=2,
                     markersize=15, label='H2'),
              Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#ffa62b', markeredgewidth=2,
                     markersize=15, label='H3'),
              Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#464196', markeredgewidth=2,
                     markersize=15, label='K1'),
              Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#990147', markeredgewidth=2,
                     markersize=15, label='K2')]
    snr_ax.legend(handles=legend, loc='best')
    print('The plot contains {} measurements'.format(sources_in_plot))
    if save_plot:
        plt.savefig(os.path.join(save_path,
                                 'snr_vs_snr_'
                                 '{}-{}_{}.pdf'.format(package_names[0],
                                                       package_names[1],
                                                       save_suffix)),
                    bbox_inches='tight', overwrite=True)
    plt.show()


def plot_contrastdiff_vs_separation(observation_list, reduction_folder,
                                    package_names, save_plot=False,
                                    save_path='', save_suffix=None):
    """ Plots the available measured contrasts as a function of the separation
        from the star for a given list of observations and a pair of package
        names.
    """
    detected_sources_a = collect_detected_sources(observation_list,
                                                  reduction_folder,
                                                  package_names[0].lower(),
                                                  show=False)
    detected_sources_b = collect_detected_sources(observation_list,
                                                  reduction_folder,
                                                  package_names[1].lower(),
                                                  show=False)
    grouped_a = detected_sources_a.group_by('MAIN_ID')
    grouped_b = detected_sources_b.group_by('MAIN_ID')
    fig = plt.figure(figsize=(6.2, 4))
    plt.xlabel('Mean separation [mas]')
    plt.ylabel('Difference in measured contrast '
               r'$\frac{{C_\mathrm{{{0}}}-C_\mathrm{{{1}}}}}'
               r'{{\mathrm{{mean}}(C_\mathrm{{{0}}}, C_\mathrm{{{1}}})}}$'.format(package_names[0].upper(), package_names[1].upper()))
    plt.axhline(0, linestyle='--', color='k')
    sources_in_plot = 0

    if package_names[0].lower() == 'andromeda':
        sigma_a = 3
    else:
        sigma_a = 1
    if package_names[1].lower() == 'andromeda':
        sigma_b = 3
    else:
        sigma_b = 1
    for key in grouped_a.groups.keys:
        source_keys_a = grouped_a[grouped_a['MAIN_ID'] == key['MAIN_ID']]
        source_keys_b = grouped_b[grouped_b['MAIN_ID'] == key['MAIN_ID']]
        for source_a, source_b in itertools.product(source_keys_a,
                                                    source_keys_b):
            if np.allclose(np.array([source_a['offset_x-[px]'],
                                     source_a['offset_y-[px]']]),
                           np.array([source_b['offset_x-[px]'],
                                     source_b['offset_y-[px]']]),
                           atol=3):
                filter_list = ['H2', 'H3', 'K1', 'K2']
                colors = ['#4984b8', '#ffa62b', '#464196', '#990147']
                for filt, col in zip(filter_list, colors):
                    if source_a['contrast_{}-[1]'.format(filt)] \
                     and source_b['contrast_{}-[1]'.format(filt)]\
                     and source_a['err_contrast_{}-[1]'.format(filt)]/source_a['contrast_{}-[1]'.format(filt)] < 0.5 \
                     and source_b['err_contrast_{}-[1]'.format(filt)]/source_b['contrast_{}-[1]'.format(filt)] < 0.5 \
                     and source_a['contrast_{}-[1]'.format(filt)] < 1e-3 \
                     and source_b['contrast_{}-[1]'.format(filt)] < 1e-3:
                        if package_names[0].lower() == 'andromeda':
                            if source_a['flag_lobe_{}'.format(filt)] != 0 \
                             or source_a['flag_pos_{}'.format(filt)] != 0 \
                             or source_a['flag_flux_{}'.format(filt)] != 0:
                                continue
                        if package_names[1].lower() == 'andromeda':
                            if source_b['flag_lobe_{}'.format(filt)] != 0 \
                             or source_b['flag_pos_{}'.format(filt)] != 0 \
                             or source_b['flag_flux_{}'.format(filt)] != 0:
                                continue

                        sources_in_plot += 1
                        plt.scatter(np.mean([source_a['sep-[mas]'],
                                             source_b['sep-[mas]']]),
                                    (source_a['contrast_{}-[1]'.format(filt)] -
                                     source_b['contrast_{}-[1]'.format(filt)]) /
                                    np.mean([source_a['contrast_{}-[1]'.format(filt)],
                                            source_b['contrast_{}-[1]'.format(filt)]]),
                                    c=col)
    legend = [Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#4984b8', markeredgewidth=2,
                     markersize=15, label='H2'),
              Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#ffa62b', markeredgewidth=2,
                     markersize=15, label='H3'),
              Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#464196', markeredgewidth=2,
                     markersize=15, label='K1'),
              Line2D([0], [0], marker='+', color='w',
                     markeredgecolor='#990147', markeredgewidth=2,
                     markersize=15, label='K2')]
    plt.legend(handles=legend, loc='best')
    print('The plot contains {} measurements'.format(sources_in_plot))
    if save_plot:
        plt.savefig(os.path.join(save_path,
                                 'contrastdiff_vs_separation_'
                                 '{}-{}_{}.pdf'.format(package_names[0],
                                                       package_names[1],
                                                       save_suffix)),
                    bbox_inches='tight', overwrite=True)
    plt.show()


def plot_separation_vs_accuracy(observation_list, reduction_folder,
                                package_names,
                                save_plot=False,
                                save_path='',
                                save_suffix=None):
    """ Plots the available measured separations as a function of the relative
        uncertainties of the measurements for a given list of observations
        and pair of package names.
    """
    filter_list = ['H2', 'H3', 'K1', 'K2']
    colors = ['#ffa62b', '#464196']
    markers = ['o', 's']
    fig = plt.figure(figsize=(6.2, 4))
    plt.xlabel('Separation from Host Star [mas]')
    plt.ylabel('Relative Error '
               r'$\frac{\Delta f}{f}$')
    plt.axhline(0, linestyle='--', color='k')
    ymin, ymax = plt.gca().get_ylim()
    plt.ylim(-0.1, 1.5)
    for package_name, col, marker in zip(package_names, colors, markers):
        sources_in_plot = 0
        if package_name.lower() == 'andromeda':
            sigma = 3
        else:
            sigma = 1
        detected_sources = collect_detected_sources(observation_list,
                                                    reduction_folder,
                                                    package_name.lower(),
                                                    show=False)
        for observation in detected_sources:
            for filt in filter_list:
                if observation['contrast_{}-[1]'.format(filt)] \
                 and observation['sep-[mas]'] < 1544:
                    if package_name.lower() == 'andromeda':
                        if observation['flag_lobe_{}'.format(filt)] != 0 \
                         or observation['flag_pos_{}'.format(filt)] != 0 \
                         or observation['flag_flux_{}'.format(filt)] != 0:
                            continue
                    sources_in_plot += 1
                    plt.scatter(observation['sep-[mas]'],
                                observation['err_contrast_{}-[1]'.format(filt)] \
                                / (sigma * observation['contrast_{}-[1]'.format(filt)]),
                                c=col, marker=marker)
        print('The plot contains {} {} measurements'.format(sources_in_plot,
                                                            package_name.lower()))
    legend = [Line2D([0], [0], marker='o', color='w',
                     markerfacecolor='#ffa62b',
                     markersize=15, label=package_names[0]),
              Line2D([0], [0], marker='s', color='w',
                     markerfacecolor='#464196',
                     markersize=15, label=package_names[1])]
    plt.legend(handles=legend, loc='best')

    if save_plot:
        plt.savefig(os.path.join(save_path,
                                 'separation_vs_accuracy_'
                                 '{}.pdf'.format(save_suffix)),
                    bbox_inches='tight', overwrite=True)
    plt.show()


def plot_contrast_vs_accuracy(observation_list, reduction_folder,
                              package_names,
                              save_plot=False,
                              save_path='',
                              save_suffix=None):
    """ Plots the available measured contrasts as a function of the relative
        uncertainties of the measurements for a given list of observations
        and pair of package names.
    """
    filter_list = ['H2', 'H3', 'K1', 'K2']
    colors = ['#ffa62b', '#464196']
    markers = ['o', 's']

    fig = plt.figure(figsize=(6.2, 4))
    plt.xlabel('Contrast')
    plt.ylabel('Relative Error '
               r'$\frac{\Delta f}{f}$')
    plt.xscale('log')
    plt.xlim(1e-7, 1e-2)
    plt.ylim(-0.1, 1.2)
    plt.axhline(0, linestyle='--', color='k')
    for package_name, col, marker in zip(package_names, colors, markers):
        sources_in_plot = 0
        if package_name.lower() == 'andromeda':
            sigma = 3
        else:
            sigma = 1
        detected_sources = collect_detected_sources(observation_list,
                                                    reduction_folder,
                                                    package_name.lower(),
                                                    show=False)
        for observation in detected_sources:
            for filt in filter_list:
                if observation['contrast_{}-[1]'.format(filt)] \
                 and observation['sep-[mas]'] < 1544:
                    if package_name.lower() == 'andromeda':
                        if observation['flag_lobe_{}'.format(filt)] != 0 \
                         or observation['flag_pos_{}'.format(filt)] != 0 \
                         or observation['flag_flux_{}'.format(filt)] != 0:
                            continue
                    sources_in_plot += 1
                    plt.scatter(observation['contrast_{}-[1]'.format(filt)],
                                observation['err_contrast_{}-[1]'.format(filt)] \
                                / (sigma * observation['contrast_{}-[1]'.format(filt)]),
                                c=col, marker=marker)
        print('The plot contains {} {} measurements'.format(sources_in_plot,
                                                            package_name.lower()))
    legend = [Line2D([0], [0], marker='o', color='w',
                     markerfacecolor='#ffa62b',
                     markersize=15, label=package_names[0]),
              Line2D([0], [0], marker='s', color='w',
                     markerfacecolor='#464196',
                     markersize=15, label=package_names[1])]
    plt.legend(handles=legend, loc='best')
    if save_plot:
        plt.savefig(os.path.join(save_path,
                                 'contrast_vs_accuracy_'
                                 '{}.pdf'.format(save_suffix)),
                    bbox_inches='tight', overwrite=True)
    plt.show()

def plot_separation_hist(observation_list, observation_info, reduction_folder,
                         package_names,
                         unit='au',
                         save_plot=False,
                         save_path='',
                         save_suffix=None):
    """ Plots a histogram of the available measured separations
        for a given list of observations and pair of package names.
    """
    filter_list = ['H2', 'H3', 'K1', 'K2']
    sources_all_packages = []
    for package_name in package_names:
        if package_name.lower() == 'andromeda':
            sigma = 3
        else:
            sigma = 1
        detected_sources = collect_detected_sources(observation_list,
                                                    reduction_folder,
                                                    package_name.lower(),
                                                    show=False)
        sources_single_package = Table()
        for observation in detected_sources:
            for filt in filter_list:
                if observation['contrast_{}-[1]'.format(filt)] \
                 and observation['sep-[mas]'] < 1544:
                    if package_name.lower() == 'andromeda':
                        if observation['flag_lobe_{}'.format(filt)] != 0 \
                         or observation['flag_pos_{}'.format(filt)] != 0 \
                         or observation['flag_flux_{}'.format(filt)] != 0:
                            continue
                    if unit == 'au':
                        parallax = observation_info[observation_info['MAIN_ID'] ==
                                                    observation['MAIN_ID']]['PLX'][0]
                        if parallax == 0:
                            continue
                        observation['sep-[mas]'] = observation['sep-[mas]'] / parallax
                    sources_single_package = vstack([sources_single_package,
                                                    observation['MAIN_ID',
                                                                'UNIQUE_ID',
                                                                'sep-[mas]']])
        sources = unique(sources_single_package, keys='UNIQUE_ID')
        sources_all_packages.append(np.array(sources['sep-[mas]']))
        print('The plot contains {} {} measurements'.format(len(sources),
                                                            package_name.lower()))
    colors = ['#ffa62b', '#464196']
    if unit == 'mas':
        bins = np.linspace(0, 1500, 15)
    else:
        bins = np.linspace(0,150, 15)
    fig = plt.figure(figsize=(6.2, 4))
    plt.xlabel('Separation from Host Star [{}]'.format(unit))
    plt.ylabel('Number of Detected Sources')
    plt.hist([sources_all_packages[0],
              sources_all_packages[1]],
              normed=False, color=colors)
    legend = [Patch(color='#ffa62b',
                    label=package_names[0]),
              Patch(color='#464196',
                    label=package_names[0])]
    plt.legend(handles=legend, loc='best')

    if save_plot:
        plt.savefig(os.path.join(save_path,
                                 'separation_histogram_'
                                 '{}.pdf'.format(save_suffix)),
                    bbox_inches='tight', overwrite=True)
    plt.show()

def plot_contrast_hist(observation_list, reduction_folder,
                       package_names,
                       save_plot=False,
                       save_path='',
                       save_suffix=None):
    """ Plots a histogram of the available measured contrasts
        for a given list of observations and pair of package names.
    """
    filter_list = ['H2', 'H3', 'K1', 'K2' ]

    sources_all_packages = []
    for package_name in package_names:
        if package_name.lower() == 'andromeda':
            sigma = 3
        else:
            sigma = 1
        detected_sources = collect_detected_sources(observation_list,
                                                    reduction_folder,
                                                    package_name.lower(),
                                                    show=False)
        sources_single_package = Table()
        for observation in detected_sources:
            for filt in filter_list:
                if observation['contrast_{}-[1]'.format(filt)] \
                 and observation['sep-[mas]'] < 1544:
                    if package_name.lower() == 'andromeda':
                        if observation['flag_lobe_{}'.format(filt)] != 0 \
                         or observation['flag_pos_{}'.format(filt)] != 0 \
                         or observation['flag_flux_{}'.format(filt)] != 0:
                            continue
                    tab = Table(observation)
                    tab.rename_column('contrast_{}-[1]'.format(filt),
                                              'contrast_[1]')
                    sources_single_package = vstack([sources_single_package,
                                                    tab['MAIN_ID',
                                                        'UNIQUE_ID',
                                                        'contrast_[1]']])
        unique_sources = unique(sources_single_package, keys='UNIQUE_ID')
        sources_all_packages.append(np.array(unique_sources['contrast_[1]']))
        print('The plot contains {} {} measurements'.format(len(unique_sources),
                                                            package_name.lower()))

    colors = ['#ffa62b', '#464196']  #, '#990147']
    bins = np.linspace(1e-7, 1e-2, 15)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    fig = plt.figure(figsize=(6.2, 4))
    plt.xlabel('Contrast')
    plt.ylabel('Number of Detected Sources')
    plt.xscale('log')
    plt.xlim(1e-7, 1e-2)
    plt.hist([sources_all_packages[0],
              sources_all_packages[1]],
              bins=logbins,
              normed=False,
              color=colors)
    legend = [Patch(color='#ffa62b',
                    label=package_names[0]),
              Patch(color='#464196',
                    label=package_names[1])]
    plt.legend(handles=legend, loc='best')

    if save_plot:
        plt.savefig(os.path.join(save_path,
                                 'contrast_histogram_'
                                 '{}.pdf'.format(save_suffix)),
                    bbox_inches='tight', overwrite=True)
    plt.show()


def plot_position_diff_hist(observation_list, reduction_folder, package_names,
                            save_plot=False, save_path='', save_suffix=None):
    """ Outputs a corner plot showing the difference in the measured pixel
        position of matching detections between the two packages for a given
        list of observations.
    """
    detected_sources_a = collect_detected_sources(observation_list,
                                                  reduction_folder,
                                                  package_names[0].lower(),
                                                  show=False)
    detected_sources_b = collect_detected_sources(observation_list,
                                                  reduction_folder,
                                                  package_names[1].lower(),
                                                  show=False)
    grouped_a = detected_sources_a.group_by('MAIN_ID')
    grouped_b = detected_sources_b.group_by('MAIN_ID')
    sources_in_plot = 0

    if package_names[0].lower() == 'andromeda':
        sigma_a = 3
    else:
        sigma_a = 1
    if package_names[1].lower() == 'andromeda':
        sigma_b = 3
    else:
        sigma_b = 1
    differences = []
    for key in grouped_a.groups.keys:
        source_keys_a = grouped_a[grouped_a['MAIN_ID'] == key['MAIN_ID']]
        source_keys_b = grouped_b[grouped_b['MAIN_ID'] == key['MAIN_ID']]
        for source_a, source_b in itertools.product(source_keys_a,
                                                    source_keys_b):
            if np.allclose(np.array([source_a['offset_x-[px]'],
                                     source_a['offset_y-[px]']]),
                           np.array([source_b['offset_x-[px]'],
                                     source_b['offset_y-[px]']]),
                           atol=3):
                filter_list = ['H2', 'H3', 'K1', 'K2']
                colors = ['#4984b8', '#ffa62b', '#464196', '#990147']
                for filt, col in zip(filter_list, colors):
                    if source_a['contrast_{}-[1]'.format(filt)] \
                     and source_b['contrast_{}-[1]'.format(filt)]\
                     and source_a['err_contrast_{}-[1]'.format(filt)]/source_a['contrast_{}-[1]'.format(filt)] < 0.5 \
                     and source_b['err_contrast_{}-[1]'.format(filt)]/source_b['contrast_{}-[1]'.format(filt)] < 0.5 \
                     and source_a['contrast_{}-[1]'.format(filt)] < 1e-3 \
                     and source_b['contrast_{}-[1]'.format(filt)] < 1e-3:
                        if package_names[0].lower() == 'andromeda':
                            if source_a['flag_lobe_{}'.format(filt)] != 0 \
                             or source_a['flag_pos_{}'.format(filt)] != 0 \
                             or source_a['flag_flux_{}'.format(filt)] != 0:
                                continue
                        if package_names[1].lower() == 'andromeda':
                            if source_b['flag_lobe_{}'.format(filt)] != 0 \
                             or source_b['flag_pos_{}'.format(filt)] != 0 \
                             or source_b['flag_flux_{}'.format(filt)] != 0:
                                continue

                        sources_in_plot += 1
                        differences.append([source_a['offset_x-[px]'] -
                                            source_b['offset_x-[px]'],
                                            source_a['offset_y-[px]'] -
                                            source_b['offset_y-[px]']])
    differences = np.array(differences)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 12.4))
    corner.corner(differences, color='black',
                  hist_kwargs={'lw':0.5, 'edgecolor':'black', 'histtype':'bar',
                               'color':'#4984b8'},
                  fig=fig,
                  quantiles=[0.16, 0.5, 0.84],
                  show_titles=True,
                  labels=[r'$\delta\mathrm{{x}}_\mathrm{{{}}}-\delta\mathrm{{x}}_\mathrm{{{}}}$ [px]'.format(*package_names),
                          r'$\delta\mathrm{{y}}_\mathrm{{{}}}-\delta\mathrm{{y}}_\mathrm{{{}}}$ [px]'.format(*package_names)],
                  label_kwargs={'color':'black'})
    print('The plot contains {} measurements'.format(sources_in_plot))
    if save_plot:
        plt.savefig(os.path.join(save_path,
                                 'position_diff_hist_'
                                 '{}-{}_{}.pdf'.format(package_names[0],
                                                       package_names[1],
                                                       save_suffix)),
                    bbox_inches='tight', overwrite=True)
    plt.show()


def show_and_select_targets(observation_list, reduction_folder,
                            package_name, selection_date):
    """ Function for the selection of observations which are used to produce
        the plots.
    """
    parts = int(np.ceil(len(observation_list) / 200))
    print('###### Sample divided into {} parts'
          ' because of limited storage. ######'.format(parts))
    selected_targets = Table()
    for idx in range(parts):
        shortened_list = observation_list[idx*200:(idx+1)*200]
        flux_images = []
        snr_images = []
        labels = []
        snrs_py, snrs_an, labels = read_in_andromeda_and_pyklip_snr_images(shortened_list,
                                                                           reduction_folder)
        # flux_images, snr_images, labels = read_in_flux_and_snr_images(shortened_list,
        #                                                               reduction_folder,
        #                                                               package_name)
        print('       Please select targets which are suitable'
              ' for flux comparison \n Part {}/{}'.format(idx+1, parts))
        plot_images_side_by_side(snrs_py, snrs_an, labels,
                                          left_unit='SNR',
                                          right_unit='SNR',
                                          left_interval=ManualInterval(-3, 6),
                                          right_interval=ManualInterval(-3, 6),
                                          savename='tables/selected_point_source'
                                                   '-targets_{}'
                                                   '_{}'.format(selection_date,
                                                                idx))
        selected_targets = vstack([selected_targets,
                                   Table.read('tables/selected_point_'
                                              'source-targets_{}_{}'
                                              '.fits'.format(selection_date,
                                                             idx))])
        selected_targets = unique(selected_targets, keys=['MAIN_ID',
                                                          'DB_FILTER',
                                                          'DATE_SHORT'])
        selected_targets.write('tables/selected_point_source-targets'
                               '_{}.fits'.format(selection_date),
                               overwrite=True)
    filelist = glob.glob('tables/selected_point_source-targets_*_*.fits')
    for fileName in filelist:
        os.remove(fileName)
    return selected_targets



def plot_comparison(plot_type, database, reduction_folder,
                    package_names=None, select_targets=False,
                    selection_date=None,
                    save_plot=False,
                    save_path='',
                    unit='mas'):
    """Wrapper function to combine all different plots into one command.
        If select_targets is False, the list from
        'tables/selected_point_source-targets'
                                      '_{}.fits'.format(selection_date)
        is used.

    Parameters
    ----------
    plot_type : str
        Currently supported plot types:
        'contrast-separation'
        'contrast-contrast'
        'snr-snr'
        'contrastdiff-separation'
        'separation-accuracy'
        'contrast-accuracy'
        'separation-hist'
        'contrast-hist'
        'pos-hist'
    database : object
        A Sphere_database with information about the observations.
        Used to get the list of all observations.
    reduction_folder : str
        Path to the folder which contains the reduced data.
    package_names : list
        List of two strings with the package names which are to be compared.
        Currently supported are ANDROMEDA and PyKLIP.
    select_targets : bool
        Whether to select the observation which are used for the comparison
        or not.
    selection_date : str
        String which is added to the saved list of targets. Which is either
         selected before the comparison and then saved, or loaded from:
        'tables/selected_point_source-targets'
                                      '_{}.fits'.format(selection_date)

    save_plot :  bool
        Whether to save the plots or not.
    save_path : str
        Path to the folder where the plots are saved.
    unit : str
        Specifies the unit of the separation in the
        separation histogram. Possible options are 'mas' or 'au'.

    Returns
    -------
    nothing

    """
    observation_list = database.return_usable_only()
    for idx, name in enumerate(observation_list['MAIN_ID']):
        observation_list['MAIN_ID'][idx] = str(remove_spaces(name))
    if select_targets:
        selected_targets = show_and_select_targets(observation_list,
                                                   reduction_folder,
                                                   package_names[0],
                                                   selection_date)
    else:
        selected_targets = Table.read('tables/selected_point_source-targets'
                                      '_{}.fits'.format(selection_date))
    print('\n###### Creating {}-plot ######'.format(plot_type))
    if plot_type == 'contrast-separation':
        plot_contrast_vs_separation(selected_targets, reduction_folder,
                                    package_names, save_plot, save_path,
                                    selection_date)
    if plot_type == 'contrast-contrast':
        plot_flux_vs_flux(selected_targets, reduction_folder, package_names,
                          save_plot, save_path, selection_date)
    if plot_type == 'snr-snr':
        plot_snr_vs_snr(selected_targets, reduction_folder, package_names,
                        save_plot, save_path, selection_date)
    if plot_type == 'contrastdiff-separation':
        plot_contrastdiff_vs_separation(selected_targets, reduction_folder,
                                        package_names, save_plot, save_path,
                                        selection_date)
    if plot_type == 'separation-accuracy':
        plot_separation_vs_accuracy(selected_targets, reduction_folder,
                                    package_names, save_plot, save_path,
                                    selection_date)
    if plot_type == 'contrast-accuracy':
        plot_contrast_vs_accuracy(selected_targets, reduction_folder,
                                  package_names, save_plot, save_path,
                                  selection_date)
    if plot_type == 'separation-hist':
        plot_separation_hist(selected_targets, observation_list,
                             reduction_folder, package_names, unit, save_plot,
                             save_path, selection_date)
    if plot_type == 'contrast-hist':
        plot_contrast_hist(selected_targets, reduction_folder, package_names,
                           save_plot, save_path, selection_date)
    if plot_type == 'pos-hist':
        plot_position_diff_hist(selected_targets, reduction_folder,
                                package_names, save_plot, save_path,
                                selection_date)
