import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn

import corner

from jwst import datamodels

import abeec
from psd_utils import tso_prior, tso_distance, tso_simulator

import utils

# Set up the datasets and folders:
datasets = {}

#datasets['nircam'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nircam/01062/'

#datasets['niriss-full'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/niriss/commissioning-FULL-1081/'

datasets['niriss-96'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/niriss/commissioning-SUBSTRIP96-1081/'
datasets['niriss-256'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/niriss/commissioning-SUBSTRIP256-1081/'

#datasets['nirspec-sub512'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB512-1130/'
#datasets['nirspec-sub512s'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB512s-1130/'
#datasets['nirspec-sub2048-G395H'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G395H/'
#datasets['nirspec-sub2048-G395M'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G395M/'
#datasets['nirspec-sub2048-G235H'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G235H/'
#datasets['nirspec-sub2048-G235M'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G235M/'
#datasets['nirspec-sub2048-G140M'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G140M/'

amplifier_location = [[0,512],[512,1024],[1024,1536],[1536,2048]]


for k in list( datasets.keys() ):

    # First, compute PSDs:
    print('\t Working on ',k, '...')

    if not os.path.exists(k+'_psds.npy'):

        if not os.path.exists(k):

            os.mkdir(k)

        files = glob.glob(datasets[k]+'*.fits')

        first_time = True
        for file in files:

            uncal = datamodels.RampModel(file)
            data = uncal.data

            fname = file.split('/')[-1]

            fout = k + '/' + fname.split('.fits')[0]

            if not os.path.exists(fout):

                os.mkdir(fout)

            if 'nircam' in k:

                corrected = utils.correct_darks(data, amplifier_location = amplifier_location)

            elif 'niriss-full' in k:

                # NIRISS subarray amplifiers go in the rows. The utils.correct_darks assume amplifiers go in the columns.
                # We do a simple swap of axes to correct the darks, and then simply re-swap the result:
                swapped_array = utils.correct_darks(np.swapaxes(data, 2, 3), amplifier_location = amplifier_location)
                corrected = np.swapaxes(swapped_array, 2, 3)

            else:

                corrected = utils.correct_darks(data)

            frequencies, psds, _ = utils.get_dark_psds( corrected, row_start = 255, column_start = 2047 )

            nintegrations, ngroups = psds.shape[0], psds.shape[1]
            print('\t Filename: ', fname,': (nints, ngroups) = ',nintegrations, ngroups)

            if first_time:

                all_psds = np.zeros([nintegrations * ngroups, len(frequencies)])

                counter = 0
                for i in range(nintegrations):

                    for j in range(ngroups):

                        all_psds[counter, :] = psds[i, j, :]
                        counter += 1

                first_time = False

            else:

                new_all_psds = np.zeros([nintegrations * ngroups, len(frequencies)])

                counter = 0 
                for i in range(nintegrations):

                    for k in range(ngroups):

                        new_all_psds[counter, :] = psds[i, k, :]
                        counter += 1

                all_psds = np.vstack(( all_psds, new_all_psds ))

        # Save results of PSD computations:
        print('\t \t > Dataset has ',all_psds.shape[0], 'groups in total. Storing individual and combined PSDs...')
        np.save(k+'_frequencies', frequencies)
        np.save(k+'_psds', all_psds)

        median_psd = np.median(all_psds, axis = 0)
        np.save(k+'_median_'+str(all_psds.shape[0])+'_groups_psds', median_psd)

    else:

        print('PSDs detected! Loading data...')

        files = glob.glob(datasets[k]+'*.fits')
        corrected = datamodels.RampModel(files[0]).data 

        frequencies = np.load(k+'_frequencies.npy')
        all_psds = np.load(k+'_psds.npy')
        median_psd = np.load(k+'_median_'+str(all_psds.shape[0])+'_groups_psds.npy')

    # Now use results to perform ABC fit. Define prior, distance and simulator:
    prior = tso_prior()

    distance = tso_distance(filename = k+'_median_'+str(all_psds.shape[0])+'_groups_psds.npy', \
                            filename_indexes = 'indexes.npy')

    simulator = tso_simulator(ncolumns = corrected.shape[3], \
                              nrows = corrected.shape[2], \
                              ngroups = all_psds.shape[0], \
                              frequency_filename = k+'_frequencies.npy')

    # Generate ABC samples:
    samples = abeec.sampler.sample(prior, distance, simulator, \
                                   M = 150, N = 100, Delta = 0.1,\
                                   verbose = True, output_file = 'results_'+k+'.pkl')

    # Extract the 300 posterior samples from the latest particle:
    tend = list(samples.keys())[-1]
    betas, sigma_poissons, sigma_flickers = samples[tend]['thetas']

    # Print statistics:
    fout = open('results_'+k+'.txt','w')
    fout.write('Final parameters for '+k+'\n')
    fout.write('beta: ',np.nanmedian(betas),'+/-',np.sqrt(np.var(betas)))
    fout.write('sigma_w: ',np.nanmedian(sigma_poissons),'+/-',np.sqrt(np.var(sigma_poissons)))
    fout.write('sigma_f: ',np.nanmedian(sigma_flickers),'+/-',np.sqrt(np.var(sigma_flickers)))
    fout.close()

    # Plot corner plot:
    stacked_samples = np.vstack((np.vstack((betas, sigma_poissons)), sigma_flickers)).T
    figure = corner.corner(stacked_samples, labels = [r"$\beta$ (from 1/f$^{\beta}$)", r"$\sigma_{w}$", r"$\sigma_{flicker}$"], \
                           quantiles=[0.16, 0.5, 0.84],\
                           show_titles=True, title_kwargs={"fontsize": 12})

    plt.savefig('corner_'+detector+'.pdf') 
