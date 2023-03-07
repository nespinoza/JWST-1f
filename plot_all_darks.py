import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from jwst import datamodels

import utils

# Set up the datasets and folders:
datasets = {}

datasets['nircam'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nircam/01062/'

datasets['niriss-full'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/niriss/commissioning-FULL-1081/'

datasets['niriss-96'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/niriss/commissioning-SUBSTRIP96-1081/'
datasets['niriss-256'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/niriss/commissioning-SUBSTRIP256-1081/'

datasets['nirspec-sub512'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB512-1130/'
datasets['nirspec-sub512s'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB512s-1130/'
datasets['nirspec-sub2048-G395H'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G395H/'
datasets['nirspec-sub2048-G395M'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G395M/'
datasets['nirspec-sub2048-G235H'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G235H/'
datasets['nirspec-sub2048-G235M'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G235M/'
datasets['nirspec-sub2048-G140M'] = '/ifs/jwst/wit/witserv/data18/nespinoza/one-over-f/data/nirspec/commissioning-SUB2048-1130/G140M/'

amplifier_location = [[0,512],[512,1024],[1024,1536],[1536,2048]]

for k in list( datasets.keys() ):

    if not os.path.exists(k):

        os.mkdir(k)

    files = glob.glob(datasets[k]+'*.fits')

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

        for i in range(corrected.shape[0]):

            if not os.path.exists(fout+'/integration'+str(i+1)):

                os.mkdir(fout+'/integration'+str(i+1))

            for j in range(corrected.shape[1]):

                if 'full' in k:

                    plt.figure(figsize=(10,8))

                else:

                    plt.figure(figsize=(10,3))

                im = plt.imshow(corrected[i, j, :, :], aspect = 'auto') 
                im.set_clim(-10,10)  

                if j < 9:

                    number = '00'+str(j+1)

                elif j < 99:

                    number = '0'+str(j+1)

                else:

                    number = str(j+1)

                plt.savefig(fout+'/integration'+str(i+1)+'/group'+number+'.pdf')         
                plt.close()
                plt.clf()

    print(k, datasets[k])
