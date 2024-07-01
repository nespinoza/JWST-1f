import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

import transitspectroscopy as ts

import utils

def correct_1f(median_frame, frame, x, y, min_bkg = 20, max_bkg = 35, mask = None, scale_factor = 1., return_1f = False):
    """
    This. Needs. A. zodii-bkg-corrected. Frame as input :).
    """
    
    new_frame = np.copy(frame)
    
    ms = frame - median_frame * scale_factor
    
    if return_1f:
        
        one_over_f = np.zeros(len(x))
    
    # Go column-by-column substracting values around the trace:
    for i in range(len(x)):
        
        column = int(x[i])
        row = int(y[i])
        
        min_row = np.max([0, row - 35])
        max_row = np.min([256, row + 35])
        
        bkg = np.append(ms[min_row:row-20, column], ms[row+20:max_row, column])
        new_frame[:, column] = new_frame[:, column] - np.nanmedian(bkg)
        
        if return_1f:
            
            one_over_f[i] = np.nanmedian(bkg)
        
    if return_1f:
        
        return one_over_f, new_frame
        
    else:
        
        return new_frame

def correct_refpix(data, noise):

    corrected = np.copy(data)
    for i in range(data.shape[1]):

        corrected[:, i] = data[:, i] - np.nanmedian( noise[256-4:, i] )

    return corrected
    


# These are the median counts for WASP-39b's ERS NIRISS/SOSS observation:
input_rates = np.load('WASP-39.npy')

# Turn off negative ramps:
idx = np.where(input_rates < 0)
input_rates[idx] = 0

# Generate ramps using 1/f and white-noise parameters taken from our commissioning-dark fits:
nintegrations = 537
ngroups = 9
nrows, ncolumns = input_rates.shape

if not os.path.exists('ers_ramps.npy'):

    ramps, noiseless_ramps, noise = utils.generate_detector_ramps(1.0217, 6.58, 6.11, input_rates, \
                                                                  nintegrations = nintegrations, ngroups = ngroups, \
                                                                  columns = ncolumns, rows = nrows, \
                                                                  frametime = 5.494, gain = 1.61, \
                                                                  return_all = True)

    np.save('ers_ramps', ramps)
    np.save('ers_noiseless_ramps', noiseless_ramps)
    np.save('ers_noise', noise)

else:

    ramps = np.load('ers_ramps.npy')
    noiseless_ramps = np.load('ers_noiseless_ramps.npy')
    noise = np.load('ers_noise.npy')

# Generate white-noise ramps:
white_noise_ramps = noiseless_ramps + np.random.normal(0., 6.58, noiseless_ramps.shape)

"""
Plot general idea of the methodology
"""
# Get last-minus-first as the ramp estimates:
lmf = np.zeros([nintegrations, nrows, ncolumns])
noise_lmf = np.zeros([nintegrations, nrows, ncolumns])
white_noise_lmf = np.zeros([nintegrations, nrows, ncolumns])

for i in range(nintegrations):

    lmf[i, :, :] = ramps[i, -1, :, :] - ramps[i, 0, :, :]
    noise_lmf[i, :, :] = noise[i, -1, :, :] - noise[i, 0, :, :]
    white_noise_lmf[i, :, :] = white_noise_ramps[i, -1, :, :] - white_noise_ramps[i, 0, :, :]

# Get median lmf:
median_lmf = np.nanmedian(lmf, axis = 0)

# Test algorithm on a random integration:
test_integration = 10

lmf_test = lmf[test_integration, :, :] - median_lmf

xtext, ytext = 1700, 150
plt.figure(figsize = (10, 10))

plt.subplot(411)
im = plt.imshow(lmf[test_integration, :, :])
plt.text(xtext, ytext, 'Original rate', fontsize = 15, color = 'white')
im.set_clim(0,100)

plt.subplot(412)
im = plt.imshow(median_lmf)
plt.text(xtext, ytext, 'Median rate', fontsize = 15, color = 'white')
im.set_clim(0,100)

plt.subplot(413)
im = plt.imshow(lmf_test)
plt.text(xtext, ytext, 'Difference', fontsize = 15, color = 'white')
im.set_clim(0,10)

plt.subplot(414)
im = plt.imshow(noise_lmf[test_integration, :, :])
plt.text(xtext, ytext, 'Real noise', fontsize = 15, color = 'white')
im.set_clim(0,10)

"""
Compare white-light lightcurve with (a) no removal of 1/f, (b) reference pixel removal of 1/f (c) "local" removal of 1/f (d) white-noise white-light
"""

data = np.genfromtxt('NE_traces2.csv', delimiter = ',')
x, y1, y2, y3 = data[:,0], data[:,1], data[:,2], data[:,3]

# For loop to extract spectrum with three methods above:
spectra_no1f = np.zeros([nintegrations, len(y1)])
spectra_ref1f = np.zeros([nintegrations, len(y1)])
spectra_local = np.zeros([nintegrations, len(y1)])
spectra_white_noise = np.zeros([nintegrations, len(y1)])

for i in range(nintegrations):

    corrected = correct_1f(median_lmf, \
                           lmf[i, :, :], \
                           x, y1, \
                           scale_factor = 1.)

    spectra_local[i, :] = ts.spectroscopy.getSimpleSpectrum(corrected, \
                                                            x, y1,\
                                                            15, correct_bkg = False)

    spectra_no1f[i,:] = ts.spectroscopy.getSimpleSpectrum(lmf[i, :, :], \
                                                          x, y1,\
                                                          15, correct_bkg = False)

    # Correct using "reference pixels":
    corrected_refpix = correct_refpix(lmf[i, :, :], noise_lmf[i, :, :])

    spectra_ref1f[i, :] = ts.spectroscopy.getSimpleSpectrum(corrected_refpix, \
                                                            x, y1,\
                                                            15, correct_bkg = False) 

    # Get spectra with white-noise only:
    spectra_white_noise[i, :] = ts.spectroscopy.getSimpleSpectrum(white_noise_lmf[i, :, :], \
                                                                  x, y1,\
                                                                  15, correct_bkg = False)

# Compare white-light lightcurves first:
wl_local = np.sum(spectra_local, axis = 1)
wl_local = wl_local / np.nanmedian(wl_local)

wl_no1f = np.sum(spectra_no1f, axis = 1)
wl_no1f = wl_no1f / np.nanmedian(wl_no1f)

wl_refpix = np.sum(spectra_ref1f, axis = 1)
wl_refpix = wl_refpix / np.nanmedian(wl_refpix)

wl_wn = np.sum(spectra_white_noise, axis = 1)
wl_wn = wl_wn / np.nanmedian(wl_wn)

print( '\n No 1/f removal: {0:.2f} ppm'.format(np.sqrt(np.var(wl_no1f))*1e6), \
       '\n Local removal:  {0:.2f} ppm'.format(np.sqrt(np.var(wl_local))*1e6),\
       '\n Refpix removal: {0:.2f} ppm'.format(np.sqrt(np.var(wl_refpix))*1e6),\
       '\n White data:     {0:.2f} ppm'.format(np.sqrt(np.var(wl_wn))*1e6) )

# Repeat, but on low SNR location of the spectra:
local = np.sum(spectra_local[:, 100:200], axis = 1)
local = local / np.nanmedian(local)

no1f = np.sum(spectra_no1f[:, 100:200], axis = 1)
no1f = no1f / np.nanmedian(no1f)

refpix = np.sum(spectra_ref1f[:, 100:200], axis = 1)
refpix = refpix / np.nanmedian(refpix)

wn = np.sum(spectra_white_noise[:, 100:200], axis = 1)
wn = wn / np.nanmedian(wn)

print( '\n No 1/f removal: {0:.2f} ppm'.format(np.sqrt(np.var(no1f))*1e6), \
       '\n Local removal:  {0:.2f} ppm'.format(np.sqrt(np.var(local))*1e6),\
       '\n Refpix removal: {0:.2f} ppm'.format(np.sqrt(np.var(refpix))*1e6),\
       '\n White data:     {0:.2f} ppm'.format(np.sqrt(np.var(wn))*1e6) )

# High SNR region:
local = np.sum(spectra_local[:, 1650:1750], axis = 1)
local = local / np.nanmedian(local)

no1f = np.sum(spectra_no1f[:, 1650:1750], axis = 1)
no1f = no1f / np.nanmedian(no1f)

refpix = np.sum(spectra_ref1f[:, 1650:1750], axis = 1)
refpix = refpix / np.nanmedian(refpix)

wn = np.sum(spectra_white_noise[:, 1650:1750], axis = 1)
wn = wn / np.nanmedian(wn)

print( '\n No 1/f removal: {0:.2f} ppm'.format(np.sqrt(np.var(no1f))*1e6), \
       '\n Local removal:  {0:.2f} ppm'.format(np.sqrt(np.var(local))*1e6),\
       '\n Refpix removal: {0:.2f} ppm'.format(np.sqrt(np.var(refpix))*1e6),\
       '\n White data:     {0:.2f} ppm'.format(np.sqrt(np.var(wn))*1e6) )

# Case for a few pixels in low SNR:
local = np.sum(spectra_local[:, 100:101], axis = 1)
local = local / np.nanmedian(local)

no1f = np.sum(spectra_no1f[:, 100:101], axis = 1)
no1f = no1f / np.nanmedian(no1f)

refpix = np.sum(spectra_ref1f[:, 100:101], axis = 1)
refpix = refpix / np.nanmedian(refpix)

wn = np.sum(spectra_white_noise[:, 100:101], axis = 1)
wn = wn / np.nanmedian(wn)

print( '\n No 1/f removal: {0:.2f} ppm'.format(np.sqrt(np.var(no1f))*1e6), \
       '\n Local removal:  {0:.2f} ppm'.format(np.sqrt(np.var(local))*1e6),\
       '\n Refpix removal: {0:.2f} ppm'.format(np.sqrt(np.var(refpix))*1e6),\
       '\n White data:     {0:.2f} ppm'.format(np.sqrt(np.var(wn))*1e6) )

# For loop to compute the same as a function of SNR:
counter = 0
for i in range(0, len(x), 5):

    counter += 1

all_local = np.zeros(counter)#len(x))
all_no1f = np.zeros(counter)#len(x))
all_refpix = np.zeros(counter)#len(x))
all_wn = np.zeros(counter)#len(x))

counter = 0
for i in range(0, len(x), 5):

    time_series = np.sum(spectra_local[:, i:i+5], axis = 1)
    all_local[counter] = np.sqrt(np.var( time_series / np.nanmedian(time_series) ))

    time_series = np.sum(spectra_no1f[:, i:i+5], axis = 1)
    all_no1f[counter] = np.sqrt(np.var( time_series / np.nanmedian(time_series) ))

    time_series = np.sum(spectra_ref1f[:, i:i+5], axis = 1)
    all_refpix[counter] = np.sqrt(np.var( time_series / np.nanmedian(time_series) ))

    time_series = np.sum(spectra_white_noise[:, i:i+5], axis = 1)
    all_wn[counter] = np.sqrt(np.var( time_series / np.nanmedian(time_series) ))

    counter += 1
