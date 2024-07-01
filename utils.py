import numpy as np

from astropy.timeseries import LombScargle

from stochastic.processes.noise import ColoredNoise

from jwst import datamodels

def get_mad_sigma(x):
    """
    Returns the Median Absolute Deviation (MAD)-based standard deviation.
    More info: https://en.wikipedia.org/wiki/Median_absolute_deviation.

    Parameters
    ----------
    x : np.array
        Array to calculate the MAD-based standard deviation.

    Returns
    -------
    float
        MAD-based standard deviation (MAD times 1.4826)

    """
    mad = np.nanmedian( np.abs( x - np.nanmedian( x ) ) )

    return 1.4826 * mad

def correct_darks(darks, dq = None, nsigma = 10, amplifier_location = None):
    """
    Given a dark exposure (darks) from a given amplifier, this script (a) calculates the median for each group 
    of each integration and removes that, (b) calculates the median of *all* groups in an integration 
    and removes that (i.e., the integration-level bias) and returns the corrected product, (c) removes again the median 
    to consider any small remaining offsets between the bias in b) and the initial median in a). If a data quality `dq` 
    array is given, this is used to not include outliers/bad pixels in the calculations. If it's not given, the 
    function estimates those after a first pass from (a) and (b) above, and then repeats that process.

    If `amplifier_location` is given, this process above is done separately for each amplifier.

    This function also sets to zero all outlier pixels (so users can replace them later with whatever they want).


    Parameters
    ----------
    darks : np.array
        Array containing the darks in a numpy array. Dimensions should be [integration, groups, pixel, pixel].
    dq : np.array, optional
        Array with same dimensions as `dark`, but with data-quality flags. All good pixels should be marked with a value of 1.
        Bad pixels should have any other value.
    nsigma : int
        N-sigma rejection for automatically identifying outliers. This is used if `dq` is not given.
    amplifier_locartion : list
        Location of amplifiers in case of more-than-one amplifier case (e.g., NIRCam or full-frame data in general). This should be a list, 
        locating the (pythonic) start and end column of the amplifiers (amplifiers are assumed to go along columns).

    Returns
    -------
    np.array
        Array containing the median-and-bias corrected integrations per group; zero values are outliers/bad pixels.

    """

    if amplifier_location is not None:

        corrected_darks = np.zeros(darks.shape)

        for i in range( len(amplifier_location) ):

            a_start, a_end = amplifier_location[i]

            corrected_darks[:, :, :, a_start:a_end] = correct_darks(darks[:, :, :, a_start:a_end])

        return corrected_darks

    # If user gives a DQ array, create a new one that sets all non-ones to nan:
    nanarray = np.ones(darks.shape)
    if dq is not None:

        nanarray[dq!=1] = np.nan

    corrected_darks = np.zeros(darks.shape)

    integrations, groups = darks.shape[0], darks.shape[1]

    for i in range(integrations):

        for j in range(groups):
        
            # Remove group-to-group median:
            corrected_darks[i, j, :, :] = darks[i, j, :, :] - np.nanmedian( darks[i, j, :, :] * nanarray[i, j, :, :] )
            
        # Get integration-level bias:
        bias = np.nanmedian( corrected_darks[i, :, :, :] * nanarray[i, :, :, :], axis = 0 )

        # Remove it:
        corrected_darks[i, :, :, :] -= bias

        # Replace nans and outlier/bad pixels (if dq is given) with zeroes; compute median and substract again:
        if dq is not None:

            for j in range(groups):

                # Recompute median:
                corrected_darks[i, j, :, :] = corrected_darks[i, j, :, :] - np.nanmedian( corrected_darks[i, j, :, :] * nanarray[i, j, :, :] )

                # Zero the nans/outliers:
                corrected_darks[i, j, :, :][np.isnan(corrected_darks[i, j, :, :])] = 0.
                corrected_darks[i, j, :, :][np.isnan(nanarray[i, j, :, :])] = 0.

    if dq is None:

        dq = np.ones(corrected_darks.shape)

        # Identify outliers in each group:
        for i in range(integrations):

            for j in range(groups):

                sigma = get_mad_sigma( corrected_darks[i, j, :, :].flatten() )
                idx_bad = np.where( np.abs( corrected_darks[i, j, :, :] ) > nsigma * sigma )
                idx_good = np.where( np.abs( corrected_darks[i, j, :, :] ) <= nsigma * sigma )

                dq[i, j, :, :][idx_bad] = 0
       
        return correct_darks(darks, dq = dq, nsigma = nsigma)
 
    return corrected_darks

def get_dark_psds(darks, row_start = 255, column_start = 2047, pixel_time = 10, jump_time = 120, nfrequencies = 65536):
    """
    Given a set of dark integrations and a set of starting points from which the readout starts, this script calculates the power spectral density for 
    a pre-defined grid of frequencies, for each integration. This ommits by default any zero-valued count.

    This script assumes readout happens in the columns. So first pixel to be read is (row_start, column_start), next one depends on the row_start:

    - If row_start = 0, then next pixel to be read is (row_start + 1, column_start).
    - If row_start = 255, then next pixel to be read is (row_start - 1, column_start).

    Similarly, if column_start = 0, then once the jump occurs on a given column the next column is column_start + 1. If column_start = 2047, next one is 
    column_start - 1.

    Parameters
    ----------
    darks : np.array
        Array containing the darks in a numpy array. Dimensions should be [integration, groups, rows, columns]. It is expected each group is zero-mean.

    row_start : float
        Optional value indicating the (pythonic) index of row at which readout begins.

    column_start : float
        Optional value indicating the column at which readout begins.

    pixel_time : float
        Time it takes to read a pixel along each column in microseconds. Default is 10 microseconds (i.e., like JWST NIR detectors).

    jump_time : float
        Time it takes to jump from one column to the next once all its pixels have been read, in microseconds. Default is 120 microseconds (i.e., like JWST NIR detectors).

    nfrequencies : int
        Number of frequencies to compute (default optimizes speed for Lomb Scargle).

    Returns
    -------
    frequencies : np.array
        Frequency array at which the PSDs are estimated.

    psds : np.array
        Array of length [integration, group, len(frequency)] containing the PSDs of each integration and group.

    median_psd : np.array
        Array of length len(frequency) having the median of all the PSDs from every integration and group.

    """

    nintegrations, ngroups, nrows, ncolumns = darks.shape

    # First, generate time-stamps --- these start from zero:
    times, _ = generate_detector_ts(1., 1., 1., columns = ncolumns, rows = nrows, return_time = True) 

    # Get frequencies --- these are the ones used in Everett's paper, which we'll use here too. These are in Hz, and are set in linspace as that 
    # optimizes speed of Lomb Scargle periodogram calculations:
    frequencies = np.linspace(1./5., 1./2e-5, nfrequencies) 

    # Set array that will save all PSDs:
    psds = np.zeros([nintegrations, ngroups, nfrequencies])

    # Now, read the pixels, one by one for each integration and group. Get PSDs, store them:
    for i in range(nintegrations):

        for j in range(ngroups):

            # Extract counts for this integration/group following counting scheme above:
            counts = np.zeros( len(times) )

            if row_start == 0:

                d_row = 1
                first_row = 0
                last_row = nrows
    
            else:

                d_row = -1
                first_row = nrows - 1
                last_row = -1

            if column_start == 0:

                d_column = 1
                first_column = 0
                last_column = ncolumns

            else:

                d_column = -1
                first_column = ncolumns - 1
                last_column = -1

            counter = 0
            for l in range(first_column, last_column, d_column):
            
                for k in range(first_row, last_row, d_row):

                    counts[counter] = darks[i, j, k, l]
                    counter += 1

            idx = np.where(counts != 0)[0]

            psds[i, j, :] = LombScargle(times[idx] * 1e-6, counts[idx], normalization = 'psd').power(frequencies)

    return frequencies, psds, np.median(psds, axis = (0,1))

def generate_detector_ramps(beta, sigma_w, sigma_flicker, rates, nintegrations = 1, ngroups = 10, columns = 2048, rows = 256, frametime = 5.494, gain = 1.61, return_all = True):
    """
    This function generates `nintegrations` integrations of `ngroups` groups each given a seed "image" of `rates`, with 1/f and white-noise characteristics 
    given by `beta`, `sigma_w` and `sigma_flicker`.

    Parameters
    ----------
    beta : float
        Power-law index of the PSD of the noise. 
    sigma_w : boolean
        Square-root of the variance of the added Normal-distributed noise process.
    sigma_flicker : float
        Variance of the power-law process in the time-domain. 
    rates : np.array
        Array containing the seed rates that define the up-the-ramp samples in counts per second.
    nintegrations : int
        Number of integrations to simulate
    ngroups : int
        Number of groups of the up-the-ramp samples
    columns : int
        Number of columns of the detector.
    rows : int
        Number of rows of the detector.
    frametime : float
        The frame time (group time really in this function) in seconds.
    gain : float
        The gain of the detector in electrons per ADU.
    return_all : boolean
        If True, returns the three components of the data: ramps + noise, ramps and noise. Memory intensive.

    Returns
    -------
    ramps : np.array
        Array of length [nintegrations, ngroups, rows, columns] containing the simulated ramps plus noise

    ramps_only : np.array
        Same as above but without noise
    
    noise : np. array
        Same as above, but only the noise

    """

    # First, generate the 1/f + white-noise pattern for the nintegrations * ngroups frames:
    ramps = np.zeros([nintegrations, ngroups, rows, columns])

    if return_all:

        noiseless_ramps = np.zeros([nintegrations, ngroups, rows, columns])
        one_over_f = np.zeros([nintegrations, ngroups, rows, columns])

    for i in range(nintegrations):

        # Generate all the ramps:
        ramps[i, :, :, :] = generate_poisson_ramp(rates, ngroups, frametime = frametime, gain = gain)

        if return_all:

            noiseless_ramps[i, :, :, :] = np.copy(ramps[i, :, :, :])

        # Add 1/f noise to each group:
        for j in range(ngroups):

            _, onef = generate_detector_ts(beta, sigma_w, sigma_flicker, columns = columns, rows = rows, return_image = True)

            if return_all:

                one_over_f[i, j, :, :] = np.copy( onef )

            ramps[i, j, :, :] += onef

    # Return the ramps:
    if not return_all:

        return ramps

    else:

        return ramps, noiseless_ramps, one_over_f

def generate_poisson_ramp(slope, ngroups, frametime = 1., bkg = 0., gain = 1.):
    """ 
    Ramp generator function | Author: Nestor Espinoza (nespinoza@stsci.edu)
    -----------------------------------------------------------------------
    
    The main idea behind this ramp-generator function is that the number of counts at each up-the-ramp 
    sample is not impacted directly by read-noise --- *reading* each up-the-ramp sample is what generates 
    additional (white-gaussian in this case) noise. In strict terms, this is the same data-generating 
    process as that of describing a cummulative Poisson Process with measurement errors at each time i:
    
    X(i) = T(i) + WN,
    
    with
    
    T(i) = T(i-1) + P(i),
    
    and where
    
    P(i) ~ Poisson(rate)
    WN ~ Normal(0, sigma^2)
    T(0) = 0
    
    This particular ramp-generator sets WN to zero (it is assumed this will be added in a post-processing step). 
    Note also this ramp generator returns values in counts. However, all calculations of the Poisson process happen 
    in electrons.
    
    Inputs
    ------
    
    :param slope: np.array
        Slope array of the ramp in ADU/s.
        
    :param ngroups: (int)
        Number of groups in the ramp.
            
    :param frametime: (float)
        Frame time in seconds.
        
    :param bkg: (optional, float)
        Value of the *detector* background, if one wants T(0) distinct from zero. Set to zero by default.
    
    :param gain: (optinal, float)
        Detector gain in electrons/ADU.
    
    """

    ngroups = ngroups + 1

    # Start the arrays that will hold the true number of counts (T) and the actual measured ramp (X):
    T = np.zeros([ngroups, slope.shape[0], slope.shape[1]])
    X = np.zeros([ngroups, slope.shape[0], slope.shape[1]])
    X[0, :, :] = bkg
    T[0, :, :] = bkg

    # Now iterate through the process:
    for i in range(1, ngroups):

        P = np.random.poisson(slope * frametime * gain)

        # Kick T and X:
        T[i, :, :] = T[i-1, :, :] + P

        X[i, :, :] = T[i, :, :]

    return X[1:, :, :] / gain

def generate_detector_ts(beta, sigma_w, sigma_flicker, columns = 2048, rows = 512, pixel_time = 10, jump_time = 120, return_image = False, return_time = False):
    """
    This function simulates a JWST detector image and corresponding time-series of the pixel-reads, assuming the noise follows a $1/f^\beta$ power-law in its 
    power spectrum. This assumes the 1/f pattern (and hence the detector reads) go along the columns of the detector.

    Parameters
    ----------
    beta : float
        Power-law index of the PSD of the noise. 
    sigma_w : boolean
        Square-root of the variance of the added Normal-distributed noise process.
    sigma_flicker : float
        Variance of the power-law process in the time-domain. 
   columns : int
        Number of columns of the detector.
    rows : int
        Number of rows of the detector.
    pixel_time : float
        Time it takes to read a pixel along each column in microseconds. Default is 10 microseconds (i.e., like JWST NIR detectors).
    jump_time : float
        Time it takes to jump from one column to the next once all its pixels have been read, in microseconds. Default is 120 microseconds (i.e., like JWST NIR detectors).
    return_image : boolean
        If True, returns an image with the simulated values. Default is False.
    return_time : boolean 
        If True, returns times as well. Default is False.

    Returns
    -------
    times : `numpy.array`
        The time-stamp of the flux values (i.e., at what time since read-out started were they read).
    time_series : `numpy.array`
        The actual flux values on each time-stamp (i.e., the pixel counts as they were read in time).
    image : `numpy.array` 
        The image corresponding to the `times` and `time_series`, if `return_image` is set to True.
    """

    # This is the number of "fake pixels" not read during the waiting time between jumps:
    nfake = int(jump_time/pixel_time)

    # First, generate a time series assuming uniform sampling (we will chop it later to accomodate the jump_time):
    CN = ColoredNoise(beta = beta, t = (rows * columns * pixel_time) + columns * jump_time)

    # Get the samples and time-indexes:
    nsamples = rows * columns + (nfake * columns)
    y = CN.sample(nsamples)
    t = CN.times(nsamples)

    # Now remove samples not actually read by the detector due to the wait times. Commented 
    # loop below took 10 secs (!). New pythonic way is the same thing, takes millisecs, and 
    # gets image for free:

    if return_time:
        t_image = t[:-1].reshape((columns, rows + nfake))
        time_image = t_image[:, :rows]
        times = time_image.flatten()

    y_image = y[:-1].reshape((columns, rows + nfake))
    image = y_image[:, :rows]
    time_series = image.flatten()

    # Set process standard-deviation to input sigma:
    time_series = sigma_flicker * (time_series / np.sqrt(np.var(time_series)) )

    # Add poisson noise:
    time_series = time_series + np.random.normal(0., sigma_w, len(time_series))

    # Reshape image:
    image = time_series.reshape((columns, rows))

    if not return_image:
        if not return_time:
            return time_series
        else:
            return times, time_series

    else:
        if return_time:
            # Return all:
            return times, time_series, image.transpose()
        else:
            return time_series, image.transpose()
