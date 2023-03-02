import numpy as np
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

def correct_darks(darks, dq = None, nsigma = 10):
    """
    Given a dark exposure (darks) from a given amplifier, this script (a) calculates the median for each group 
    of each integration and removes that, (b) calculates the median of *all* groups in an integration 
    and removes that (i.e., the integration-level bias) and returns the corrected product, (c) removes again the median 
    to consider any small remaining offsets between the bias in b) and the initial median in a). If a data quality `dq` 
    array is given, this is used to not include outliers/bad pixels in the calculations. If it's not given, the 
    function estimates those after a first pass from (a) and (b) above, and then repeats that process.

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


    Returns
    -------
    np.array
        Array containing the median-and-bias corrected integrations per group; zero values are outliers/bad pixels.

    """

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

                corrected_darks[i, j, :, :][np.isnan(corrected_darks[i, j, :, :])] = 0.
                corrected_darks[i, j, :, :][np.isnan(nanarray[i, j, :, :])] = 0.

                # Recompute median:
                corrected_darks[i, j, :, :] = corrected_darks[i, j, :, :] - np.nanmedian( corrected_darks[i, j, :, :] * nanarray[i, j, :, :] )

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
