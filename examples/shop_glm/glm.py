# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Example analyzing the FIAC dataset with NIPY.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib
import warnings
from tempfile import NamedTemporaryFile
from os.path import join as pjoin, exists as pexists
from os import makedirs
from glob import glob

# Third party
import numpy as np
from progressbar import ProgressBar

# From NIPY
from nipy.fixes.scipy.stats.models.regression import (OLSModel, ARModel,
                                                      isestimable )
from nipy.modalities.fmri import formula, design, hrf
from nipy.io.api import load_image, save_image
from nipy.core import api
from nipy.core.image.image import rollaxis as image_rollaxis
from nipy.core.reference.coordinate_map import drop_io_dim
from nipy.core.api import Image
from nipy.algorithms.statistics import onesample 

# Local
import util as shop_util
import subject as shop_subj
reload(shop_util)  # while developing interactively

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

GROUP_MASK = np.ones(shop_subj.Subject('nc').fmri.shape[:3]) # no mask
TINY_MASK = np.zeros(GROUP_MASK.shape, np.bool)
TINY_MASK[30:32,40:42,30:32] = 1

#-----------------------------------------------------------------------------
# Public functions
#-----------------------------------------------------------------------------

# For group analysis

def run_model(subj):
    """
    Single subject fitting of SHOP model
    """
    #----------------------------------------------------------------------
    # Set initial parameters of the FIAC dataset
    #----------------------------------------------------------------------
    # Number of volumes in the fMRI data
    nvol = shop_util.FRAME_TIMES.shape[0]
    # The TR of the experiment
    TR = shop_util.TR
    # The time of the first volume
    Tstart = 0.0
    # The array of times corresponding to each 
    # volume in the fMRI data
    frame_times = shop_util.FRAME_TIMES
    # This recarray of times has one column named 't'
    # It is used in the function design.event_design
    # to create the design matrices.

    #----------------------------------------------------------------------
    # Experimental design, handled in Subject class
    #----------------------------------------------------------------------

    subject = shop_subj.Subject(subj)
    frame_times = subject.datarec['t']
    model_formula = subject.formula('model')
    X, cons = model_formula.design(subject.datarec,
                                   contrasts=subject.contrasts)

    #----------------------------------------------------------------------
    # Data loading
    #----------------------------------------------------------------------
    
    # Load in the fMRI data, saving it as an array
    # It is transposed to have time as the first dimension,
    # i.e. fmri[t] gives the t-th volume.

    fmri_im = image_rollaxis(subject.fmri, 't')

    fmri = fmri_im.get_data() # now, it's an ndarray

    nvol, volshape = fmri.shape[0], fmri.shape[1:] 
    nslice, sliceshape = volshape[0], volshape[1:]

    #----------------------------------------------------------------------
    # Model fit
    #----------------------------------------------------------------------

    # The model is a two-stage model, the first stage being an OLS (ordinary
    # least squares) fit, whose residuals are used to estimate an AR(1)
    # parameter for each voxel.

    m = OLSModel(X)
    ar1 = np.zeros(volshape)

    # Fit the model, storing an estimate of an AR(1) parameter at each voxel

    pbar = ProgressBar(maxval=nslice)
    for s in range(nslice):
        pbar.update(s+1)
        d = np.array(fmri[:,s])
        flatd = d.reshape((d.shape[0], -1))
        result = m.fit(flatd)
        ar1[s] = ((result.resid[1:] * result.resid[:-1]).sum(0) /
                  (result.resid**2).sum(0)).reshape(sliceshape)

    # We round ar1 to nearest one-hundredth
    # and group voxels by their rounded ar1 value,
    # fitting an AR(1) model to each batch of voxels.

    # XXX smooth here?
    # ar1 = smooth(ar1, 8.0)

    ar1 *= 100
    ar1 = ar1.astype(np.int) / 100.
    ar1 = np.clip(ar1, -1, 1)

    # Setup a dictionary to hold all the output
    # XXX ideally these would be memmap'ed Image instances

    output = {}
    for n in cons:
        tempdict = {}
        for v in ['sd', 't', 'effect']:
            tempdict[v] = np.memmap(NamedTemporaryFile(prefix='%s%s.nii' \
                                                       % (n,v)), dtype=np.float, 
                                    shape=volshape, mode='w+')
        output[n] = tempdict
    
    # Loop over the unique values of ar1

    for val in np.unique(ar1):
        if val != -1.:
            armask = np.equal(ar1, val)
            m = ARModel(X, val)
            d = fmri[:,armask]
            results = m.fit(d)

            # Output the results for each contrast

            for n in cons:
                resT = results.Tcontrast(cons[n])
                output[n]['sd'][armask] = resT.sd
                output[n]['t'][armask] = resT.t
                output[n]['effect'][armask] = resT.effect

    # Dump output to disk
    odir = pjoin(shop_util.DATADIR, 'results', subj)
    coordmap = drop_io_dim(fmri_im.coordmap, 't')

    for n in cons:
        if not pexists(pjoin(odir, n)):
            makedirs(pjoin(odir, n))
        for v in ['t', 'sd', 'effect']:
            im = Image(output[n][v], coordmap)
            save_image(im, pjoin(odir, n, '%s.nii' % v))


def fixed_effects(contrast):
    """
    Fixed effects for SHOP data
    """

    # First, find all the effect and standard deviation images
    # for the subject and this design type

    subjects = glob(pjoin(shop_util.DATADIR, 'results', '*', contrast))
    
    fixdir = pjoin(pjoin(shop_util.DATADIR, 'fixed_results'))
    effects = [pjoin(s, 'effect.nii') for s in subjects]
    sds = [pjoin(s, 'sd.nii') for s in subjects]

    # Get our hands on the relevant coordmap to
    # save our results
    coordmap = load_image(effects[0]).coordmap

    # Compute the "fixed" effects 

    fixed_effect = 0
    fixed_var = 0
    for effect, sd in zip(effects, sds):
        effect_im = load_image(effect); sd_im = load_image(sd)
        var = sd_im.get_data()**2

        # The optimal, in terms of minimum variance, combination of the
        # effects has weights 1 / var
        #
        # XXX regions with 0 variance are set to 0
        # XXX do we want this or np.nan?
        ivar = np.nan_to_num(1. / var)
        fixed_effect += effect_im.get_data() * ivar
        fixed_var += ivar

    # Now, compute the fixed effects variance and t statistic
    fixed_sd = np.sqrt(fixed_var)
    isd = np.nan_to_num(1. / fixed_sd)
    fixed_t = fixed_effect * isd

    # Save the results
    odir = shop_util.ensure_dir(fixdir, contrast)
    for a, n in zip([fixed_effect, fixed_sd, fixed_t],
                    ['effect', 'sd', 't']):
        old = coordmap.affine
        im = Image(a, coordmap)
        save_image(im, pjoin(odir, '%s.nii' % n))
        
def group_analysis(design, contrast):
    """
    Compute group analysis effect, sd and t
    for a given contrast and design type
    """
    array = np.array # shorthand
    # Directory where output will be written
    odir = futil.ensure_dir(futil.DATADIR, 'group', design, contrast)

    # Which subjects have this (contrast, design) pair?
    subjects = futil.subject_dirs(design, contrast)

    sd = array([array(load_image(pjoin(s, "sd.nii"))) for s in subjects])
    Y = array([array(load_image(pjoin(s, "effect.nii"))) for s in subjects])

    # This function estimates the ratio of the
    # fixed effects variance (sum(1/sd**2, 0))
    # to the estimated random effects variance
    # (sum(1/(sd+rvar)**2, 0)) where
    # rvar is the random effects variance.

    # The EM algorithm used is described in 
    #
    # Worsley, K.J., Liao, C., Aston, J., Petre, V., Duncan, G.H., 
    #    Morales, F., Evans, A.C. (2002). \'A general statistical 
    #    analysis for fMRI data\'. NeuroImage, 15:1-15

    varest = onesample.estimate_varatio(Y, sd)
    random_var = varest['random']

    # XXX - if we have a smoother, use
    # random_var = varest['fixed'] * smooth(varest['ratio'])

    # Having estimated the random effects variance (and
    # possibly smoothed it), the corresponding
    # estimate of the effect and its variance is
    # computed and saved.

    # This is the coordmap we will use
    coordmap = futil.load_image_fiac("fiac_00","wanatomical.nii").coordmap

    adjusted_var = sd**2 + random_var
    adjusted_sd = np.sqrt(adjusted_var)

    results = onesample.estimate_mean(Y, adjusted_sd) 
    for n in ['effect', 'sd', 't']:
        im = api.Image(results[n], coordmap.copy())
        save_image(im, pjoin(odir, "%s.nii" % n))


def group_analysis_signs(design, contrast, mask, signs=None):
    """
    This function refits the EM model with a vector of signs.
    Used in the permutation tests.

    Returns the maximum of the T-statistic within mask

    Parameters
    ----------

    design: one of 'block', 'event'

    contrast: str

    mask: array-like

    signs: ndarray, optional
         Defaults to np.ones. Should have shape (*,nsubj)
         where nsubj is the number of effects combined in the group analysis.

    Returns
    -------

    minT: np.ndarray, minima of T statistic within mask, one for each
         vector of signs

    maxT: np.ndarray, maxima of T statistic within mask, one for each
         vector of signs
    
    """

    maska = np.asarray(mask).astype(np.bool)

    # Which subjects have this (contrast, design) pair?

    subjects = futil.subject_dirs(design, contrast)

    sd = np.array([np.array(load_image(pjoin(s, "sd.nii")))[:,maska]
                   for s in subjects])
    Y = np.array([np.array(load_image(pjoin(s, "effect.nii")))[:,maska]
                  for s in subjects])

    if signs is None:
        signs = np.ones((1, Y.shape[0]))

    maxT = np.empty(signs.shape[0])
    minT = np.empty(signs.shape[0])

    for i, sign in enumerate(signs):
        signY = sign[:,np.newaxis] * Y
        varest = onesample.estimate_varatio(signY, sd)
        random_var = varest['random']

        adjusted_var = sd**2 + random_var
        adjusted_sd = np.sqrt(adjusted_var)

        results = onesample.estimate_mean(Y, adjusted_sd) 
        T = results['t']
        minT[i], maxT[i] = np.nanmin(T), np.nanmax(T)
    return minT, maxT


def permutation_test(design, contrast, mask=GROUP_MASK,
                     nsample=1000):
    """
    Perform a permutation (sign) test for a given design type and
    contrast. It is a Monte Carlo test because we only sample nsample
    possible sign arrays.

    Parameters
    ----------

    design: one of ['block', 'event']

    contrast: str

    nsample: int

    Returns
    -------

    min_vals: np.ndarray

    max_vals: np.ndarray
    """

    maska = np.asarray(mask).astype(np.bool)

    subjects = futil.subject_dirs(design, contrast)
    
    Y = np.array([np.array(load_image(pjoin(s, "effect.nii")))[:,maska]
                  for s in subjects])
    nsubj = Y.shape[0]

    signs = 2*np.greater(np.random.sample(size=(nsample, nsubj)), 
                         0.5) - 1
    ipvars('signs')
    min_vals, max_vals = group_analysis_signs(design, 
                                              contrast, 
                                              maska, 
                                              signs)
    return min_vals, max_vals


if __name__ == '__main__':
    pass
    # Sanity check while debugging
    #permutation_test('block','sentence_0',mask=TINY_MASK,nsample=3)
