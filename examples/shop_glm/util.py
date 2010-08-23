# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Support utilities for FIAC example, mostly path management.

The purpose of separating these is to keep the main example code as readable as
possible and focused on the experimental modeling and analysis, rather than on
local file management issues.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
# Stdlib
import os

from StringIO import StringIO
from os import makedirs, listdir
from os.path import exists, abspath, isdir, join as pjoin

# Third party
import numpy as np
from matplotlib.mlab import csv2rec, rec2csv

# From NIPY
from nipy.io.api import load_image, save_image
from nipy.core.image.xyz_image import XYZImage

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

# We assume that there is a directory holding the data and it's local to this
# code.  Users can either keep a copy here or a symlink to the real location on
# disk of the data.
DATADIR = 'fiac_data'

# Sanity check
if not os.path.isdir(DATADIR):
    e="The data directory %s must exist and contain the FIAC data." % DATADIR
    raise IOError(e)

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

# Path management utilities
def load_image_fiac(*path):
    """Return a NIPY image from a set of path components.
    """
    return load_image(pjoin(DATADIR, *path))


def subject_dirs(design, contrast, nsub=16):
    """Return a list of subject directories.
    """
    rootdir = DATADIR
    subjects = [ f for f in  [pjoin(rootdir, "fiac_%02d" % s, design, "fixed",
                                    contrast) for s in range(nsub)]
                 if isdir(f) ]
    return subjects

    
def path_info(subj,run):
    """Construct path ifnormation dict for current subject/run.

    Returns
    -------
    path_dict : a dict with all the necessary path-related keys, including
    'rootdir'.
    """
    path_dict = {'subj':subj, 'run':run}
    if exists(pjoin(DATADIR, "fiac_%(subj)02d",
                    "block", "initial_%(run)02d.csv") % path_dict):
        path_dict['design'] = 'block'
    else:
        path_dict['design'] = 'event'
    rootdir = pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s") % path_dict
    path_dict['rootdir'] = rootdir
    return path_dict


def path_info2(subj,design):
    path_dict = {'subj':subj, 'design':design}
    rootdir = pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s") % path_dict
    path_dict['rootdir'] = rootdir
    return path_dict
    

def results_table(path_dict):
    # Which runs correspond to this design type?
    rootdir = path_dict['rootdir']
    runs = filter(lambda f: isdir(pjoin(rootdir, f)),
                  ['results_%02d' % i for i in range(1,5)] )

    # Find out which contrasts have t-statistics,
    # storing the filenames for reading below

    results = {}

    for rundir in runs:
        rundir = pjoin(rootdir, rundir)
        for condir in listdir(rundir):
            for stat in ['sd', 'effect']:
                fname_effect = abspath(pjoin(rundir, condir, 'effect.nii'))
                fname_sd = abspath(pjoin(rundir, condir, 'sd.nii'))
            if exists(fname_effect) and exists(fname_sd):
                results.setdefault(condir, []).append([fname_effect,
                                                       fname_sd])
    return results


def get_experiment_initial(path_dict):
    """Get the record arrays for the experimental/initial designs.

    Returns
    -------
    experiment, initial : Two record arrays.

    """
    # The following two lines read in the .csv files
    # and return recarrays, with fields
    # experiment: ['time', 'sentence', 'speaker']
    # initial: ['time', 'initial']

    rootdir = path_dict['rootdir']
    if not exists(pjoin(rootdir, "experiment_%(run)02d.csv") % path_dict):
        e = "can't find design for subject=%(subj)d,run=%(subj)d" % path_dict
        raise IOError(e)

    experiment = csv2rec(pjoin(rootdir, "experiment_%(run)02d.csv") % path_dict)
    initial = csv2rec(pjoin(rootdir, "initial_%(run)02d.csv") % path_dict)

    return experiment, initial


def get_fmri(path_dict):
    """Get the images for a given subject/run.

    Returns
    -------
    fmri : ndarray
    
    anat : NIPY image
    """
    fmri_im = load_image(
        pjoin("%(rootdir)s/swafunctional_%(run)02d.nii") % path_dict) 

    # Make sure we know the order of the coordinates

    fmri_im = fmri_im.reordered_world('xyzt').reordered_axes('ijkl')

    return LPIImage.from_image(fmri_im)

def ensure_dir(*path):
    """Ensure a directory exists, making it if necessary.

    Returns the full path."""
    dirpath = pjoin(*path)
    if not isdir(dirpath):
        makedirs(dirpath)
    return dirpath

def output_dir(path_dict,tcons,fcons):
    """Get (and make if necessary) directory to write output into.
    """
    rootdir = path_dict['rootdir']
    odir = pjoin(rootdir, "results_%(run)02d" % path_dict)
    ensure_dir(odir)
    for n in tcons:
        ensure_dir(odir,n)
    for n in fcons:
        ensure_dir(odir,n)

    return odir
                   

def test_sanity():
    from nipy.modalities.fmri.fmristat.tests import FIACdesigns

    """
    Single subject fitting of FIAC model
    """

    # Based on file
    # subj3_evt_fonc1.txt
    # subj3_bloc_fonc3.txt

    for subj, run, dtype in [(3,1,'event'),
                             (3,3,'block')]:
        nvol = 191
        TR = 2.5 
        Tstart = 1.25

        volume_times = np.arange(nvol)*TR + Tstart
        volume_times_rec = formula.make_recarray(volume_times, 't')

        path_dict = {'subj':subj, 'run':run}
        if exists(pjoin(DATADIR, "fiac_%(subj)02d",
                        "block", "initial_%(run)02d.csv") % path_dict):
            path_dict['design'] = 'block'
        else:
            path_dict['design'] = 'event'

        experiment = csv2rec(pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s", "experiment_%(run)02d.csv")
                             % path_dict)
        initial = csv2rec(pjoin(DATADIR, "fiac_%(subj)02d", "%(design)s", "initial_%(run)02d.csv")
                                % path_dict)

        X_exper, cons_exper = design.event_design(experiment, volume_times_rec, hrfs=delay.spectral)
        X_initial, _ = design.event_design(initial, volume_times_rec, hrfs=[hrf.glover]) 
        X, cons = design.stack_designs((X_exper, cons_exper),
                                       (X_initial, {}))

        Xf = np.loadtxt(StringIO(FIACdesigns.designs[dtype]))
        for i in range(X.shape[1]):
            yield nitest.assert_true, (matchcol(X[:,i], Xf.T)[1] > 0.999)


