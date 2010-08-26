# Stdlib

from os.path import join as pjoin

# Third party imports

import numpy as np
from matplotlib.mlab import csv2rec, rec_append_fields, rec2csv
import pylab

# Nipy imports

from nipy.core.api import Image
import nipy.modalities.fmri as F

# Local

import util as shop_util

class Subject(object):

    def __init__(self, subject):
        self.subject = subject
        f = shop_util.load_subject_shop(subject)
        self.fmri = f['fmri']
        self.behavior = f['behavior']

    @property
    def buy(self):
        return shop_util.impute(self.behavior['buy'])

    @property
    def pref(self):
        return shop_util.impute(self.behavior['pref'])

    @property
    def ppe(self):
        return shop_util.impute(self.behavior['ppe'])

    @property
    def buy_regressor(self):
        return F.utils.blocks(zip(shop_util.BUY_START, shop_util.BUY_END), self.buy, name='buy')

    @property
    def ppe_regressor(self):
        return F.utils.blocks(zip(shop_util.PPE_START, shop_util.PPE_END), self.ppe, name='ppe')

    @property
    def pref_regressor(self):
        return F.utils.blocks(zip(shop_util.PREF_START, shop_util.PREF_END), self.pref, name='pref')

    def _convolve(self, regressor, name=None):
        t = F.formula.Term('t')
        return F.utils.convolve_functions(regressor,
                                          F.hrf.afni(t),
                                          [0,
                                           shop_util.FRAME_TIMES.max()+10],
                                          0.01,
                                          name=name)

    @property
    def buy_convolved(self):
        if not hasattr(self, "_buy_c"):
            self._buy_c = self._convolve(self.buy_regressor, name='buy_c')
        return self._buy_c
    
    @property
    def ppe_convolved(self):
        if not hasattr(self, "_ppe_c"):
            self._ppe_c = self._convolve(self.ppe_regressor, name='ppe_c')
        return self._ppe_c

    @property
    def pref_convolved(self):
        if not hasattr(self, "_pref_c"):
            self._pref_c = self._convolve(self.pref_regressor, name='pref_c')
        return self._pref_c

    @property
    def drift(self):
        t = F.formula.Term('t')
        midpt = shop_util.TR * (shop_util.FRAME_TIMES.shape[0]/2)
        intervals = [[0, midpt],
                     [midpt, shop_util.TR * shop_util.FRAME_TIMES.shape[0]]]
        fns = []
        for r, interval in zip([1,2], intervals):
            for o in range(3):
                def f(x, o=o, u=interval[1], l=interval[0]):
                    return (x-l)**o * (x >= l) *  (x < u)
                fns.append(F.formula.aliased_function("run_%d_drift_%d" % (r, o), f)(t))
        return F.formula.Formula(fns)

    @property
    def motion(self):
        for d in ['cData', 'allData']:
            try:
                run1 = csv2rec(pjoin(shop_util.DATADIR, '..',
                                     "%s_%s" % (self.subject, d),
                                     "3dmotionSMG1.1D"),
                               delimiter=' ',
                               names=['tr'] + ["V%d" % i for i in range(7)])
                run2 = csv2rec(pjoin(shop_util.DATADIR, '..',
                                     "%s_%s" % (self.subject, d),
                                     "3dmotionSMG2.1D"),
                               delimiter=' ',
                               names=['tr'] + ["V%d" % i for i in range(7)])
            except:
                pass
        return np.hstack([run1, run2])

    @property
    def datarec(self, doc='Recarray used to construct design matrix'):
        return rec_append_fields(self.motion, 't', shop_util.FRAME_TIMES)

    @property
    def time(self, doc="Recarray used to construct finer" +
             " resolution time plots"):
        dtype = np.dtype([('t', np.float)])
        return np.arange(0, shop_util.FRAME_TIMES.max()+1, 0.5).view(dtype)
    

    @property
    def time_formula(self, doc='Formula of objects that are functions' +
                     " of time, and can be evaluated on a finer time scale."):
        t = F.formula.Term('t')
        return self.drift + F.formula.Formula([self.buy_convolved,
                                               self.pref_convolved,
                                               self.ppe_convolved,
                                               self.buy_regressor,
                                               self.pref_regressor,
                                               self.ppe_regressor,
                                               t])
    
    def formula(self, which):
        if which == 'all':
            return self.time_formula + F.formula.Formula(F.formula.terms(*self.motion.dtype.names[1:]))
        elif which == 'model':
            return (self.drift +
                    F.formula.Formula([self.buy_convolved,
                                       self.ppe_convolved,
                                       self.pref_convolved]) +
                    F.formula.Formula(F.formula.terms(*self.motion.dtype.names[1:])))
        else:
            raise ValueError('formula arguments are ["all", "model"]')

    @property
    def contrasts(self):
        return dict(zip(['buy', 'ppe', 'pref'],
                        [F.formula.Formula([c]) for c in
                         [self.buy_convolved,
                          self.ppe_convolved,
                          self.pref_convolved]]))
    
def plot_subject(subject):
    import os
    from tempfile import mkdtemp
    s = Subject(subject)

    tmpd = mkdtemp()
    d = s.time_formula.design(s.time)

    for n in sorted(d.dtype.names):
        if n != 't':
            pylab.figure(num=1, figsize=(6,6))
            pylab.clf()
            pylab.plot(d['t'], d[n])
            a = pylab.gca()
            a.set_xlabel('t')
            a.set_ylabel(n)
            pylab.savefig("%s/%s.pdf" % (tmpd, n.replace('(t)', '')))

            if n[:3] != 'run':
                pylab.figure(num=1, figsize=(6,6))
                pylab.clf()
                if "_c" in n:
                    pylab.plot(d['t'][50:150], d[n][50:150])
                else:
                    pylab.step(d['t'][50:150], d[n][50:150], where='post')
                a = pylab.gca()
                a.set_xlabel('t')
                a.set_ylabel(n)
                pylab.savefig("%s/%s_fine.pdf" % (tmpd, n.replace('(t)', '')))

    fd = os.path.abspath(os.path.dirname(__file__))
    os.system('''
    cd %s;
    pdftk *pdf cat output design_timecourse_%s.pdf;
    cp file_%s.pdf %s;
    cd ; rm -fr %s
    '''  % (tmpd, subject, subject, fd, tmpd))

def save_design(subject):
    s = Subject(subject)
    rec2csv(s.formula('all').design(s.datarec), 'design_%s.csv' % subject)
