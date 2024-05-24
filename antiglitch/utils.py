import os
import numpy as np
from numpy.fft import rfft, irfft
import scipy.signal as sig

from functools import partial
rfft = partial(rfft, norm='ortho')
irfft = partial(irfft, norm='ortho')

from .model import fglitch_from_sample, fglitch_normed


def center(data):
    """Shift data from the start to the center of a time series"""
    return np.roll(data, len(data)//2)

def downsample_invasd(invasd, tlen=1024):
    """Reduce frequency resolution of the inverse ASD to a given number of time samples"""
    tmp = np.abs(rfft(sig.hann(tlen)*np.roll(irfft(invasd), tlen//2)[:tlen]))
    tmp[0] = 0.
    return tmp

def to_fd(data):
    """Convert to frequency domain with t=0 in center"""
    return rfft(np.roll(data, -len(data)//2))

def extract_glitch(npz, halfwidth=512):
    """Returns 1/4 second inverse ASD and 2 sec of whitened glich, @8192Hz"""
    invasd = ((4096.*npz['psd'])**-0.5)[:4097]
    invasd[:10] = 0.
    filt = np.zeros(4*8192)
    filt[:8192] = sig.hann(8192)*np.roll(irfft(invasd), 4096)
    fdfilt = np.abs(rfft(filt)) # This is a one-second resolution invasd

    whts = irfft(fdfilt*rfft(npz['data']))
    invasd_ds = downsample_invasd(invasd)

    return invasd_ds, whts[len(whts)//2 - halfwidth:len(whts)//2 + halfwidth],  whts

def snr(snip):
    tmp = np.zeros(4*8192)
    tmp[:len(snip.glitch)] = snip.glitch.copy()
    gt = TimeSeries(np.roll(tmp,-len(snip.glitch)//2), delta_t=1/8192)
    dt = TimeSeries(snip.whts_long, delta_t=1/8192)
    dt[:4096]=0.;dt[len(dt)-4096:]=0.;
    snrts = matched_filter(gt, dt, psd=FrequencySeries([1./4096]*(1+4*4906), delta_f=1/4.), low_frequency_cutoff=15)

    return np.abs(snrts[16384-2048:16384+2048]).max()
    
def measure(inf_data, snip):
    itrace = inf_data['mean']
    inf = {key: float(itrace[key])
            for key in ['amp_r', 'amp_i', 'f0','gbw','time']}
    strace = inf_data['sd']
    inf |= {key+'_sd': float(strace[key])
            for key in ['amp_r', 'amp_i', 'f0','gbw','time']}
    cpamp = inf['amp_r']+1.j*inf['amp_i']
    inf['amp'] = np.abs(cpamp)
    inf['phase'] = np.angle(cpamp)
    inf['time'] = inf['time']
    snip.set_infer(inf)
    inf['snr'] = snr(snip)
    inf['power'] = np.sum((snip.whts[256:768])**2)/512
    inf['residual'] = np.sum(((snip.whts-snip.glitch)[256:768])**2)/512
    inf['peak_frequency'] = float(freqs[(snip.invasd*np.abs(snip.fglitch)).argmax()])
    return inf, snip

class Snippet:
    """A class to load and whiten data, """
    def __init__(self, ifo, key, num, datadir):
        self.ifo, self.key, self.num = ifo, key, num
        npz = np.load(f"{datadir}/{ifo}-{key}-{num:04d}.npz")
        self.invasd, self.whts, self.whts_long = extract_glitch(npz)
    def set_infer(self, inf):
        self.inf = inf
    @property
    def fglitch(self):
        """Glitch centered at the start index"""
        return fglitch_from_sample(**self.inf)
    @property
    def glitch(self):
        """Whitened glitch time series"""
        ftmp = fglitch_from_sample(**self.inf)
        tmp = irfft(self.invasd * ftmp)
        return center(tmp)

    def plot(self):
        myfreqs = np.linspace(0,4096,513)
        
        import matplotlib.pyplot as plt
        fig1 = plt.figure()
        plt.title(f"{self.ifo} {self.key} {self.num}")
        plt.loglog(myfreqs, np.abs(rfft(self.whts)), c='k', lw=1)
        plt.loglog(myfreqs, np.abs(rfft(self.glitch)), c='b', lw=1)
        plt.loglog(myfreqs, np.abs(rfft(self.whts - self.glitch)), c='orange')
        plt.ylim(1e-2,1e2)
        
        fig2 = plt.figure()
        plt.title(f"{self.ifo} {self.key} {self.num}")
        plt.plot(self.whts, c='k', lw=1)
        plt.plot(self.whts - self.glitch, c='orange', ls=':')
        
        return fig1, fig2

class SnippetNormed:
    """A class to load and whiten data, """
    def __init__(self, ifo, key, num, datadir):
        self.ifo, self.key, self.num = ifo, key, num
        npz = np.load(f"{datadir}/{ifo}-{key}-{num:04d}.npz")
        self.invasd, self.whts, self.whts_long = extract_glitch(npz)
    def set_infer(self, inf):
        self.inf = inf
    @property
    def fglitch(self):
        """Glitch centered at the start index"""
        return fglitch_normed(self.invasd, **self.inf)
    @property
    def glitch(self):
        """Whitened glitch time series"""
        ftmp = fglitch_normed(self.invasd, **self.inf)
        tmp = irfft(ftmp)
        return center(tmp)

    def plot(self):
        myfreqs = np.linspace(0,4096,513)
        
        import matplotlib.pyplot as plt
        fig1 = plt.figure()
        plt.title(f"{self.ifo} {self.key} {self.num}")
        plt.loglog(myfreqs, np.abs(rfft(self.whts)), c='k', lw=1)
        plt.loglog(myfreqs, np.abs(rfft(self.glitch)), c='b', lw=1)
        plt.loglog(myfreqs, np.abs(rfft(self.whts - self.glitch)), c='orange')
        plt.ylim(1e-2,1e2)
        
        fig2 = plt.figure()
        plt.title(f"{self.ifo} {self.key} {self.num}")
        plt.plot(self.whts, c='k', lw=1)
        plt.plot(self.whts - self.glitch, c='orange', ls=':')
        
        return fig1, fig2

