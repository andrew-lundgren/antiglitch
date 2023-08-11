import os
import numpy as np
from numpy.fft import rfft, irfft
import scipy.signal as sig

from functools import partial
rfft = partial(rfft, norm='ortho')
irfft = partial(irfft, norm='ortho')

from .model import fglitch_from_sample

for path in ['data', '/home/andrew.lundgren/detchar/GlitchSearch/data']:
    if os.path.isdir(path):
        datadir = path
        break

def center(data):
    return np.roll(data, len(data)//2)

def downsample_invasd(invasd, tlen=1024):
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
    fdfilt = np.abs(rfft(filt))

    whts = irfft(fdfilt*rfft(npz['data']))
    invasd_ds = downsample_invasd(invasd)

    return invasd_ds, whts[len(whts)//2 - halfwidth:len(whts)//2 + halfwidth],  whts

class Snippet:
    """A class to load and whiten data, """
    def __init__(self, ifo, key, num):
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

