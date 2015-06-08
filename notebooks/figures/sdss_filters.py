"""
SDSS Filters
------------

This example downloads and plots the filters from the Sloan Digital Sky
Survey, along with a reference spectrum.
"""
import os
import urllib2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow

DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'downloads')
REFSPEC_URL = 'ftp://ftp.stsci.edu/cdbs/current_calspec/1732526_nic_002.ascii'
FILTER_URL = 'http://www.sdss.org/dr7/instruments/imager/filters/%s.dat'

def fetch_filter(filt):
    assert filt in 'ugriz'
    url = FILTER_URL % filt
    
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    loc = os.path.join(DOWNLOAD_DIR, '%s.dat' % filt)
    if not os.path.exists(loc):
        print "downloading from %s" % url
        F = urllib2.urlopen(url)
        open(loc, 'w').write(F.read())

    F = open(loc)
        
    data = np.loadtxt(F)
    return data


def fetch_vega_spectrum():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    refspec_file = os.path.join(DOWNLOAD_DIR, REFSPEC_URL.split('/')[-1])

    if  not os.path.exists(refspec_file):
        print "downloading from %s" % REFSPEC_URL
        F = urllib2.urlopen(REFSPEC_URL)
        open(refspec_file, 'w').write(F.read())

    F = open(refspec_file)

    data = np.loadtxt(F)
    return data


def plot_sdss_filters():
    Xref = fetch_vega_spectrum()
    Xref[:, 1] /= 2.1 * Xref[:, 1].max()

    #----------------------------------------------------------------------
    # Plot filters in color with a single spectrum
    fig, ax = plt.subplots()
    ax.plot(Xref[:, 0], Xref[:, 1], '-k', lw=2)

    for f,c in zip('ugriz', 'bgrmk'):
        X = fetch_filter(f)
        ax.fill(X[:, 0], X[:, 1], ec=c, fc=c, alpha=0.4)

    kwargs = dict(fontsize=20, ha='center', va='center', alpha=0.5)
    ax.text(3500, 0.02, 'u', color='b', **kwargs)
    ax.text(4600, 0.02, 'g', color='g', **kwargs)
    ax.text(6100, 0.02, 'r', color='r', **kwargs)
    ax.text(7500, 0.02, 'i', color='m', **kwargs)
    ax.text(8800, 0.02, 'z', color='k', **kwargs)

    ax.set_xlim(3000, 11000)

    ax.set_title('SDSS Filters and Reference Spectrum')
    ax.set_xlabel('Wavelength (Angstroms)')
    ax.set_ylabel('normalized flux / filter transmission')


def plot_redshifts():
    Xref = fetch_vega_spectrum()
    Xref[:, 1] /= 2.1 * Xref[:, 1].max()

    #----------------------------------------------------------------------
    # Plot filters in gray with several redshifted spectra
    fig, ax = plt.subplots()

    redshifts = [0.0, 0.4, 0.8]
    colors = 'bgr'

    for z, c in zip(redshifts, colors):
        plt.plot((1. + z) * Xref[:, 0], Xref[:, 1], color=c)

    ax.add_patch(Arrow(4200, 0.47, 1300, 0, lw=0, width=0.05, color='r'))
    ax.add_patch(Arrow(5800, 0.47, 1250, 0, lw=0, width=0.05, color='r'))

    ax.text(3800, 0.49, 'z = 0.0', fontsize=14, color=colors[0])
    ax.text(5500, 0.49, 'z = 0.4', fontsize=14, color=colors[1])
    ax.text(7300, 0.49, 'z = 0.8', fontsize=14, color=colors[2])

    for f in 'ugriz':
        X = fetch_filter(f)
        ax.fill(X[:, 0], X[:, 1], ec='k', fc='k', alpha=0.2)

    kwargs = dict(fontsize=20, color='gray', ha='center', va='center')
    ax.text(3500, 0.02, 'u', **kwargs)
    ax.text(4600, 0.02, 'g', **kwargs)
    ax.text(6100, 0.02, 'r', **kwargs)
    ax.text(7500, 0.02, 'i', **kwargs)
    ax.text(8800, 0.02, 'z', **kwargs)

    ax.set_xlim(3000, 11000)
    ax.set_ylim(0, 0.55)

    ax.set_title('Redshifting of a Spectrum')
    ax.set_xlabel('Observed Wavelength (Angstroms)')
    ax.set_ylabel('normalized flux / filter transmission')


if __name__ == '__main__':
    plot_sdss_filters()
    plot_redshifts()
    plt.show()
