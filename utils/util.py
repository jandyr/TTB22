#!/usr/bin/python3
#--------  Code Dependencies   ----------
#\__________General Utilities____________/
import pprint
import time
import timeit
#import argparse
import sys
import os
import subprocess
import platform
#import psutil
import gc
import re
import copy
import numpy as np
from scipy import fftpack
from scipy import signal
from scipy.signal import find_peaks, peak_widths
import scipy.fft
#from scipy import optimize
#from scipy.optimize import minimize
#from scipy.interpolate import griddata
#\__________Plotting__________/
import matplotlib.pyplot as plt
#\__________Local functions__________/
import inout.ipop as ipop             #I/O functions
import plots.plot as p                #Plot functions
#\__________ObsPy functions__________/
from obspy import read
from obspy import UTCDateTime
import obspy.signal                    # To estimate envelpe
from obspy.signal.array_analysis import array_processing
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.invsim import corn_freq_2_paz
#\__________mtspec functions__________/
from multitaper import mtspec
import multitaper.mtspec as spec
import multitaper.utils  as mtutils
import multitaper.mtcross  as mtcross
"""
multitaper.utils uses Numba, which translates
Python functions to optimized machine code at
runtime using the LLVM compiler library.
"""
#\_____________________________________/
#
#\__________Scripts List__________/
"""
Script         Description


# ---------- Beamforming - FK Analysis  ----------
BeamFK
spect          Spectrogram
rfreq          Flatten and restrict array range
anF            F-test peak analysing script
multcint       Multitaper Spectrum
preproc        Trace preprocessing
Pol_Car        Convert Polar <-> Cartezian
pline          Print stuff along a line
bar            Displays progress with a bar on terminal
"""
#\__________Scripts__________/

"""
t,freq,qi,adap = spec.spectrogram(x,dt,twin=20.,olap=0.5,
nw,kspec)
"""
#
# ---------- Spectogram  ----------
def spect(tr, dt, nw , kspec):
    """ 
    See multcint for parameters


    twin (int or float) – Window length for fft in seconds.
    If this parameter is too small, the calculation will take forever.
    If None, it defaults to a window length matching 128 samples.

    olap=0.5 Overlap
    """ 
#
#------------ Relevant parameters
    pline(['\n>> Choose parameters.'], width=13, align='^')
    pline(['    1) Time window length (s) [0=adaptative].'], width=13, align='^')
    pline(['    2) Overlap [0.5]'], width=13, align='^')
    pline(['    3) Minimum frequency [Fmin]'], width=13, align='^')
    pline(['    3) Maximum frequency [Fmax]'], width=13, align='^')
#-- Default list -> ent
    dummy = '0 .99 n'
#           /    \ +-> do not estimate reshaped PSD estimate
#       method  Crictical F-test probability
#ent[i] =  0      1         2
    t2,freq,QIspec,MTspec = spec.spectrogram(x,dt,twin=20.,olap=0.5,
                                   nw=nw,kspec=5,fmin=0.05,fmax=20.)
#
# -------------- End of function   ----------------------------
#
# ---------- Beamforming - FK Analysis  ----------
def BeamFK(st, MTparam, supl_data, **kwargs):

    """ 
    Perform FK analysis with ObsPy. The data is bandpass filtered, prewhitening disabled.
    <Arguments>
    st                -> A stream
    MTparam           ->
    supl_data         ->
    stime, etime      -> Relative time limits (s) on the time window for event,
                          later to be transformed to UTCDateTime format.
    sll_o, sl_s       -> Slowness grid (s/km) and step fraction.
                 slm_y +─────+
                       │     │
                       │     │
                 sll_y +─────+    
                    sll_x   slm_x
    win_len, win_frac -> window length and step fraction (s).
    semb_thres, vel_thres -> infinitesimally small numbers; must not be changed.
    timestamp         -> written in 'mlabday', read directly by plotting routine.
    """ 
#
#------------ Select event times --------------
    pline(['\n>> Select start and end times (s) for beanforming'], width=13, align='^')
    pline(['  stime etime'], width=13, align='^')
    pline(['    └─────│──> Initial event time'], width=13, align='^')
    pline(['          └──> Final event time'], width=13, align='^')
    ent = input(f'   Enter t0 t1:\n')
    if not ent: raise ValueError("t0 t1 mandatory.")
    stime0, etime0 = np.array(ent.rstrip().split(' '), dtype=float)
    dummy = UTCDateTime(st[0].stats.starttime)
    stime = dummy + stime0
    etime = dummy + etime0
#------------ Relevant parameters --------------
    kwargs = dict(
# slowness grid : Xmin , Xmax , Ymin , Ymax , Slow Step
    sll_x =-3.0, slm_x =3.0, sll_y =-3.0, slm_y =3.0, sl_s =0.03,
# Changed for TTB 4/8/23
#    sll_x =-5.0, slm_x =5.0, sll_y =-5.0, slm_y =5.0, sl_s =0.025,
# sliding window properties
    win_len =1.0, win_frac =0.1,
# frequency properties
    frqlow =MTparam.get('Fmin'), frqhigh =MTparam.get('Fmax'), prewhiten =0,
# restrict output
    semb_thres=-1.e9, vel_thres=-1.e9 , timestamp='julsec',
    stime=stime , etime=etime
    )
#
    pline(['\n>> F-K parameters.'], width=13, align='^')
    pprint.pprint(kwargs)
#    pline(['\n>> F-K parameters.'], width=13, align='^')
#    pline(['    1) Slowness grid (km).'], width=13, align='^')
#    pline(['        y=['+str(kwargs.get(sll_y))+', '+str(kwargs.get(slm_y))+']'], width=6, align='^')
#    pline(['        x=['+str(kwargs.get(sll_x))+', '+str(kwargs.get(slm_x))+']'], width=6, align='^')
#    pline(['        Slow Step= '+str(kwargs.get(sl_s))], width=6, align='^')
#    pline(['    2) Sliding window.'], width=13, align='^')
#    pline(['        win_len= '+str(kwargs.get(win_len))+', win_frac'+str(kwargs.get(win_frac))], width=6, align='^')
#-- Geophone numbering was INVERTED as for Geode headers. Sort files by phone now.
    dummy = supl_data.get('phone_geo')
    dummy = dummy[np.argsort(dummy[:,0])]
    supl_data.update([('phone_geo', dummy)])
#                            └────> phone# (int), x, y (floats)
#
#------------- FK processing (obspy) --------------------
    for ind, tr in enumerate(st):
#-- Normalize
        tr.data = tr.data/max(tr.data)
#------------ Geophone positions in geographic
        tr.stats.coordinates = AttribDict({
                'latitude':  supl_data.get('phone_geo')[ind,0],
                'elevation': float(10),
                'longitude': supl_data.get('phone_geo')[ind,1]})
#------------ Execute array_processing
    out = array_processing(st, **kwargs)
#-- return
    return out, stime0, etime0
#
# -------------- End of function   ----------------------------
#
# -------------- rfreq  ----------------------------
def rfreq(arr, args):
    """
    Auxiliary function for flattening arrays and restricting their range
    arr  -> Original range of frequencies [ndarray [n,0]]
    args -> If arr is freq then len(args) = 2 and args[0] = Fmin and args[1] = Fmax
            Otherwise len(args) > 2 and args = Boolean [ndarray [n,0]] to restrict the range of arr.
    """
    rshx = lambda x: x.reshape(-1,) if x.ndim > 1 else x
#-- Verify if arr == None
    if arr is None: return arr, args
#------------ Restrict to frequencies f = [Fmin, Fmax]
    if len(args) == 2:
#-- Begin with freq
        arr = rshx(arr)
        bidx = (args[0] <= arr) & (arr <= args[1])
#               Fmin                        Fmax
        arr = arr[bidx]
#-- return relevant info
        return arr, bidx
#-- Deals with [ndarray [n,1]]
    elif np.shape(arr)[1] == 1:
        arr = rshx(arr)
        arr = arr[args]
#-- return relevant info
        return arr, args
#-- Deals with [ndarray [n,2]]
    elif np.shape(arr)[1] == 2:
        arr = arr[args, :]
#-- return relevant info
        return arr, args        
#-- Sanity
    else:
        raise ValueError("Wrong I/P in rfreq!")
#
# -------------- End of function   ----------------------------
#
# -------------- anF  ----------------------------
def anF(freq,F,pf,Fcrit=0.99):
    """
    F-test peak analysing script
    tr    -> trace-like [ndarray [nfft,0]]
    dt    -> Sampling interval in seconds of the time series
    F     -> F test values [ndarray [nfft,1]]
    pf    -> Probabilities of F values [ndarray [nfft,1]]
    Fcrit -> Crictical F-test probability
    """
#
# -------------- Find percentile F-test value corresponding to Fcrit
    Fthr = np.percentile(F, Fcrit*100.)
#    └────────────────────────│──────> F-test thrshold corresponding to Fcrit
#                             └──────> Fcrit in percentage, e.g., 99%
#-- Find indices of peaks in F values
    idx, _ = find_peaks(F, height=Fthr)
#-- Stack to 2-D
    Fpeak = np.stack((freq[idx], F[idx]), axis=-1)
#-- Find the 1st, 2nd and 3rd quartiles of F values
# Parameter for estimating the quantiles:
# (i)   discontinuous-> ‘inverted_cdf’, averaged_inverted_cdf’ and ‘closest_observation’
# (ii)  continuous   ->  ‘interpolated_inverted_cdf’, ‘hazen’, weibull’, ‘median_unbiased’ and ‘normal_unbiased’
# (iii) ‘linear’     -> (default) with ‘lower’, ‘higher’, ‘midpoint’ or ‘nearest’
# (iv)  The 75th percentile represents the value below which 75% of the data falls.
#-- Detect outliers of spectral lines > Fcrit
    Qlist = [0.75, 0.95, 0.99]
#    Fout = Fpeak[np.argwhere(Fpeak[:,1] > Qlist[0]),:].reshape(-1,2)
    Q = np.quantile(Fpeak[:,1], Qlist)                    #, method='linear'
    for ind, dummy in enumerate(Q[::-1]):                 # Reverse array
        Fout = Fpeak[np.argwhere(Fpeak[:,1] > dummy),:].reshape(-1,2)
        Qind = Qlist[ind]
        if len(Fout) >= 10:
            break
#-- Print figures of merit
    pline('\n>> There are ' + str(np.shape(Fpeak)[0]) + ' F-test >= ' + str(Fcrit*100.) +'%. Of those')
    pline('    ' + str(np.shape(Fout)[0]) + ' stand as outliers considering quantiles ' + str(Qind*100) + '%:')
    pline(['Frequency', 'F-test'], width=13, align='^')
    for dummy in np.around(Fout, decimals=3):
        pline([' ',*dummy], width=13, align='>')
#-- return relevant info
    return Fpeak, Fout
#
# -------------- End of function   ----------------------------
#
# ---------- Trace Pre-processing  ----------
def otrstr(tr, MTparam, supl_data, verbose='y'):
    """ 
    MTparam
     └─────> Fmin Fmax Ndec dtr lines taper iadapt nw kspec fcrit e_respec
         i =   0    1   2    3    4     5     6    7     8    9     10
                                            |<--- Multitaper only --->|
    """
#
#------------ Preprocessing
#-- Fix # of corners for filters
    nc = 4
#-- Copy of original trace
    tr0 = copy.deepcopy(tr)
#------------  Filter spectral lines
#-- 60Hz lines
    if np.isclose(MTparam.get('lines')[0], float(1)):
        try:
            supl_data.get('lines')                  #.flatten()
        except:
            raise ValueError("60Hz lines not in supl_data")
#
        dummy = supl_data.get('lines').ndim
        rfreq = supl_data.get('lines')[0] if dummy==1 else supl_data.get('lines')[:,0]
        delf  = supl_data.get('lines')[1] if dummy==1 else supl_data.get('lines')[:,1]
#-- Limits to fmax=Fmax
        dummy = rfreq <= MTparam.get('Fmax')
        rfreq = rfreq[dummy]
        delf  = delf[dummy]
#-- Filter spectral lines
        for ind, dummy in np.ndenumerate(rfreq):
            tr.filter(type='bandstop', freqmin=dummy-delf[ind] , freqmax=dummy+delf[ind],
                corners=nc, zerophase=True)
            tr0.filter(type='bandstop', freqmin=dummy-delf[ind] , freqmax=dummy+delf[ind],
                corners=nc, zerophase=True)
        if verbose=='y': pline(['    -> 60Hz lines filtered out: ',*rfreq], width=12, align='>')
#-- Other lines
    elif MTparam.get('lines')[0] > 1:
#        raise ValueError("Do not use decimation, needs to change stats.")
        rfreq = MTparam.get('lines')[0]
        delf  = MTparam.get('lines')[1]
        tr.filter(type='bandstop', freqmin=rfreq-delf, freqmax=rfreq+delf,
            corners=nc, zerophase=True)
        if verbose=='y': pline(['    -> Line filtered out: '+str(rfreq)+'+-'+str(delf)+'Hz'], width=12, align='>')
#
#------------ Decimate
    if MTparam.get('Ndec') != int(1):
        if len(tr) % MTparam.get('Ndec') == 0:
#-- Low-pass corner frequency with 40% margin
            dummy = 0.4 * tr.stats.sampling_rate / MTparam.get('Ndec')
#                    └────────────│───────────────────────│────── 40% margin
#                                 └───────────────────────|────── sampling_rate
#                                                         └────── decimation factor
            tr.filter('lowpass', freq=dummy, corners=nc, zerophase=True)
            tr.decimate(MTparam.get('Ndec'), no_filter=True, strict_length=True)
#-- Decimate original trace as well to agree with processed trace
            tr0.filter('lowpass', freq=dummy, corners=nc, zerophase=True)
            tr0.decimate(MTparam.get('Ndec'), no_filter=True, strict_length=True)
        else:
            raise NameError(">> Trace length does not divide by the decimation level.\n")
        if verbose=='y': pline('    -> Decimated series by factor ' + str(MTparam.get('Ndec')))
#
#------------ Detrend
    if MTparam.get('dtr') == int(1):
        tr.detrend("linear")
    elif MTparam.get('dtr') == int(2):
        tr.detrend("linear")
        tr.detrend("demean")
    elif MTparam.get('dtr') == int(3):
        tr.detrend("demean")
    else:
      raise ValueError("wrong detrend parameter.")
#
    dummy = ["linear", "linear+demean", "demean"]
    if verbose=='y': pline('    -> Series '+ dummy[MTparam.get('dtr')-1]+' detrended')
#
#------------ Taper trace
    if MTparam.get('taper') != int(0):
        dummy = (MTparam.get('taper') / 2) / 100.
        tr.taper(max_percentage =dummy, type='blackmanharris')
        if verbose=='y': pline('    -> Series 0.05 tapered with Blackmanharris')
#
#------------ Low/Band/High pass trace
    dummy = MTparam.get('Fmin') * MTparam.get('Fmax')
    if not np.isclose(dummy, float(0)):
        Fmin = MTparam.get('Fmin')
        Fmax = MTparam.get('Fmax')
        tr.filter(type='bandpass', freqmin=Fmin, freqmax=Fmax, corners=nc, zerophase=True)
#         └─> For the whole stream of traces in one go use "st" instead of "tr".
        if verbose=='y': pline('    -> Bandpass from '+str(Fmin)+' to '+str(Fmax)+'Hz.')
#
#------------ Gain: Use envelope or a multiplicative gain
    if not np.isclose(MTparam.get('gain'), float(0)):
#-- Stream objects have attribute 'traces'.
        if hasattr(tr, 'traces'):
            maxg = float('-inf')
            ming = float('inf')
            if np.isclose(MTparam.get('gain'), float(1)):           #-> envelope
                for ind in range(len(tr)):
                    gain = max(obspy.signal.filter.envelope(tr0[ind].data))
                    gain = np.round(gain/max(obspy.signal.filter.envelope(tr[ind].data)),0)
                    tr[ind].data *= gain
#
                    maxg = np.maximum(gain, maxg)
                    ming = np.minimum(gain, ming)
            else:                                                   #-> multiplicative
                    gain = MTparam.get('gain')
                    maxg = gain; ming = gain
                    for trace in tr: trace.data *= gain
#
            if verbose=='y': pline('    -> St gained from '+str(ming)+' to '+str(maxg)+'(envelope-derived).')
#-- Trace objects have attribute 'data'.
        elif hasattr(tr, 'data'):
            if np.isclose(MTparam.get('gain'), float(1)):           #-> envelope
                gain = max(obspy.signal.filter.envelope(tr0.data))
                gain = np.round(gain / max(obspy.signal.filter.envelope(tr.data)),0)
            else:                                                   #-> multiplicative
                gain = MTparam.get('gain')
#
            tr.data *= gain
            if verbose=='y': pline('    -> Trace gained with '+str(gain)+' (envelope-derived).')
#-- Sanity
        else:
            raise ValueError("Bad hasattr")
#
#-- return relevant info
    return tr, tr0
#
# -------------- End of function   ----------------------------
#
# ---------- Univariate Thomson multitaper estimates  ----------
def multcint(tr, dt, MTparam):
    """ Multitaper Spectrum Estimate with Confidence Intervals:
        ├── tr         -> numpy.ndarray Array with the data
        ├── dt         -> Sampling interval in seconds of the time series
        ├── nw         -> Time-bandwidth product. Usually 2, 3, 4 and numbers in between
        ├── kspec      -> # of tapers, defaults to int(2*time_bandwidth) - 1, which is the
        |                   practical maximum. More tapers will not be reliable and will
        |                   increase processing time. Fewer tapers -> faster calculation
        ├── iadapt     -> Method: 0 - adaptive multitaper; 1 - unweighted (wt =1 for all
        |                   tapers); 2 - wt by the eigenvalue of DPSS.
        ├── fcrit      -> Crictical F-test probability, e.g., 0.95
        └── e_respec   -> Reshape spectrum (time-consuming)
    """
#-- Unpack MT parameters
    (iadapt, nw, kspec, fcrit, e_respec) = MTparam
#      0     1     2      3       4
#
#------------ Multitaper Spectrum
    psd = mtspec.MTSpec(tr,nw,kspec,dt,iadapt=iadapt)
    pline('\n>> Multitaper class invoked')
    print(type(psd))
#-- Start timer
    tic = timeit.default_timer()
#
#------------ Performs the F test for a line component
    """
    Computes F-test for single spectral line components at
     the frequency bins given in the MTSpec class.
    O/P:
    F -> ndarray [nfft] vector of F test values
    p -> real ndarray [nfft] vector with probability of line component
    """
    F,p = psd.ftest()
#-- Get elapsed time
    toc = timeit.default_timer()
    pline(' > F-test done in '+ "%.2f" % (toc - tic) + " seconds")
    tic = timeit.default_timer()
#
#------------ Quadratic Spectrum
    """
    Function to calculate the Quadratic Spectrum. The first 2 derivatives of
     the spectrum are estimated and the bias associated with curvature (2nd derivative)
     is reduced. Calculate the Stationary Inverse Theory Spectrum.
    O/P:
    qispec -> The quadratic spectrum estimate [ndarray [nfft,0]]
    ds     -> The estimate of the 1st derivative [ndarray [nfft,0]]
    dds    -> The estimate of the 2nd derivative [ndarray [nfft,0]]
    """
    qispec  = psd.qiinv()[0]
#-- Get elapsed time
    toc = timeit.default_timer()
    pline(' > Quadratic Spectrum done in '+ "%.2f" % (toc - tic) + " seconds")
    tic = timeit.default_timer()
#
#------------ Spectrum's confidence intervals
    """
    Estimate adaptively weighted jackknifed 95% confidence limits
    -> CHECK IF THIS IS VALID FOR QUADRATIC SPECTRUM!!!
    O/P:
    spci -> The real array of jackknife error estimates, with 
             confidence intervals for the spectrum.
    """
    spci = psd.jackspec()
#-- Get elapsed time
    toc = timeit.default_timer()
    pline(' > Confidence inter. done in '+ "%.2f" % (toc - tic) + " seconds")
    tic = timeit.default_timer()
#
#------------ Reshape spectrum (time-consuming)
    """
    Reshape eigenfunctions (yk) around significant spectral lines. Significant
     means above crictical F probability, fcrit. Hints: fcrit= 0.9, 0.95.
    O/P:
    respec -> The reshaped PSD estimate [ndarray [nfft]]
    specnl -> The PSD without the line components [ndarray [nfft]]
    yk     -> The eigenftions without the line components [ndarray [nfft,kspec]]
    sline  -> The PSD of the line components only [ndarray [nfft]]
    """
    if e_respec == 'y':
        respec, specnl, yk, sline = psd.reshape(fcrit=fcrit,p=p)
#-- Get elapsed time
        toc = timeit.default_timer()
        pline(' > Reshape done in '+ "%.2f" % (toc - tic) + " seconds")
        tic = timeit.default_timer()
#
#------------ Spectra at positive frequencies: Fmin <= freq & freq <=  Fmax
    """
    rspec(*args) -> The spectra at positive frequencies, assuring a real input signal was used.
    O/P:
    args -> Array to return the positive frequencies, e.g., qispec, spci ... [ndarray]
    """
    freq,spec        = psd.rspec()
    freq,qispec,spci = psd.rspec(qispec,spci)
    F                = F[0:psd.nf]
    p                = p[0:psd.nf]
    if e_respec == 'y':
        freq,respec,specnl = psd.rspec(respec,specnl)
    else:
        respec           = None
        specnl           = None
#
#-- return relevant info
    return freq,spec,F,p,respec,specnl,qispec,spci
#
# -------------- End of function   ----------------------------
#
# -------------- Preprocess data  ----------------------------
def preproc(tr, MTpar, *args, action = None):
  """ Trace preprocessing:
      ├── action = 'rgain'   -> Remove instrument gain with a de-scaling factor.
      ├── action = 'lines'   -> Filter 60Hz spectral lines.
      |                          args[0] = [[l1, df1], ...] an array of frequency lines
      |                                    to be notched with (up to Nyquist).
      |                          args[1] = integer filter's number of corners      
      ├── action = 'decimate' -> Decimate by N
      |                          args[0] = N
      ├── action = 'detrend'  -> Detrend based on args[0]
      |                          args[0] = 1-> Detrend "linear"
      |                          args[0] = 2-> Detrend "linear" and demean      
      |                          args[0] = 3-> Demean only
      └── action = 'bfilt'    -> Bandpass filter
                                 args[0] = Fmin
                                 args[1] = Fmax
                                 args[2] = integer filter's number of corners.
  """
#------------ Remove Geode gain with a de-scaling factor.
#             Return a de-gained trace. Do not use this correction,
#              it is here as a reminder ONLY.
# Call -> tr_p = preproc(tr, 'rgain', 0)
  if action == 'rgain':
    raise ValueError("Do not use rgain.")
#-- Fixed gain:  Gain(dB)   Factor
    dscale = [ [0 ,    2.6974e-3],
               [12,    6.7536e-4],
               [24,    1.6985e-4],
               [36,    4.2704e-5] ]
#    dummy = int(tr.stats.seg2.FIXED_GAIN.split()[0])
    dummy = trHd['trGain']
    op = [op for op in range(len(dscale)) if int(dscale[op][0]) == dummy]
#-- Correction factor
    op = float(dscale[op[0]][1])
    tr = tr * op
#------------ Return trace
    return tr
#------------------- Filter spectral lines -------------------------
  if action == 'lines':
#------------ Bandstop filter trace's spectral lines
#-> MUST BE DONE IN THE ORIGINAL DATA SERIES.
#-- Some aliases
    dummy = np.asarray(args[0]).reshape((int(len(args[0])/2), -1))
    rfreq = dummy[:,0]
    delf  = dummy[:,1]
#-- Limits to fmax=Fmax
    dummy = rfreq <= MTpar['Fmax']
    rfreq = rfreq[dummy]
    delf  = delf[dummy]
#-- Filter spectral lines
    for ind, dummy in np.ndenumerate(rfreq):
      tr.filter(type='bandstop', freqmin=dummy-delf[ind] , freqmax=dummy+delf[ind],
                corners=args[1], zerophase=True)
#
    pline(['>> 60Hz lines filtered out: ',*rfreq], width=12, align='>')
#------------ Return trace
    return tr
#
#------------------------ Decimate ------------------------
#------------ Decimate (downsample) trace data by integer factor plist[1].
#-- Apply lowpass filter to ensure no aliasing artifacts (no_filter=False)
#    and keep sanity assuring len(tr) divides by plist[0] (strict_length=True).
  if action == 'decimate':
    if len(tr) % args[0] == 0:
      dummy = [len(tr),tr.stats.sampling_rate]
      tr.decimate(args[0], no_filter=False, strict_length=True)
      dummy = dummy.extend([len(tr),tr.stats.sampling_rate])
    else:
      raise NameError(">> Trace length does not divide by the decimation level.\n")
#
    pline('>> Decimated series by factor ' + str(args[0]))
#------------ Return trace
    return tr
#
#------------------------ Detrend data ------------------------
#
  if action == 'detrend':
    if args[0]   == 1:
      tr.detrend("linear")
    elif args[0] == 2:
      tr.detrend("linear")
      tr.detrend("demean")
    elif args[0] == 3:
      tr.detrend("demean")
    else:
      raise ValueError("wrong parameter in preproc/detrend.")
#
    dummy = ["linear", "linear+demean", "demean"]
    pline('>> Series '+ dummy[args[0]-1])
#------------ Return trace
    return tr
#
#------------------------ Bandpass data series ------------------------
#
# Call -> tr_p = preproc(tr, 'bfilt', Fmin, Fmax, nc, 0)
  if action == 'bfilt':
#-- Splits integer plist[3] into its digits to produce fband parameters,
#    returns a view w/ ravel(). plist[3]=1 defaults to plist[3]=[1,2,4].
#    fmin , smar, nc = splint(plist[3], 3, [1,2,4])
#------------ Bandpass filter trace with a zero-phase-shift, achieved through
#              two runs forward and backward. In this way if original filter has
#              nc corners, we end up with 2*nc corners de facto. nc==corners.
    tr.filter(type='bandpass', freqmin=args[0], freqmax=args[1], corners=args[2], zerophase=True)
#    └─> For the whole stream of traces in one go use "st" instead of "tr".
#
    dummy = [args[0],args[1],args[2]]
    pline('>> Bandpass fmin, fmax, #corners:')
    pline(['->', *dummy], width=9, align='^')   #*np.array([dummy])]
#------------ Return trace
    return tr
#
# -------------- End of function   ----------------------------
#
# -------------- Polar to Cartezian --------------
def Pol_Car(x, y, coord='polar', degrees=True):
  """
  Program to convert Polar <-> Cartezian
   x = abscissa if coord != 'polar' or the radial coordinate r if coord == 'polar'
   y = ordinate if coord != 'polar' or the angular coordinate (polar angle) if coord == 'polar'
   if degrees = True convert to radians
  """
  if coord == 'polar':
    y = np.radians(y) if degrees else y
    dummy =  x.copy()
    x = dummy *np.sin(y)
    y = dummy *np.cos(y)
  elif coord == 'cartesian':
    dummy = x.copy()
    x = np.sqrt(dummy**2 + y**2)
    y = np.atan2(y, dummy)
  else:
    raise NameError("Wrong entry in Pol_Car.\n")
  return x, y
#
# -------------- End of function   ----------------------------
#
# -------------- pline  ----------------------------
"""
Print stuff along a line
line -> A string or a list of strings or one string followed by real numbers.
        The string can be a blank
"""
def pline(line, width=None, align='^'):
  if isinstance(line, str):
    width = len(line) if width == None else width
    print ('{:{align}{width}}'.format(line, align=align, width=width))
  elif all(isinstance(dummy, str) for dummy in line):
    n = len(line)
    f = ['{:{align}{width}} '] * n
    print(''.join(f).format(*line, align=align, width=width))
  else:
    n = len(line)
    f = ['{:^'+str(len(line[0]))+'}']
    dummy =  line[1:]
    if dummy == float:
      f.extend(['{:{align}{width}.1f} '] * (n - 1))
    else:
      f.extend(['{:{align}{width}}'] * (n - 1))
    print(''.join(f).format(line[0], *dummy, align=align, width=width))
# -------------- End of function   ---------------------
#
# -------------- bar  ----------------------------
"""
Displays progress with a bar on terminal.
A) Parameters
count   -> count / (expected total of counts)
blen=50 -> bar length
unit='%'-> count unit
msn=''  -> msn to add at the end
B) Use as
total = 100; i = 0
while i <= total:
    bar1(i/total, msn='job')
    i += 1
sys.stdout.write('\n')

"""
def bar(count, blen=50, unit='%', msn=''):
    filled_len = int(round(blen * count))
    percents = round(100.0 * count, 1)
    percents = int(percents)
    bar = '=' * filled_len + '-' * (blen - filled_len)
#
    sys.stdout.write('[%s] %s%s  %s\r' % (bar, percents, unit, msn))  #'%'
    sys.stdout.flush()
#
    return
#
# -------------- End of function   ---------------------

