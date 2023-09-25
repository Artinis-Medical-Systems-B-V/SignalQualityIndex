import numpy as np
from scipy.signal import kaiserord, firwin, filtfilt, detrend, correlate


def SQI(OD1, OD2, oxy, dxy, Fs):
    """
    This function reads in NIRS data and returns a scalar value representing the signal quality score.
    Use as score = SQI(OD1,OD2,oxy,dxy,Fs), where:
    :param OD1: - 1xN vectors containing optical density signals from higher wavelength. N is the number of samples.
    :param OD2: - 1xN vectors containing optical density signals from lower wavelength. N is the number of samples.
    :param oxy: - 1xN vector containing O2Hb signal (concentration changes in oxygenated hemoglobin)
    :param dxy: - 1xN vector containing HHb signal (concentration changes in deoxygenated hemoglobin)
    :param Fs:  - scalar, sampling rate in Hz
    :return:    - scalar, signal quality score ranging from 1 (very low quality) to 5 (very high quality)

    All input arguments are necessary.
    The performance of this function has been tested on 10-second interval signal segments recorded with Artinis devices
     (OxyMon, OctaMon, Brite23, Brite24)

    This script makes use of the following functions in the mentioned libraries:
    kaiserord, firwin, filtfilt, detrend, correlate, windows - from scipy.signal Python library
    std, log, sum, abs, concatenate, zeros - from numpy Python library

    Version 1.0, copyright (c) by Artinis Medical Systems http://www.artinis.com.
    Last modified on 31-01-2023
    Author: Naser Hakimi (naser@artinis.com)

    Cite as: M. Sofía Sappia, Naser Hakimi, Willy N. J. M. Colier, and Jörn M. Horschig, "Signal quality index:
    an algorithm for quantitative assessment of functional near infrared spectroscopy signal quality," Biomed. Opt.
    Express 11, 6732-6754 (2020)

    This project received funding from the European Union’s Horizon 2020 Framework Program under grant agreements
    No. 813234 (RHUMBO) and No. 813843 (INFANS). This study was also funded by the
    European Fund for Regional Development (EFRO) and the Dutch provinces Gelderland and Overijssel (PROJ-00872).

    Creative Commons Licence
    Signal Quality Index: an algorithm for quantitative assessment of functional near infrared spectroscopy signal
    quality by M. Sofía SAPPIA, Naser HAKIMI, Willy N.J.M. COLIER, Jörn M. HORSCHIG is licensed under a Creative
    Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

    Based on a work at https://github.com/Artinis-Medical-Systems-B-V/SignalQualityIndex.
    Permissions beyond the scope of this license may be available at science@artinis.com.
    -----------------------------------------------------------------------------------------------
   """

    # ### Setting algorithm parameters
    # Thresholds for features in rating stages one and two
    thrUp_intensity = 2.5
    thrLow_intensity = 0.04
    thr_sumHbratio = 0.67
    thr_acorrDiffODs = 40
    # Slope and intercept for score conversion in rating stage three
    slope = 1.795613343002295
    intercept = 0.846108994828045

    # ##################################################################################################################
    # ######################## RATING STAGE ONE: Identifying very low quality signals ##################################
    # ##################################################################################################################
    # First feature in rating stage one: counts are outside of linear range
    if np.any(OD1 < thrLow_intensity) or np.any(OD1 > thrUp_intensity) or np.any(OD2 < thrLow_intensity) or \
            np.any(OD2 > thrUp_intensity):
        SQIscore = 1
        return SQIscore

    # Second feature in rating stage one: at least one of the optical density signals is a flat line
    if (np.std(OD1) == 0) or (np.std(OD2) == 0):
        SQIscore = 1
        return SQIscore

    # ########################################### Filtering the signals ################################################
    # ##################################################################################################################
    # Band-pass filtering, with Kaiser window
    OD1_filt = fir_filter(detrend(OD1), Fs, cutoff=[0.4, 3], type='bandpass')
    OD2_filt = fir_filter(detrend(OD2), Fs, cutoff=[0.4, 3], type='bandpass')
    oxy_filt = fir_filter(detrend(oxy), Fs, cutoff=[0.4, 3], type='bandpass')
    dxy_filt = fir_filter(detrend(dxy), Fs, cutoff=[0.4, 3], type='bandpass')
    # ##################################################################################################################
    # ##################################################################################################################
    # Third feature in rating stage one: Different Scale of Oxy and Deoxy
    if np.log(np.sum(np.abs(oxy_filt)) / np.sum(np.abs(dxy_filt))) < thr_sumHbratio:
        SQIscore = 1
        return SQIscore
    # ##################################################################################################################

    # ##################################################################################################################
    # ######################## RATING STAGE TWO: Identifying very high quality signals #################################
    # ##################################################################################################################
    # Feature in rating stage two: The difference between the auto-correlation signals of filtered OD signals should be
    # low
    autocorr_od1_filt = correlate(OD1_filt, OD1_filt) / np.sum(OD1_filt**2)
    autocorr_od2_filt = correlate(OD2_filt, OD2_filt) / np.sum(OD2_filt ** 2)
    if (1 / np.std(autocorr_od1_filt - autocorr_od2_filt)) > thr_acorrDiffODs:
        SQIscore = 5
        return SQIscore
    # ##################################################################################################################

    # ##################################################################################################################
    # ############################## RATING STAGE THREE: Signal quality rating #########################################
    # ##################################################################################################################
    # Feature in rating stage three: the standard deviation of O2Hb is higher than the standard deviation of HHb for
    # high quality NIRS signals
    logStdHb = np.log(np.std(oxy_filt) / np.std(dxy_filt))
    SQIscore = logStdHb * slope + intercept

    # Forcing the score to be within 1 (very low quality) and 5 (very high quality)
    if SQIscore < 1:
        SQIscore = 1
    elif SQIscore > 5:
        SQIscore = 5

    return SQIscore
    # ##################################################################################################################
    # ##################################################################################################################
    # ##################################################################################################################
    # ##################################################################################################################

def fir_filter(signal, Fs, cutoff, type='bandpass', width=0.2, ripple=65):
    numtaps, beta = kaiserord(ripple, width / (0.5 * Fs))
    if numtaps > len(signal) / 3.5:
        numtaps = int(len(signal) / 3.5)
    numtaps |= 1  # Set the lowest bit to 1, making the numtaps odd
    filter_coefs_peak = firwin(numtaps=numtaps, cutoff=cutoff, window=('kaiser', beta), pass_zero=type, fs=Fs)
    return filtfilt(filter_coefs_peak, 1, signal)
