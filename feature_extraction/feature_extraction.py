import numpy as np
import pandas as pd
import math
import scipy.stats as stats
from scipy.interpolate import interp1d
from statsmodels.tsa import stattools
from astropy.stats import LombScargle

ID = 'object_id'
target = 'target'

def amplitude(data):
    N = len(data)
    sorted_data = np.sort(data)
    return (np.median(sorted_data[-int(math.ceil(0.05 * N)):]) - np.median(sorted_data[0:int(math.ceil(0.05 * N))])) / 2.0

def rcs_max(data):
    sigma = np.std(data)
    N = len(data)
    m = np.mean(data)
    s = np.cumsum(data - m) * 1.0 / (N * sigma)
    return np.max(s)

def rcs_min(data):
    sigma = np.std(data)
    N = len(data)
    m = np.mean(data)
    s = np.cumsum(data - m) * 1.0 / (N * sigma)
    return np.min(s)

def Autocor_length(data):
    if len(data) == 1:
        return np.nan
    nlags = 5
    AC = stattools.acf(data, nlags=nlags)
    k = next((index for index, value in enumerate(AC) if value < np.exp(-1)), None)
    while k is None:
        nlags += 5
        AC = stattools.acf(data, nlags=nlags)
        k = next((index for index, value in enumerate(AC) if value < np.exp(-1)), None)
    return k

def SmallKurtosis(data):
    n = len(data)
    if n <= 3 :
        return np.nan
    mean = np.mean(data)
    std = np.std(data)
    S = sum(((data - mean) / std) ** 4)
    c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return c1 * S - c2

def con(data):
    data = data.values
    N = len(data)
    if N < 3:
        return np.nan
    sigma = np.std(data)
    m = np.mean(data)
    count = 0
    for i in range(N - 3 + 1):
        flag = 0
        for j in range(3):
            if(data[i + j] > m + 2 * sigma or data[i + j] < m - 2 * sigma):
                flag = 1
            else:
                flag = 0
                break
        if flag:
            count += 1
    return count * 1.0 / (N - 3 + 1)

def MedianAbsDev(data):
    median = np.median(data)
    devs = (abs(data - median))
    return np.median(devs)

def MeanAbsDev(data):
    median = np.median(data)
    devs = (abs(data - median))
    return np.mean(devs)

def MedianBRP(data):
    if len(data) < 3:
        return np.nan
    magnitude = data
    median = np.median(magnitude)
    amplitude = (np.max(magnitude) - np.min(magnitude)) / 10
    n = len(magnitude)
    count = np.sum(np.logical_and(magnitude < median + amplitude, magnitude > median - amplitude))
    return float(count) / n

def PercentAmplitude(data):
    if len(data) < 3:
        return np.nan
    magnitude = data
    median_data = np.median(magnitude)
    distance_median = np.abs(magnitude - median_data)
    max_distance = np.max(distance_median)
    percent_amplitude = max_distance / median_data
    return percent_amplitude

def Q31(data):
    if len(data) < 3:
        return np.nan
    return np.percentile(data, 75) - np.percentile(data, 25)

def AndersonDarling(data):
    if len(data) < 3:
        return np.nan
    ander = stats.anderson(data)[0]
    return 1 / (1.0 + np.exp(-10 * (ander - 0.3)))

def LinearTrend(group):
    if group.shape[0] <= 3:
        return np.nan
    magnitude = group['flux'].values
    time = group['mjd'].values
    regression_slope = stats.linregress(time, magnitude)[0]
    return regression_slope

def MaxSlope(group):
    if group.shape[0] <= 3:
        return np.nan
    magnitude = group['flux'].values
    time = group['mjd'].values
    slope = np.abs(magnitude[1:] - magnitude[:-1]) / (time[1:] - time[:-1])
    return np.max(slope)

def StetsonK(group):
    if group.shape[0] == 1:
        return np.nan
    magnitude = group['flux'].values
    error = group['flux_err'].values
    mean_mag = np.sum(magnitude/(error*error)) / np.sum(1.0/(error * error))
    N = len(magnitude)
    sigmap = np.sqrt(N * 1.0 / (N - 1)) * (magnitude - mean_mag) / error
    K = 1 / np.sqrt(N * 1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2))
    return K

def slotted_autocorrelation(data, time, T, K, second_round=False, K1=100):
    slots = np.zeros((K, 1))
    i = 1
    time = time - np.min(time)
    m = np.mean(data)
    data = data - m
    prod = np.zeros((K, 1))
    pairs = np.subtract.outer(time, time)
    pairs[np.tril_indices_from(pairs)] = 10000000
    ks = np.int64(np.floor(np.abs(pairs) / T + 0.5))

    idx = np.where(ks == 0)
    prod[0] = ((sum(data ** 2) + sum(data[idx[0]] * data[idx[1]])) / (len(idx[0]) + len(data)))
    slots[0] = 0

    if second_round is False:
        for k in np.arange(1, K):
            idx = np.where(ks == k)
            if len(idx[0]) != 0:
                prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                slots[i] = k
                i = i + 1
            else:
                prod[k] = np.infty
    else:
        for k in np.arange(K1, K):
            idx = np.where(ks == k)
            if len(idx[0]) != 0:
                prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                slots[i - 1] = k
                i = i + 1
            else:
                prod[k] = np.infty
        np.trim_zeros(prod, trim='b')

    slots = np.trim_zeros(slots, trim='b')
    return prod / prod[0], np.int64(slots).flatten()

def SlottedA_length(group):
    if group.shape[0] < 5:
        return np.nan
    magnitude = group['flux'].values
    time = group['mjd'].values
    N = len(time)
    deltaT = time[1:] - time[:-1]
    sorted_deltaT = np.sort(deltaT)
    T = sorted_deltaT[int(N * 0.05)+1]
    K = 100
    [SAC, slots] = slotted_autocorrelation(magnitude, time, T, K)
    SAC2 = SAC[slots]
    k = next((index for index, value in enumerate(SAC2) if value < np.exp(-1)), None)
    while k is None:
        K = K+K
        if K > (np.max(time) - np.min(time)) / T:
            break
        else:
            [SAC, slots] = slotted_autocorrelation(magnitude, time, T, K, second_round=True, K1=K//2)
            SAC2 = SAC[slots]
            k = next((index for index, value in enumerate(SAC2) if value < np.exp(-1)), None)
    if k is None or len(list(slots)) == 0:
        return np.nan
    return slots[k] * T

def StetsonK_AC(group):
    if group.shape[0] < 5:
        return np.nan
    magnitude = group['flux'].values
    time = group['mjd'].values
    N = len(time)
    deltaT = time[1:] - time[:-1]
    sorted_deltaT = np.sort(deltaT)
    T = sorted_deltaT[int(N * 0.05)+1]
    K = 100
    [SAC, slots] = slotted_autocorrelation(magnitude, time, T, K)
    autocor_vector = SAC[slots]
    N_autocor = len(autocor_vector)
    sigmap = (np.sqrt(N_autocor * 1.0 / (N_autocor - 1)) * (autocor_vector - np.mean(autocor_vector)) / np.std(autocor_vector))
    K = (1 / np.sqrt(N_autocor * 1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))
    return K

def Beyond1Std(group):
    if group.shape[0] <= 3:
        return np.nan
    magnitude = group['flux'].values
    error = group['flux_err'].values
    n = len(magnitude)
    weighted_mean = np.average(magnitude, weights=1 / error ** 2)
    var = sum((magnitude - weighted_mean) ** 2)
    std = np.sqrt((1.0 / (n - 1)) * var)
    count = np.sum(np.logical_or(magnitude > weighted_mean + std,
                                 magnitude < weighted_mean - std))
    return float(count) / n

def Beyond2Std(group):
    if group.shape[0] <= 3:
        return np.nan
    magnitude = group['flux'].values
    error = group['flux_err'].values
    n = len(magnitude)
    weighted_mean = np.average(magnitude, weights=1 / error ** 2)
    var = sum((magnitude - weighted_mean) ** 2)
    std = np.sqrt((1.0 / (n - 1)) * var)
    count = np.sum(np.logical_or(magnitude > weighted_mean + 2 * std,
                                 magnitude < weighted_mean - 2 * std))
    return float(count) / n

def Eta_e(group):
    if group.shape[0] <= 3:
        return np.nan
    magnitude = group['flux'].values
    time = group['mjd'].values
    w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
    w_mean = np.mean(w)
    N = len(time)
    sigma2 = np.var(magnitude)
    S1 = sum(w * (magnitude[1:] - magnitude[:-1]) ** 2)
    S2 = sum(w)
    eta_e = (w_mean * np.power(time[N - 1] - time[0], 2) * S1 / (sigma2 * S2 * N ** 2))
    return eta_e

def StructureFunction_index(group):
    if group.shape[0] <= 3:
        return {'index_21': np.nan,
                'index_31': np.nan,
                'index_32': np.nan,
                'index_21_31_ratio': np.nan,
                'index_21_32_ratio': np.nan,
                'index_31_32_ratio': np.nan,
               }
    magnitude = group['flux'].values
    time = group['mjd'].values

    Nsf = 100
    Np = 100
    sf1 = np.zeros(Nsf)
    sf2 = np.zeros(Nsf)
    sf3 = np.zeros(Nsf)
    f = interp1d(time, magnitude)

    time_int = np.linspace(np.min(time), np.max(time), Np)
    mag_int = f(time_int)

    for tau in np.arange(1, Nsf):
        sf1[tau-1] = np.mean(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 1.0))
        sf2[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 2.0)))
        sf3[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 3.0)))
    sf1_log = np.log10(np.trim_zeros(sf1))
    sf2_log = np.log10(np.trim_zeros(sf2))
    sf3_log = np.log10(np.trim_zeros(sf3))

    m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
    m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
    m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)

    return {'index_21': m_21,
            'index_31': m_31,
            'index_32': m_32,
            'index_21_31_ratio': m_21 / m_31,
            'index_21_32_ratio': m_21 / m_32,
            'index_31_32_ratio': m_31 / m_32,
            }

def DetectedMjdDiff(group):
    mjd = group[group['detected'] == 1]['mjd'].values
    if len(mjd) == 0:
        return np.nan
    return np.max(mjd) - np.min(mjd)

def FreqExtract(group, bins=20):
    k = bins
    if group.shape[0] <= 5:
        return_dict = {('f0'+str(f) if f < 10 else 'f'+str(f)): (np.nan) for f in range(k)}
        return_dict['Nyquist_freq'] = np.nan
        return return_dict
    
    time = group['mjd'].values
    time -= time[0]
    time *= 24
    freq, power = LombScargle(time, group['flux']).autopower(nyquist_factor=1)
	#freq, power = LombScargle(time, group['flux'], group['flux_err']).autopower(nyquist_factor=1)
    freq /= (0.00188 / k)
    buff = pd.DataFrame({'freq': freq.astype(int), 'power': power})
    buff.loc[buff['freq'] >= k, 'freq'] = k - 1
    buff = buff.groupby('freq')['power'].sum()
    return_dict = {('f0'+str(f) if f < 10 else 'f'+str(f)): (buff.loc[f] if f in buff.index else np.nan) for f in range(k)}
    return_dict['Nyquist_freq'] = freq.max()
    return return_dict

def get_aggregations():
    return {
        'mjd': ['size'],
        'flux': ['min', 'max', 'mean', 'skew',
                 amplitude, rcs_max, rcs_min, Autocor_length, SmallKurtosis, con, MedianAbsDev, MeanAbsDev, MedianBRP,
                 PercentAmplitude, Q31, AndersonDarling],
        'flux_err': ['max', 'mean'],
        'detected': ['mean'],
        'flux_0_error_2': ['sum'],
        'flux_1_error_2': ['sum'],
        'flux_2_error_2': ['sum'],
        'flux_3_error_2': ['sum'],
    }
	
def aggregation(df, groupby_columns):
    df['flux_0_error_2'] = np.power(1 / df['flux_err'], 2.0)
    df['flux_1_error_2'] = df['flux'] * df['flux_0_error_2']
    df['flux_2_error_2'] = df['flux'] * df['flux_1_error_2']
    df['flux_3_error_2'] = df['flux'] * df['flux_2_error_2']
    aggs = get_aggregations()
    df_agg = df.groupby(groupby_columns).agg(aggs)
    df_agg.columns = ['_'.join(col).strip('_ ') for col in df_agg.columns.values]
    df_agg = df_agg.reset_index().rename(columns={'mjd_size': 'observations'})
    
    df_agg['flux_weighted_mean_10'] = df_agg['flux_1_error_2_sum'] / df_agg['flux_0_error_2_sum']
    df_agg['flux_weighted_mean_32'] = df_agg['flux_3_error_2_sum'] / df_agg['flux_2_error_2_sum']
    df_agg['flux_max_min_ratio'] = (-1) * df_agg['flux_max'] / df_agg['flux_min']
    df_agg['flux_amplitude_diff_ratio'] = df_agg['flux_amplitude'] / (df_agg['flux_max'] - df_agg['flux_min'])
    df_agg['flux_rcs'] = df_agg['flux_rcs_max'] - df_agg['flux_rcs_min']
    df_agg['flux_amplitude_mean_ratio'] = (df_agg['flux_amplitude']) / df_agg['flux_mean']
    df_agg['flux_amplitude_weighted_mean_10_ratio'] = (df_agg['flux_amplitude']) / df_agg['flux_weighted_mean_10']
    df_agg['flux_amplitude_weighted_mean_32_ratio'] = (df_agg['flux_amplitude']) / df_agg['flux_weighted_mean_32']

    # multi-column functions
    df_agg['StetsonK'] = df.groupby(groupby_columns).apply(StetsonK).values
    df_agg['SlottedA_length'] = df.groupby(groupby_columns).apply(SlottedA_length).values
    df_agg['StetsonK_AC'] = df.groupby(groupby_columns).apply(StetsonK_AC).values
    df_agg['Beyond1Std'] = df.groupby(groupby_columns).apply(Beyond1Std).values
    df_agg['Beyond2Std'] = df.groupby(groupby_columns).apply(Beyond2Std).values
    df_agg['MaxSlope'] = df.groupby(groupby_columns).apply(MaxSlope).values
    df_agg['LinearTrend'] = df.groupby(groupby_columns).apply(LinearTrend).values
    df_agg['Eta_e'] = df.groupby(groupby_columns).apply(Eta_e).values
    df_agg['DetectedMjdDiff'] = df.groupby(groupby_columns).apply(DetectedMjdDiff).values
    
    df_agg = pd.concat([df_agg, pd.DataFrame(list(df.groupby(groupby_columns).apply(StructureFunction_index)))], axis=1)
    df_agg = pd.concat([df_agg, pd.DataFrame(list(df.groupby(groupby_columns).apply(FreqExtract)))], axis=1)
    
    # np.log
    df_agg['flux_2_error_2_sum'] = np.log(df_agg['flux_2_error_2_sum'] + 1)
    
    return df_agg
	
def feature_extraction(df_meta, df):
    passband_agg = aggregation(df, [ID, 'passband'])
    passband_agg = passband_agg.merge(df_meta[[ID, target]], on=ID, how='left')
    object_agg = aggregation(df, ID)
    object_agg = df_meta.merge(object_agg, on=ID, how='left')
    
    return passband_agg, object_agg

if __name__ == '__main__':
    train_meta = pd.read_csv('../input/training_set_metadata.csv')
    train = pd.read_csv('../input/training_set.csv')
    train_passband_agg, train_object_agg = feature_extraction(train_meta, train)
    train_passband_agg.to_csv('train_passband_aggregation.csv', index=False)
    train_object_agg.to_csv('train_object_aggregation.csv', index=False)

    test_meta = pd.read_csv('../input/test_set_metadata.csv')
    test = pd.read_csv('../input/test_set.csv')
    test_passband_agg, test_object_agg = feature_extraction(test_meta, test)
    test_passband_agg.to_csv('test_passband_aggregation.csv', index=False)
    test_object_agg.to_csv('test_object_aggregation.csv', index=False)
