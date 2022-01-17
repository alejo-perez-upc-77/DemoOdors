import numpy as np
import re
import scipy
from py_stareeglab.data_preprocessing.filters import butter_bandpass

def feature_parser(features):
    '''
    :param features: list of strings configured in the input_data.json with an specific format
    :return: Dictionary where key,value = feature name,feature value
    '''
    # Assign a Regular expression to parser the feature calculation orders looking for the specific patterns
    regex = r"(\w*)\((.*?)\)([+/*-])(\w*)\((.*?)\)|(\w*)\((.*?)\)"
    feature_parser_dict = [] #empty list to store dictionaries of feature parameters parsered
    for feature in features: #iterate through the feature strings list
        features_parsed = {}
        groups = re.findall(regex, feature.replace(" ", "")) #chatches every element of the feature order in each element
        lst = [x for x in groups[0] if x != ""] #delete the blank elements
        if len(lst) == 2: # if there is just one band, assigns to the dictionary following key value pairs
            features_parsed['band'] = lst[0]
            features_parsed['channels'] = list(map(lambda x: int(x) - 1, lst[1].split(',')))
        else: # if there are two bands assigns to the dict following key value pairs
            features_parsed['band_1'] = lst[0]
            features_parsed['channels_1'] = list(map(lambda x: int(x) - 1, lst[1].split(',')))
            features_parsed['operation'] = lst[2]
            features_parsed['band_2'] = lst[3]
            features_parsed['channels_2'] = list(map(lambda x: int(x) - 1, lst[4].split(',')))

        feature_parser_dict.append(features_parsed)
    return feature_parser_dict #eventually delivers a list of parsed feature computing instruccions


def feature_computer(band_power_dict, feature_parser_dict): #TODO: Parser the channels string in the previous function
    '''
    :param band_power_dict: dictionary whose keys are band names while values are arrays of N element where N = Number of channel
    :param feature_parser_dict: features_parsed resulted by the output of "feature_parser" func
    :return:
    '''
    feature_computed = {} #empty dictionary to store the computed features
    for feature in feature_parser_dict:
        if len(feature) == 2:  # if 1 band
            name = feature['band']  # name of band is assigned
            channel_compu = feature['channels'] #create as list of channels splitting
            band_mean = np.array([band_power_dict[name][ch] for ch in channel_compu]).mean() #compute the mean of channels of the band locating the coeff in band_power_dict
            feature_computed[feature['band'] + '_mean(' + ','.join(list(map(lambda x: str(x+1),channel_compu))) + ')'] = band_mean #assign a dictionary key value pair the mean
        else:
            channel_compu_1 = feature['channels_1']  # create as list of channels splitting of 1st band
            channel_compu_2 = feature['channels_2'] # create as list of channels splitting for the 2nd band
            band_1 = feature['band_1']
            band_2 = feature['band_2']
            band_mean_1 = np.array([band_power_dict[band_1][ch] for ch in channel_compu_1]).mean()#same process as before but with 2 bands
            band_mean_2 = np.array([band_power_dict[band_2][ch] for ch in channel_compu_2]).mean()
            string_computed = (feature['band_1'] + '_mean(' + ','.join(list(map(lambda x: str(x+1),channel_compu_1))) + ')' +
                               feature['operation'] + feature['band_2'] + '_mean(' +
                               ','.join(list(map(lambda x: str(x+1),
                               channel_compu_2))) + ')')  # creates a string key for dictionary with info of the feature
            # decides according to operation introduced which operator to use
            if feature['operation'] == '*':
                feature_computed[string_computed] = band_mean_1 * band_mean_2
            elif feature['operation'] == '/':
                feature_computed[string_computed] = band_mean_1 / band_mean_2
            elif feature['operation'] == '+':
                feature_computed[string_computed] = band_mean_1 + band_mean_2
            elif feature['operation'] == '-':
                feature_computed[string_computed] = band_mean_1 - band_mean_2

    return feature_computed

def channel_wrapper(feature_parser_dict):
    '''
    :param feature_parser_dict: output of "feature_parser" function, list of dictionaries with name of bands,
    channels and operations as keys
    :return: feature_channels: list of lists, channels which intervene in every feature
    '''
    channel_complex = []
    feature_channels = []

    for row in feature_parser_dict:  # Iterates the input dictionary storing the channels differentiating whether there
        # are 2 bands or one, appends the channels of every feature
        if 'channels' in row.keys():
            channel_complex.append(row['channels'])  # stores channel
        if 'channels_1' in row.keys():
            channel_complex.append((row['channels_1'], (row['channels_2'])))  # stores channels of both bands in a tuple

    for feat_ch in channel_complex: # iterates the anterior list of lists and tuples and appends the channels for each
        # feature
        if type(feat_ch) is list:
            feature_channels.append(list(dict.fromkeys([ch for ch in feat_ch])))  # appends channels in a list deleting
            # duplicated ones
        else:  # appends channels of both bands in a list deleting duplicated ones
            feature_channels.append(list(dict.fromkeys([ch for tuple_ in feat_ch for ch in tuple_])))
    return feature_channels  # delivers a list of lists with channels of each feature


def filter_iir_fir(fs_EEG, F_Low, F_High, Filter_Order, F_type,  data, timestamp, iir_transient=4):
    """
    :param fs_EEG: EEG Sample rate of the device
    :param F_Low: Cutoff frequency 1 for the Bandpass Filter
    :param F_High: Cutoff frequency 2 for the Bandpass Filter
    :param Filter_Order:
    :param F_type: FIR or IIR
    :param data: EEG Epoch
    :param timestamp: timestamp of the EEG Epoch
    :param iir_transient: seconds to subtract from the EEG Buffer
    :return:
    EEG_FIr: EEG Epoch filtered and inverted
    time_stamp_EEG_Filt: EEG timestamp Epoch without transient inverted
    """
    # Compute W
    fc1N = F_Low / (fs_EEG / 2.0)
    fc2N = F_High / (fs_EEG / 2.0)  # create the normalized frecuencies
    if F_type == 'FIR':
        # Calculate b coefficient of the FIR filter
        FIR = scipy.signal.firwin(Filter_Order, cutoff=[fc1N, fc2N], window="blackmanharris",
                                  pass_zero=False)  # creates the filter
        # Filter at sample level from left to right
        EEG_FIr = scipy.signal.lfilter(FIR, 1, data, axis=1, zi=None)
        # Remove transient from EEG
        EEG_FIr = EEG_FIr[:, Filter_Order:]
        # Remove transient from time stamp
        time_stamp_EEG_Filt = timestamp[Filter_Order:]
        # Flip Buffer EEG and TimeStamp
        EEG_FIr = np.flip(EEG_FIr, 1)
        time_stamp_EEG_Filt = np.flip(time_stamp_EEG_Filt)
        # Correct delay introduced by filter (fs_EEG/2 + 1 (secs))
        time_stamp_EEG_Filt = time_stamp_EEG_Filt + ((Filter_Order / 2 + 1) * 1 / fs_EEG)

        return EEG_FIr, time_stamp_EEG_Filt

    if F_type == 'IIR':  # Check: normalised frequencies? # Ask if Transient is correct
        seconds = int(iir_transient * fs_EEG)
        # Calculate b coefficient of the IIR filter
        b, a = butter_bandpass(F_Low, F_High, fs_EEG, Filter_Order)
        # Filter at sample level from left to right
        EEG_IIR = scipy.signal.lfilter(b, a, data)
        # Remove 4 seconds from the EEG_IIR as transient (orientative)
        EEG_IIR = EEG_IIR[:, seconds:]
        # Same with timestamp
        time_stamp_EEG_Filt = timestamp[seconds:]
        # Flip Buffer EEG and TimeStamp
        EEG_IIR = np.flip(EEG_IIR, 1)
        time_stamp_EEG_Filt = np.flip(time_stamp_EEG_Filt)

        return EEG_IIR, time_stamp_EEG_Filt


def standard_feature_set(feature_set):
    """
    Function that returns an standard set of features in a list of strings to be delivered to feature_parser.
    If a list of strings is passed it returns the list. Otherwise, if the keyword "standard_8" or "standard_20" is
    passed, it returns a predefined set.

    :param feature_set: keyword ra
    :return:
    """
    if type(feature_set) == list:
        return feature_set

    elif feature_set == "standard_8":

        features = ["Gamma(5)-Gamma(4)", "Gamma(5,7) – Gamma(4,6)", "alpha(3)-alpha(1)",
                    "Beta1(1,3)/alpha(1,3)", "Theta(1,2,3)/alpha(1,2,3)"]

        return features

    elif feature_set == "standard_20":

        features = ["Gamma(9) - Gamma(8)", "Gamma(9,14) - Gamma(8,10)", "alpha(3) - alpha(1)",
                    "alpha(3) - alpha(1)", "Beta1(1,3) / alpha(1,3)", "Gamma(11,12,13,15,16,17,18,19)",
                    "Theta(1,2,3) / alpha(1,2,3)"]

        return features

def ch_reference_variable_creator(channel_reference, n_channels_EEG):
    """
    String, list, or integer to be parsed and passed to the function sdc_channel_reference afterwards.
    :param channel_reference: List of channels for the rest to be referenced
    :param n_channels_EEG: number of EEG channels obtained from the device metadata
    :return:
    """
    if type(channel_reference) is list:
        output = list(map(lambda x: x - 1, channel_reference))
    elif channel_reference == "average":
        output = list(range(0, n_channels_EEG))
    elif type(channel_reference) is int:
        output = [channel_reference - 1]

    return output
