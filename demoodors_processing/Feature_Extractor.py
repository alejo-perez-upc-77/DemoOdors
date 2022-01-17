"""
Main module of Data acquisition
Inlet opening and sample pulling
Filtering and ploting
"""
import json
import threading
from statistics import mean, stdev, median
import time
from Plot_Signals import plot_signals_acc
from Extractor_Utilites import *
import scipy
import matplotlib.pyplot as plt
import numpy as np
from pylsl import StreamInlet, resolve_byprop, StreamOutlet, StreamInfo
from py_stareeglab.data_preprocessing import data_processing
from py_stareeglab.feature_extraction import spectral_features
import os

script_dir = os.path.dirname(__file__)
json_path = os.path.join(script_dir, 'Config.json')
with open(json_path, 'r') as js:
    Config = json.load(js)

############# Time Variables ####################
t_rate = []  # Set list to append tmies of epochs processing
timeout = Config['Input_Variables'][0]['Debug_Rate']  # seconds between printing performances
debug_plot = Config['Input_Variables'][0]['Debug_Plot']  # seconds between printing performances


def clear_performance_show():
    """
    performance assessment function activated in a specific lapse of time (variable timeout). This function does not take
    into account the matplotlib plotting stages
    :return: assessment information
    """
    threading.Timer(timeout, clear_performance_show).start()  # Set the interval of time between the calling of this function
    global t_rate  # include the list t_rate
    if t_rate == []: pass
    else:
        print('\nRate time mean:', mean(t_rate),  # compute metrics
              '\nRate time std:', stdev(t_rate),
              '\nRate time median:', median(t_rate),
              '\nMax interval:', max(t_rate),
              '\nnºEpochs:', len(t_rate))
        t_rate.clear()  # clear the list no to store too much data


############### IMPORT VARIABLE OF POWERBANDS AND FEATURE TO COMPUTE ###################
band = Config['Input_Variables'][0]['Power_Bands']
features = standard_feature_set(Config['Input_Variables'][0]['Feature_Calculation'])


############## CREATION OF OUTLET NIC NAME PARAMETERS ########################
outlet_EEG_name = Config['Input_Variables'][0]["Outlet_Name_NIC"]+'-EEG'
outlet_Accelerometer_name = Config['Input_Variables'][0]["Outlet_Name_NIC"]+'-Accelerometer'
outlet_Markers_name = Config['Input_Variables'][0]["Outlet_Name_NIC"]+'-Markers'


################ CREATION OF FILTERING PARAMETERS ###########################
Filter_type = Config['Input_Variables'][0]["Preprocessing_Parameters"]["Filter_Type"]
IIR_Transient = Config['Input_Variables'][0]["Preprocessing_Parameters"]["IIR_Transient"]
F_Low = Config['Input_Variables'][0]["Preprocessing_Parameters"]["Cutoff_Frequency_Low"]
F_High = Config['Input_Variables'][0]["Preprocessing_Parameters"]["Cutoff_Frequency_High"]
Filter_Order = Config['Input_Variables'][0]["Preprocessing_Parameters"]["Filter_Order"]


Artefact_Threshold = Config['Input_Variables'][0]["Preprocessing_Parameters"]["Artefact_Threshold"]

################ FEATURE ORDERS PARSER ###########################
feature_parser_dict = feature_parser(features)
feature_channels = channel_wrapper(feature_parser_dict)

##############################################################################
############################# CREATION OF INLETS #############################
##############################################################################


########################## EEG AND fs_EEG ###############################

print('Connecting to LSL Streaming (EEG)')
time.sleep(1)
result = resolve_byprop('name', outlet_EEG_name, 1, 0.1)
if result == []:
    print('Impossible to Connect to LSL')
    time.sleep(5);
else:
    print('Succesfully Connected to LSL')

streamEEG = result[0]
n_channels_EEG = streamEEG.channel_count()
inletEEG = StreamInlet(streamEEG)
print('Connected to outlet ' + streamEEG.name() + '@' + streamEEG.hostname())
time.sleep(0.5)
inletEEG.open_stream()
time.sleep(0.5)
fs_EEG = streamEEG.nominal_srate()  # Catches the sample frequency


############## CREATION OF BUFFER PARAMETERS EEG ########################

Shift_Len_EEG = int(Config['Input_Variables'][0]["Rate_Parameters"]["Shift_Len"]*fs_EEG)
# to obtain samples from seconds
if Filter_type == "FIR":
    Buffer_Len_EEG = int(Config['Input_Variables'][0]["Rate_Parameters"]["Buffer_Len"]*fs_EEG + Filter_Order)
# to obtain samples from seconds
elif Filter_type == "IIR":
    Buffer_Len_EEG = int(Config['Input_Variables'][0]["Rate_Parameters"]["Buffer_Len"]*fs_EEG + IIR_Transient*fs_EEG)
# to obtain samples from seconds
######################### MARKERS ############################

print('Connecting to LSL Streaming (Markers)')
time.sleep(1)

result = resolve_byprop('name', outlet_Markers_name, 1, 0.1)

if result == []:
    print('Impossible to Connect to LSL')
    time.sleep(5);
else:
    print('Succesfully Connected to LSL')

streamMarkers = result[0]
inletMarkers = StreamInlet(streamMarkers)
print('Connected to outlet ' + streamMarkers.name() + '@' + streamMarkers.hostname())
time.sleep(0.5)
inletMarkers.open_stream()
time.sleep(0.5)


################## ACCELEROMETER AND Buffer_Len_Accel#######################

print('Connecting to LSL Streaming (Accelerometer)')
time.sleep(1)

result = resolve_byprop('name', outlet_Accelerometer_name, 1, 0.1)

if result == []:
    print('Impossible to Connect to LSL')
    time.sleep(5);
else:
    print('Succesfully Connected to LSL')

stream_Accelerometer = result[0]
n_channels_Accelerometer = stream_Accelerometer.channel_count()
inletAccelerometer = StreamInlet(stream_Accelerometer)
print('Connected to outlet ' + stream_Accelerometer.name() + '@' + stream_Accelerometer.hostname())
time.sleep(0.5)
inletAccelerometer.open_stream()
time.sleep(0.5)
fs_Accel = stream_Accelerometer.nominal_srate()

Buffer_Len_Accel = int(Config['Input_Variables'][0]["Rate_Parameters"]["Buffer_Len"]*fs_Accel)

###################### Channel Reference variable creation ##########################
Channel_Reference = ch_reference_variable_creator(
    Config['Input_Variables'][0]["Preprocessing_Parameters"]["Channel_Reference"], n_channels_EEG)

############################################################################################################
############################### OUTLET FOR STREAMING FEATURES AND TRIGGERS #################################
############################################################################################################
Outlet_Name_Features = Config['Input_Variables'][0]["Outlet_Push"]["Outlet_Name"] + '-Features'
Outlet_Name_Markers = Config['Input_Variables'][0]["Outlet_Push"]["Outlet_Name"] + '-Markers'
Source_ID = Config['Input_Variables'][0]["Outlet_Push"]["Outlet_Source_ID"]
Info_Outlet_Features = StreamInfo(Outlet_Name_Features, 'EEG', len(feature_parser_dict)*2, 0, 'float32',
                                  source_id=Source_ID)

Info_Outlet_Markers = StreamInfo(Outlet_Name_Markers, 'Markers', 2, 0, 'float32', source_id=Source_ID)
# next make an outlet
Outlet_Features = StreamOutlet(Info_Outlet_Features)
Outlet_Markers = StreamOutlet(Info_Outlet_Markers)

############################################################################################################
#################################### CREATION OF BUFFER AND SHIFTS #########################################
############################################################################################################

############################################ EEG ###########################################################
Buffer_EEG = np.zeros([n_channels_EEG, Buffer_Len_EEG])
Shift_EEG = np.zeros([n_channels_EEG, Shift_Len_EEG])
time_stamp_shift_EEG = np.zeros([Shift_Len_EEG])
time_stamp_buffer_EEG = np.zeros([Buffer_Len_EEG])

############################################ ACCELEROMETER ##################################################
Buffer_Accelerometer = np.zeros([n_channels_Accelerometer, Buffer_Len_Accel])
time_stamp_buffer_Accelerometer = np.zeros([Buffer_Len_Accel])
##########
executions = 0

############################################################################################################
#################################### BUFFER CLEANING #######################################################
############################################################################################################
chunk_clearer_EEG = inletEEG.pull_chunk(0.1)  # Pulls Data Chunk to Clear Buffer
chunk_clearer_Accelerometer = inletAccelerometer.pull_chunk(0.1)  # Pulls Data Chunk to Clear Buffer
chunk_clearer_Markers = inletMarkers.pull_chunk(0.1)  # Pulls Data Chunk to Clear Buffer
############################################################################################################
#################################### First Time Correction #################################################
############################################################################################################
inletEEG.time_correction(1)
inletAccelerometer.time_correction(1)
inletMarkers.time_correction(1)

# Creation of t0 immediately before of entering the main loop
t0_EEG = inletEEG.pull_sample()[-1] + inletEEG.time_correction()
t0_Acc = inletAccelerometer.pull_sample()[-1] + inletAccelerometer.time_correction()

if debug_plot == 1:
    plt.ion()
    f, (ax1, ax2, ax3) = plt.subplots(3, 1)  # Creation of 3 figures for plotting EEG Accelerometer and Features
clear_performance_show()  # Calls function to begin the performance assessment

while True:
    # store time to compute interval of data obtention + processing
    # Let's pull samples to fill Shift
    count = 0
    # at the beginning of the following while loop we are replacing upon the last Shift (at the right)!
    while count < Shift_Len_EEG:
        sample_EEG, time_sample_EEG = inletEEG.pull_sample(0.1)  # Pulls sample from the EEG stream
        time_sample_EEG = time_sample_EEG + inletEEG.time_correction()  # applies time correction
        Shift_EEG[:, count] = np.array([channel for channel in sample_EEG])[:, ]
        # New Sample -8 Channels- pushes on shift
        time_stamp_shift_EEG[count] = time_sample_EEG  # New time_stamp sample pushes on the left
        ######################################################
        count += 1
    ######################################################
    ##############Let's push the Buffer####################
    ######################################################
    #########################
    Buffer_EEG[:, :-Shift_Len_EEG] = Buffer_EEG[:, Shift_Len_EEG:]
    # pushes the buffer towards the right to make space at the left
    Buffer_EEG[:, -Shift_Len_EEG:] = Shift_EEG  # push the shift at the right
    time_stamp_buffer_EEG[:-Shift_Len_EEG] = time_stamp_buffer_EEG[Shift_Len_EEG:]
    # push the time_stamp_buffer at the right to make space at the left
    time_stamp_buffer_EEG[-Shift_Len_EEG:] = time_stamp_shift_EEG  # push the time_shift at the right
    #######################################################
    ######################################################
    chunk_Markers, time_stamp_Markers = inletMarkers.pull_chunk(0.1)  # Pulls sample from the EEG stream
    if (chunk_Markers != []) & (time_stamp_Markers != []):  # check if the array is empty or not
        chunk_markers_array = [i[0] for i in chunk_Markers]  # Not empty --> store in array as well as timestamp
        timestamp_markers_array = [i+inletMarkers.time_correction() for i in time_stamp_Markers]  # time correction within the timestamp
        print(chunk_markers_array)
        print(timestamp_markers_array)
        Triggers = chunk_markers_array
    else:  # if markers are empty store in variable Triggers an empty list to be sent
        Triggers = []
    ######################################################
    ################# ACCELEROMETER BUFFER ###############
    ######################################################
    chunk_Accelerometer, time_chunk_Accelerometer = inletAccelerometer.pull_chunk()
    # Pulls a chunk from the EEG stream to get samples from seconds
    Shift_Accelerometer = np.array([[x[0] for x in chunk_Accelerometer],
                                    [y[1] for y in chunk_Accelerometer],
                                    [z[2] for z in chunk_Accelerometer]])
    # Parses it into an array of three channels (rows)
    time_stamp_shift_Accelerometer = np.array(time_chunk_Accelerometer) + inletAccelerometer.time_correction()
    # Corrects the timestamps array
    Shift_Len_Accel = Shift_Accelerometer.shape[1]  # defines a variable with the num of samples of chunk
    if Shift_Len_Accel > Buffer_Len_Accel:  # if it is larger than the buffer
        Shift_Len_Accel = Buffer_Len_Accel  # The new Shift len will be the Buffer len
        Shift_Accelerometer = Shift_Accelerometer[:, -Shift_Len_Accel:]
        # Keep only the last samples (buffer dimensions)
        time_stamp_shift_Accelerometer = time_stamp_shift_Accelerometer[-Shift_Len_Accel:]  # Same with timestamps
    else:
          pass

    Buffer_Accelerometer[:, :-Shift_Len_Accel] = Buffer_Accelerometer[:, Shift_Len_Accel:]
    # push the buffer at the left to make space at the right
    Buffer_Accelerometer[:, -Shift_Len_Accel:] = Shift_Accelerometer  # push the shift at the right
    time_stamp_buffer_Accelerometer[:-Shift_Len_Accel] = time_stamp_buffer_Accelerometer[Shift_Len_Accel:]
    # push the time_stamp_buffer to the left to make space at the right
    time_stamp_buffer_Accelerometer[-Shift_Len_Accel:] = time_stamp_shift_Accelerometer
    # push the time_shift towards the right
    Buffer_Accelerometer_Flip = np.flip(Buffer_Accelerometer, 1)  # Flips the Accelerometer numpy array
    time_stamp_buffer_Accelerometer_Flip = np.flip(time_stamp_buffer_Accelerometer)  # Flips the Accelerometer timestamp
    #################################################
    ini_t_rate = time.time()
    ################################################
    ## FILTERING

    EEG_Filt, time_stamp_EEG_Filt = filter_iir_fir(fs_EEG, F_Low, F_High, Filter_Order, Filter_type,
                                                  Buffer_EEG, time_stamp_buffer_EEG, IIR_Transient)

    # *******************************
    # This method from starEEGlab python toolkit is not working and only delivered nans, that's why we need
    # a work around to filter the data.
    #EEG_Fir , b = filters.fir_filter(Buffer_EEG.transpose(), fs_EEG, F_Low, F_High, Filter_Order)
    # ******************************

    # CHANNEL REFERENCE
    EEG_FiltR = data_processing.sdc_channel_reference(EEG_Filt, np.array(Channel_Reference))
    # Buffer is referenced to a channel introduced in array

    # DETREND
    EEG_FiltRD = scipy.signal.detrend(EEG_FiltR,)  # signal is detrended linearly

    # ARTIFACTS
    Artefact_Flags = np.array([np.any(ch >= Artefact_Threshold) for ch in np.absolute(EEG_FiltRD)])
    # Retrieves an array of N elements (N channels) with booleans
    # if there is a sample above the threshold predefined sets the element to True

    # BAND POWER OBTENTION
    band_power = spectral_features.star_band_psd(np.expand_dims(EEG_FiltRD, 2), fs_EEG,
                                                 band, normalized_band_flag=False)
    # passes epoch to a function to obtain spectral bandpowers
    band_power = band_power[:, :, 0]  # squeezes array into 2 dimensions
    channels = [band_power[:, column] for column in range(band_power.shape[1])]  # obtains lists of all value associated
    # to every channel for each band
    band_names = [i['Name'] for i in band] # obtains the names of bands
    band_power_dict = {}
    for name, ch in zip(band_names, channels):
        band_power_dict[name] = ch  # Creates a dictionary with each band name and its channels

    feature_computed = feature_computer(band_power_dict, feature_parser_dict)  # Delivers a dictionary with feature-value pairs
    ######################################################
    ######################################################

    #################### BOOLEAN FLAGS EXTRACTION ############################
    boolean_feature_flags = np.zeros(len(feature_computed), dtype=bool)  # Creates a 1-dim array with length equal to the
    # number of features with False booleans
    for idx, feature_ch in enumerate(feature_channels):  # Iterates the the list of lists feature_channels
        for ch in feature_ch:  # Iterates the list of channels of every feature
            if Artefact_Flags[ch]:  # if the channel in question is True in the Artefact_Flags vector
                boolean_feature_flags[idx] = True  # sets True the index position = index of feature  in boolean_feature_flags vector
    ###################################################################
    ###################################################################
    fin_t_rate = time.time()  # store time to compute interval
    t_rate.append(fin_t_rate - ini_t_rate)  # Interval time of the epoch processed

# FEATURE STREAMING
    boolean_feature_flags = list(boolean_feature_flags)
    # converts both boolean features and computed feature values in lists
    feature_values = list(feature_computed.values())
    features_push = [feature_values, boolean_feature_flags]  # builds a list with booleans and values
    Outlet_Features.push_chunk(features_push)  # push the feature-boolean list
    Outlet_Markers.push_chunk(Triggers)  # push Triggers
    ###################################################################
    ###################################################################
# plot debug
    if debug_plot == 1:
        plot_signals_acc(ax1, ax2, ax3, time_stamp_EEG_Filt, EEG_FiltRD, time_stamp_buffer_Accelerometer_Flip,
                         Buffer_Accelerometer_Flip, feature_computed, t0_EEG, t0_Acc)
# calls the function to plot the real-time streaming with accelerometer


    executions += 1
    print('Execution: ', executions)

    # Print Time intervals of stages in multiple of 20 execution


if __name__ == '__main__':
    print('End of the programme')












