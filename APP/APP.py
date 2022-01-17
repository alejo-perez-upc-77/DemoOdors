"""Main module of Data acquisition
Inlet opening and sample pulling
Filtering and ploting
"""

# LIST OF IMPORTS
import time
from APP_Utilities import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pylsl import StreamInlet, resolve_byprop
from audiomix3 import AudioMix
import pygame


# JSON CONFIG FILE HANDLING
app_dir = os.path.dirname(__file__)
json_path = os.path.join(app_dir, 'Config_APP.json')
with open(json_path, 'r') as js:
    Config_APP = json.load(js)

image_path = os.path.join(app_dir, 'BackgroundVA14.png')
image = cv2.imread(image_path)
plt.ion()

#  PYGAME WALKAROUND SETUP FOR MUSIC OUTPUT
display = pygame.display.set_mode((800, 600))
# IMPORT HYSTERESIS PARAMETER
hist = Config_APP["Input_Variables"][0]["Song_Params"]["Hysteresis"]
# AUDIOMIX INSTANCE 
audio = AudioMix(0, 0, hist)



##############################################################
######################  Inlet Opening ########################
##############################################################
Outlet_Name_Features = Config_APP["Input_Variables"][0]["Outlet_Name_LSL_Parser"] + "-Features"
Outlet_Name_Markers = Config_APP["Input_Variables"][0]["Outlet_Name_LSL_Parser"] + "-Markers"
######################## FEATURES ############################

print('Connecting to LSL Streaming (Features)')
result = resolve_byprop('name', Outlet_Name_Features, 1, 0.1)
if result == []:
    print('Impossible to Connect to LSL')
    time.sleep(5);
else:
    print('Succesfully Connected to LSL')

streamFeatures = result[0]
inletFeatures = StreamInlet(streamFeatures)
print('Connected to outlet ' + streamFeatures.name() + '@' + streamFeatures.hostname())
time.sleep(0.5)
inletFeatures.open_stream()
time.sleep(0.5)

######################### MARKERS ############################

print('Connecting to LSL Streaming (Markers)')
time.sleep(1)

result = resolve_byprop('name', Outlet_Name_Markers, 1, 0.1)

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

rescale_epoch = Config_APP["Input_Variables"][0]["Rate_Parameters"]["td2"]  # Time interval to scale the epoch of
# features in stage 2

    ####################################################################################
    #########################       BIG LOOP OPENING       #############################
    ####################################################################################
while True:
    ####################################################################################
    #######################       Creation of variables       ##########################
    ####################################################################################
    Feature_Bunch, _ = inletFeatures.pull_sample()  # Pulls a sample to know the number of features
    _, n_features = feature_chunk_parser(Feature_Bunch)  # Catches  the number of features

    Buffer_features = np.empty([n_features, 0])  # Creates an array with 0 columns and N_features rows
    Buffer_timestamp_features = np.empty([0, 0])  # Timestamp array alongside features

    Buffer_features_2 = np.empty([n_features, 0])  # Creates an array with 0 columns and N_features rows (second stage)
    Buffer_timestamp_features_2 = np.empty([0, 0])  # Timestamp array alongside features to be used in stage 2

    Feature_1, Feature_2 = fusion_parameters(Config_APP['Input_Variables'][0]["Feature_Fusion"]["Standard_Param"])

    # TIME CORRECTION OF 2 INLETS
    inletFeatures.time_correction(1)
    inletMarkers.time_correction(1)

    ####################################################################################
    #############################       STAGE 1       ##################################
    ####################################################################################
    # LOOP OF TRIGGER WAITING
    print('Waiting for Trigger 1')
    while True:  # Pulls Triggers until having a 1
        Trigger, timestamp_trigger = inletMarkers.pull_chunk()
        if Trigger == []:
            continue
        else:
            if 1.0 in Trigger[0]:
                print('Trigger One, beginning of baseline calculation')
                break
            else:
                pass

    # Sets a timeout of T seconds to finish stage 1
    timeout_1 = time.time() + Config_APP["Input_Variables"][0]["Rate_Parameters"]["td"]

    # BUFFER CLEANING
    inletFeatures.pull_chunk()
    inletMarkers.pull_chunk()

    # BEGINNING OF STAGE 1
    while True:

        Trigger, timestamp_trigger = inletMarkers.pull_chunk()  # Pulls Trigger
        Feature_Bunch, timestamp_features = inletFeatures.pull_sample()  # Pulls a sample of Features
        features_parsed = feature_chunk_parser(Feature_Bunch)[0]  # Parsers Features
        Buffer_features = np.append(Buffer_features, features_parsed, axis=1)  # Pushes features into the Buffer
        Buffer_timestamp_features = np.append(Buffer_timestamp_features, timestamp_features +
                                              inletFeatures.time_correction())
        # Same with timestamps
        if time.time() > timeout_1:  # When time is achieved, exits the loop
            print("end of stage 1")
            break

    # BASELINE COMPUTATION, MEAN FOR EVERY FEATURE IN BUFFER
    Baseline = np.array([np.nanmean(feature) for feature in Buffer_features]).reshape(n_features, 1)
    print(Baseline)

    ####################################################################################
    #############################       STAGE 2       ##################################
    ####################################################################################

    # WAITING LOOP FOR STAGE 2
    print('\nWaiting for Trigger 2')
    while True:  # Pulls Triggers until having a 2
        Trigger, timestamp_trigger = inletMarkers.pull_chunk()
        if Trigger == []:
            continue
        else:
            if 2.0 in Trigger[0]:
                print('Trigger 2, beginning experiment')
                break
            else:
                pass

    # Sets a timeout of T seconds to finish stage 2
    timeout_2 = time.time() + Config_APP["Input_Variables"][0]["Rate_Parameters"]["td2"]
    # Sets a timestamp to subtract the other samples to have Timestamps from the beginning
    t0_stage_2 = inletFeatures.pull_chunk()[1][-1] + inletFeatures.time_correction()
    inletMarkers.pull_chunk()  # Clear BUFFER

    while True:

        Trigger, timestamp_trigger = inletMarkers.pull_chunk()  # Pulls Trigger
        Feature_Bunch, timestamp_features = inletFeatures.pull_sample()  # Pulls a sample of Features
        features_parsed = feature_chunk_parser(Feature_Bunch)[0]/Baseline  # Parsers Features and divides by Baseline
        Buffer_features_2 = np.append(Buffer_features_2, features_parsed, axis=1)  # Pushes features into the Buffer
        Buffer_timestamp_features_2 = np.append(Buffer_timestamp_features_2, timestamp_features - t0_stage_2 +
                                                inletFeatures.time_correction())
        # Pushes feature timestamps into Buffer, correcting time and subtracting t0
        if time.time() > timeout_2:
            print('end of the stage 2')
            print(Buffer_features_2.shape)
            break  # Finishes stage 2 when timeout is reached

    Buffer_features_2_rescaled = np.array([-1 + 2*((row-np.nanmin(row))/(np.nanmax(row)-np.nanmin(row)))
                                           for row in Buffer_features_2])  # Rescales the whole Buffer
                                                       # Between [-1,+1]
    print(Buffer_features_2_rescaled)
    # Feature Fusion Calculation
    feature_fusion_1, feature_fusion_2 = feature_fusioner(Feature_1, Feature_2, Buffer_features_2_rescaled, True)
    feature_fusion_buffer = np.array([feature_fusion_1, feature_fusion_1])

    ######################### PLOT #############################################
    fig, ax = plt.subplots(figsize=(5, 5))

    plt.axis('off')
    plt.xlim([0.0, 2.0])

    fig.set_facecolor('black')
    ####################################################################################
    #############################       STAGE 3       ##################################
    ####################################################################################
    print("begin of stage 3")
    timeout_3 = time.time() + Config_APP["Input_Variables"][0]["Rate_Parameters"]["td3"]
    while True:

        Trigger, timestamp_trigger = inletMarkers.pull_chunk()  # Pulls Trigger
        Feature_Bunch, timestamp_features = inletFeatures.pull_sample()  # Pulls a sample of Features
        features_parsed = feature_chunk_parser(Feature_Bunch)[0]/Baseline  # Parsers Features and divides by Baseline
        # Pushes features into the Buffer
        Buffer_features_2_rescaled = np.append(Buffer_features_2_rescaled, features_parsed, axis=1)
        Buffer_timestamp_features_2 = np.append(Buffer_timestamp_features_2, timestamp_features - t0_stage_2 +
                                                inletFeatures.time_correction())
        # Pushes feature timestamps into Buffer, correcting time and subtracting t0

        Buffer_features_2_rescaled = rescale_last_epoch(Buffer_features_2_rescaled, Buffer_timestamp_features_2,
                                                        rescale_epoch)

        # Feature Fusion Calculation
        feature_fusion_1, feature_fusion_2 = feature_fusioner(Feature_1, Feature_2, Buffer_features_2_rescaled, False)
        new_fusion = np.array([feature_fusion_1, feature_fusion_2]).reshape(len(feature_fusion_buffer), 1)
        feature_fusion_buffer = np.append(feature_fusion_buffer, new_fusion, axis=1)
        feature_fusion_buffer = rescale_last_epoch(feature_fusion_buffer, Buffer_timestamp_features_2, rescale_epoch)

        # RESCALE feature_fusion_buffer

        print(feature_fusion_buffer[0, -1], feature_fusion_buffer[1, -1])
        ax.clear()  # axis clearing to update
        ax.imshow(image, extent=[-1.414, 1.414, -1.414, 1.414])
        ax.plot(feature_fusion_buffer[0, -1], feature_fusion_buffer[1, -1], 'o', linewidth=2, color='yellow')
        plt.pause(0.001)
        audio.update(feature_fusion_buffer[0, -1],  feature_fusion_buffer[1, -1])  # Calls the update algorithm to
        # switch songs (or not)
        if time.time() > timeout_3:  # When the timeout is reached, it exits the loop
            print('end of stage 3')
            print(Buffer_features_2_rescaled.shape)
            print(feature_fusion_buffer.shape)
            break
            plt.close('all')
        if Trigger == []:
            continue
        else:
            if 3.0 in Trigger[0]:  # When a marker 3 arrives, it exits the loop
                print('Trigger 3, end of stage 3')
                plt.close('all')

                break
            else:
                pass
    audio.stop()  # at the end of the experiment, the app music is shut down
    plt.close()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
            exit()
    pygame.display.update()
pygame.quit()
