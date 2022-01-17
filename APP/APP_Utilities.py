import numpy as np
import os
import json

# JSON CONFIG HANDLING
app_dir = os.path.dirname(__file__)
json_path = os.path.join(app_dir, 'Config_APP.json')
with open(json_path, 'r') as js:
    Config_APP = json.load(js)

def feature_chunk_parser(feature_chunk):
    """
    Parsers the feature chunk received from the module one, returns an array for the features and the number of features
    received
    :param feature_chunk: Sample pulled by LSL. It contains N Features and N Booleans
    :return:
        features_parsed: Array in 1 column of feature values and nans (where has been an artefact)
        chunk_len_half: Number of features = len(feature_chunk)/2
    """
    chunk_len_half = int(len(feature_chunk)/2)
    feature_values = np.array(feature_chunk[:chunk_len_half])  # splits list by the half
    boolean_flags = np.array(feature_chunk[chunk_len_half:], dtype=float)
    boolean_flags[boolean_flags == 1] = np.nan  # converts 1 in nan
    boolean_flags[boolean_flags == 0] = 1  # converts 0 in 1
    features_parsed = feature_values * boolean_flags  # multiplies 2 arrays to have a nan where there was an artefact
                                                      # at the beginning
    features_parsed = features_parsed.reshape(chunk_len_half, 1)  # Put the vector
    return features_parsed, chunk_len_half


def feature_fusioner(feature_1, feature_2, Buffer_features_2_rescaled, entire_flag):  # implement 4 features
    """
    :param feature_1: string with the normalized syntax, e.g., "1+3+4". It has to be an addition or subtraction
    with either 3 or 2 features
    :param feature_2: string with the normalized syntax, e.g., "1+3+4". It has to be an addition or subtraction
    with either 3 or 2 features
    :param Buffer_features_2_rescaled: Matrix with which features will be computer
    :param entire_flag: If True the feature fusion output will be arrays with the same length as the
    Buffer_features_2_rescaled. If False,
    just a value computed with the last samples of Buffer_features_2_rescaled
    :return: Either a list with 2 values or a list or 2 arrays depending on the entire_flag argument
    """
    Buffer_features_2_rescaled = Buffer_features_2_rescaled
    feature_fusion_list = []
    for feature_ in [feature_1, feature_2]:
        feature_ = feature_.replace(" ", "")  # Delete Blank spaces
        feature_ = [x for x in feature_]  # Separates strings
        feature = []
        for x in feature_:  # loop to subtract 1 from each integer number
            try:
                x = int(x) - 1
                x = str(x)
                feature.append(x)
            except:
                feature.append(x)
        if len(feature) == 5:  # Case  of 3 features first positive
            if entire_flag:  # computes with all the matrix
                feature_fusion_string = ("(Buffer_features_2_rescaled[" + feature[0] + "]" + feature[1] +
                                         "Buffer_features_2_rescaled[" + feature[2] + "]" +
                                         feature[3] + "Buffer_features_2_rescaled[" + feature[4] + "])")
                feature_fusion_value = eval(feature_fusion_string) / 3
                # uses eval to evaluate the statement as an expression
            if not entire_flag:  # computes with just the last samples
                feature_fusion_string = (
                            "(Buffer_features_2_rescaled[" + feature[0] + "][-1]" + feature[1] +
                            "Buffer_features_2_rescaled[" + feature[2] + "][-1]" +
                            feature[3] + "Buffer_features_2_rescaled[" + feature[4] + "][-1])")
                feature_fusion_value = eval(feature_fusion_string) / 3
                # uses eval to evaluate the statement as an expression
            feature_fusion_list.append(feature_fusion_value)

        if len(feature) == 6:  # Case  of 3 features first negative
            if entire_flag:  # computes with all the matrix
                feature_fusion_string = ("(-Buffer_features_2_rescaled[" + feature[1] + "]" + feature[2] +
                                         "Buffer_features_2_rescaled[" + feature[3] + "]" +
                                         feature[4] + "Buffer_features_2_rescaled[" + feature[5] + "])")
                feature_fusion_value = eval(feature_fusion_string) / 3
                # uses eval to evaluate the statement as an expression
            if not entire_flag:  # computes with just the last samples
                feature_fusion_string = (
                            "(-Buffer_features_2_rescaled[" + feature[1] + "][-1]" + feature[2] +
                            "Buffer_features_2_rescaled[" + feature[3] + "][-1]" +
                            feature[4] + "Buffer_features_2_rescaled[" + feature[5] + "][-1])")
                feature_fusion_value = eval(feature_fusion_string) / 3
                # uses eval to evaluate the statement as an expression
            feature_fusion_list.append(feature_fusion_value)

        if len(feature) == 3:  # Case  of 2 features first positive
            if entire_flag:  # computes with all the matrix
                feature_fusion_string = ("(Buffer_features_2_rescaled[" + feature[0] + "]" + feature[1] +
                                         "Buffer_features_2_rescaled[" + feature[2] + "])")
                feature_fusion_value = eval(feature_fusion_string) / 2
                # uses eval to evaluate the statement as an expression
            if not entire_flag:  # computes with just the last samples
                feature_fusion_string = (
                            "(Buffer_features_2_rescaled[" + feature[0] + "][-1]" + feature[1] +
                            "Buffer_features_2_rescaled[" + feature[2] + "][-1])")
                feature_fusion_value = eval(feature_fusion_string) / 2
                # uses eval to evaluate the statement as an expression
            feature_fusion_list.append(feature_fusion_value)

        if len(feature) == 4:  # Case  of 2 features first negative
            if entire_flag:  # computes with all the matrix
                feature_fusion_string = ("(-Buffer_features_2_rescaled[" + feature[1] + "]" + feature[2] +
                                         "Buffer_features_2_rescaled[" + feature[3] + "])")
                feature_fusion_value = eval(feature_fusion_string) / 2
                # uses eval to evaluate the statement as an expression
            if not entire_flag:  # computes with just the last samples
                feature_fusion_string = (
                            "(-Buffer_features_2_rescaled[" + feature[1] + "][-1]" + feature[2] +
                            "Buffer_features_2_rescaled[" + feature[3] + "][-1])")
                feature_fusion_value = eval(feature_fusion_string) / 2
                # uses eval to evaluate the statement as an expression
            feature_fusion_list.append(feature_fusion_value)

    return feature_fusion_list


def rescale_last_epoch(buffer, timestamp_buffer, rescale_epoch):
    """
    Rescales a buffer row-wise including the samples of features fusions within an interval of time specified by the
    costumer, rescale_epoch
    :param buffer: numpy array to rescale its last N feature samples. N is the number samples within rescale_epoch
    seconds
    :param timestamp_buffer: timestamp array alongside the buffer. This is an array of one dimension will be iterated in
    order to know the sample index that occurred "rescale_epoch" seconds ago
    :param rescale_epoch: interval of time in seconds
    :return:
    """
    count_time_difference = -1
    while count_time_difference > - len(timestamp_buffer):  # Finds the first sample to be rescale_epoch
        # seconds older than the last one
        time_difference = timestamp_buffer[-1] - timestamp_buffer[count_time_difference]
        if time_difference >= rescale_epoch:  # if so, rescales the N samples of buffer from the one that is
            # rescale_epoch x seconds old
            buffer[:, count_time_difference:] = np.array([-1 + 2*((row-np.nanmin(row))/(np.nanmax(row)-np.nanmin(row)))
                                                           for row in buffer[:, count_time_difference:]])
            break
        count_time_difference -= 1
    return buffer

def songs_generator():
    """
    function to generate 4 list of songs
    :return: 4 lists of songs
    """
    Base_Path = os.path.dirname(__file__)

    content_folder = os.path.join(Base_Path, 'somesongs___ogg\Content')
    content_songs = [os.path.join(content_folder, f) for f in os.listdir(content_folder) if ".ogg" in f]

    melancholic_folder = os.path.join(Base_Path, 'somesongs___ogg\Melancholic')
    melancholic_songs = [os.path.join(melancholic_folder, f) for f in os.listdir(melancholic_folder) if ".ogg" in f]

    disgusting_folder = os.path.join(Base_Path, 'somesongs___ogg\Disgusting')
    disgusting_songs = [os.path.join(disgusting_folder, f) for f in os.listdir(disgusting_folder) if ".ogg" in f]

    happy_folder = os.path.join(Base_Path, 'somesongs___ogg\Happy')
    happy_songs = [os.path.join(happy_folder, f) for f in os.listdir(happy_folder) if ".ogg" in f]

    return content_songs, melancholic_songs, disgusting_songs, happy_songs


def fusion_parameters(parameter):
    """
    :param parameter: keyword, if it is "standard", it delivers a predefined set of feature fusions, if not, it catches
    the ones in the Json of user choice
    :return: 2 feature fusion variables
    """
    if parameter == 'standard_8':
        feature_fusion1 = "1+2+3"
        feature_fusion2 = "5-4"
    elif parameter == 'standard_20':
        feature_fusion1 = "1+2+3+4"
        feature_fusion2 = "-4+5+6"
    else:
        feature_fusion1 = Config_APP['Input_Variables'][0]["Feature_Fusion"]["Feature_Fusion_1"]
        feature_fusion2 = Config_APP['Input_Variables'][0]["Feature_Fusion"]["Feature_Fusion_2"]

    return feature_fusion1, feature_fusion2

