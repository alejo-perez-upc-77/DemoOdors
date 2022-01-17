import numpy as np
import matplotlib.pyplot as plt

def plot_signals_acc(ax1, ax2,ax3, time_stamp_buffer_EEG, Buffer_EEG, time_stamp_buffer_Accelerometer,
                     Buffer_Accelerometer, feature_computed, t0_EEG,t0_Acc):
    '''
    Plots the realtime EEG, and accelerometer signal buffers refreshing both axes in order to update
    '''

    ax1.clear()
    ax2.clear()
    ax3.clear()

    for index, p in enumerate(range(len(Buffer_EEG))):
        ax1.plot(time_stamp_buffer_EEG-t0_EEG, Buffer_EEG[p] - np.mean(Buffer_EEG[p]) - index*2);
    ax1.set_title('EEG')
    ax1.set_ylabel('Channels')
    for p in range(len(Buffer_Accelerometer)):  # opcional
        ax2.plot(time_stamp_buffer_Accelerometer-t0_Acc, Buffer_Accelerometer[p] - np.mean(Buffer_Accelerometer[p]) );
    ax2.set_title('Accelerometer')
    ax2.set_ylabel('Channels')

    ax3.bar(feature_computed.keys(), feature_computed.values())
    ax3.set_xticklabels(list(feature_computed.keys()), rotation='20', fontsize=8);
    ax3.set_yscale('log')
    ax3.set_title('Features')
    ax3.set_ylabel('Power')

    plt.pause(0.0001)
