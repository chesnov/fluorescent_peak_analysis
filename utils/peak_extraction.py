import numpy as np
import sys
from contextlib import contextmanager
import seaborn as sns
import numpy as np

#load txt files as csv into numpy arrays
import numpy as np
import kaleido #required
kaleido.__version__ #0.2.1

import plotly
plotly.__version__ #5.5.0

#now this works:
import plotly.graph_objects as go
from scipy.signal import find_peaks
import os
from os.path import isdir, join
from os import listdir
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import colorsys

#Import pingouin
import pingouin as pg

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    cv2.setNumThreads(0)
except():
    pass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.caiman_wrapper import *


pipeline_version = '0.1.0'

@contextmanager
def suppress_output():
    """Context manager to suppress all outputs (stdout and stderr)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

            
def load_data(file_path):
    #Load numpy array from the csv file
    array = np.loadtxt(file_path, delimiter=',')
    #Transpose the array
    array = array.T
    #Reshape array to 2D if it is of a shape (n,) where n is the number of elements (single ROI edge case)
    if len(array.shape) == 1:
        array = array.reshape(-1, 1)

    #Convert to pandas dataframe
    df = pd.DataFrame(array)

    #Assign column names as 'ROI_0', 'ROI_1', etc.
    df.columns = [f'ROI_{i}' for i in range(df.shape[1])]
    
    #Check how long is the recording
    num_samples = array.shape[0]

    #If the recording is longer than 1 minute, remove everything after 1 minute
    if num_samples > 6000:
        array = array[:6000, :]
        df = df.iloc[:6000, :]

    return array, df


def compute_firing_frequency(peaks, num_samples, sampling_rate):
    """
    Compute the firing frequency (peaks per second) based on identified peaks.
    
    Parameters:
    - peaks: array-like, indices of the detected peaks.
    - num_samples: int, total number of samples in the recording.
    - sampling_rate: int, number of samples per second (e.g., 100).
    
    Returns:
    - firing_frequency: list, firing frequency (peaks per second) for each second.
    """
    # Calculate the total recording time in seconds
    total_seconds = num_samples // sampling_rate
    
    # Initialize a list to hold the firing frequency for each second
    firing_frequency = []
    
    # Loop through each second and count the number of peaks in that interval
    for sec in range(total_seconds):
        start_idx = sec * sampling_rate  # Start of the second in samples
        end_idx = (sec + 1) * sampling_rate  # End of the second in samples
        peaks_in_second = len([peak for peak in peaks if peak >= start_idx and peak < end_idx])
        
        firing_frequency.append(peaks_in_second)
    
    return firing_frequency


# Define rolling window smoothing function
def rolling_window_smooth(signal, window_size):
    """Apply rolling window smoothing using a simple moving average."""
    window = np.ones(window_size) / window_size  # Create a uniform window
    smoothed_signal = np.convolve(signal, window, mode='same')  # Apply smoothing
    return smoothed_signal


def precise_peak_locs(smoothed_peaks, trace, median_val):
    heights = []
    peaks = []
    for peak in smoothed_peaks:
        #smoothed peak location might not the actual peak due to smoothing
        #Get the position within original trace corresponding to the largest value in the window
        true_peak = np.argmax(trace[peak-15:peak+15]) + peak - 15 #Due to smoothing the peak is shifted to the right
        peaks.append(true_peak)
        absolute_height = trace[true_peak]
        real_height = absolute_height - median_val
        heights.append([absolute_height, real_height])
    return peaks, heights


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def peak_finder(trace):
    window_size = 15  # To be adjusted if needed
    smoothed_trace = rolling_window_smooth(trace, window_size)
    #remove the edges
    smoothed_trace = smoothed_trace[window_size:-window_size]
    smoothed_trace = np.pad(smoothed_trace, (window_size, window_size), 'constant', constant_values=(0, 0))

    #calculate 50th percentile
    percentile = np.percentile(smoothed_trace, 50)
    below_percentile = smoothed_trace[smoothed_trace < percentile]
    below_percentile_std = np.std(below_percentile)
    median_val = np.median(below_percentile)
    # mad = np.median(np.abs(smoothed_trace - median_val)) * 1.4826
    height_threshold = median_val + 9 * below_percentile_std
    prominence_threshold = 8 * below_percentile_std

    # median_val = np.median(smoothed_trace)
    # mad = np.median(np.abs(smoothed_trace - median_val)) * 1.4826
    # height_threshold = median_val + 3 * mad
    # prominence_threshold = 6 * mad
    width_threshold = 10 #Empirically determined
    smoothed_peaks, _ = find_peaks(smoothed_trace, height=height_threshold, prominence=prominence_threshold, width=width_threshold)
    peaks, heights = precise_peak_locs(smoothed_peaks, trace, median_val)
    
    return peaks, np.array(heights), height_threshold, smoothed_trace


def plot_peaks(roi, trace, smoothed_trace, peaks, peak_heights, height_threshold, data_output_dir):
    x_arr = np.arange(trace.shape[0]) / 100
    fig = go.Figure()
    #Set figure size
    fig.update_layout(width=1000, height=600)
    fig.add_trace(go.Scatter(x=x_arr, y=trace, name=roi))
    #check if the 0th peak is a None
    if peaks[0] is not None:
        fig.add_trace(go.Scatter(x=[p/100 for p in peaks], y=trace[peaks], mode='markers', name='Peaks'))
    fig.add_trace(go.Scatter(x=x_arr, y=smoothed_trace, name='Smoothed Trace'))
    fig.add_hline(y=height_threshold, line_dash='dash', line_color='gray', annotation_text='Threshold', annotation_position='top right')

    #Now do the same for dark background
    fig.update_layout(template='plotly_dark')
    fig.update_layout(xaxis_title='Time [s]', yaxis_title='Intensity [dF/F]', plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    

    #Add peak heights (from properties) next to the peaks
    # for i, peak in enumerate(peaks):
    #     fig.add_annotation(x=peak, y=trace[peak], text=f'{peak_heights[i]:.2f}', showarrow=False)
    
    output_file_root = os.path.join(data_output_dir, roi + '_peaks')
    #Save the plot as html
    fig.write_html(output_file_root + '.html') 
    #Save the plot as pdf
    fig.write_image(output_file_root + '.pdf', format='pdf')


# Plot the firing frequency
def plot_firing_frequency(firing_freq_per_sec):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(firing_freq_per_sec)), y=firing_freq_per_sec))
    fig.update_layout(title='Firing Frequency per Second', xaxis_title='Second', yaxis_title='Firing Frequency')
    fig.show()


def peak_to_peak_distance(df, recordings_duration, data_output_dir, experiment_id):
    #calculate the average peak-to-peak distance for each ROI (each time sample is 1ms)
    peak_to_peak_dists_columnames = ['roi_id', 'mean_peak_to_peak_distance[ms]', 'mean_firing_frequency[Hz]']
    peak_to_peak_dists = pd.DataFrame(columns=peak_to_peak_dists_columnames)
    for roi in df['roi_id'].unique():
        roi_df = df[df['roi_id'] == roi]
        peak_times = roi_df['peak_time'].values
        peak_to_peak_distances = np.diff(peak_times)
        avg_peak_to_peak_distance = np.mean(peak_to_peak_distances)
        mean_firing_frequency = np.array(len(peak_times) / recordings_duration)
        row_df = pd.DataFrame([[roi, avg_peak_to_peak_distance, mean_firing_frequency]], columns=peak_to_peak_dists_columnames)
        peak_to_peak_dists = pd.concat([peak_to_peak_dists, row_df], ignore_index=True)

    #Save peak_to_peak_dists as csv
    peak_to_peak_dists.to_csv(os.path.join(data_output_dir, experiment_id + '_firing_frequency.csv'), index=False)


def calc_noise_levels(array, roi_id):
    #iterating over each roi, calculate the noise level, assuming baseline is at the 5th percentile
    trace = array[:, roi_id]
    percentile = np.percentile(trace, 50)
    below_percentile = trace[trace < percentile]
    below_percentile_std = np.std(below_percentile)
    return below_percentile_std


#plot noise level for each roi
def plot_noise_level(array):
    noise_levels = []
    for roi_id in range(array.shape[1]):
        noise_levels.append(calc_noise_levels(array, roi_id))
    #plot a line of best fit
    line = np.polyfit(np.arange(len(noise_levels)), noise_levels, 1)
    line_fn = np.poly1d(line)   
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(noise_levels)), y=noise_levels, mode='markers', name='ROIs'))
    fig.add_trace(go.Scatter(x=np.arange(len(noise_levels)), y=line_fn(np.arange(len(noise_levels))), name='Line of Best Fit'))

    #Make the plot black
    fig.update_layout(template='plotly_dark')
    fig.update_layout(xaxis_title='ROI', yaxis_title='Median Absolute Deviation', plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    fig.show()


def plot_noise_level_histogram(experiment_df, data_output_dir):
    #Plot a histogram of the noise levels using plotly
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=experiment_df['noise_level'], nbinsx=50))
    fig.update_layout(title='Noise Level Distribution', xaxis_title='Noise Level', yaxis_title='Count')
    #Makee the plot dark
    fig.update_layout(template='plotly_dark')
    #Save as html and as pdf
    fig.write_html('noise_level_histogram.html')

    output_file_root = os.path.join(data_output_dir, '_noise_level_histogram')
    #Save the plot as html
    fig.write_html(output_file_root + '.html') 
    #Save the plot as pdf
    fig.write_image(output_file_root + '.pdf', format='pdf')
    

def raw_signal_pearson(array):
    # Compute the Pearson correlation coefficient between each pair of ROIs
    corr_matrix = np.corrcoef(array, rowvar=False)
    # Plot correlation matrix using plotly
    fig = go.Figure(data=go.Heatmap(z=corr_matrix, colorscale='Viridis'))
    fig.update_layout(width=800, height=800)
    #make the plot dark
    fig.update_layout(template='plotly_dark')
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    #Add label to heat bar
    fig.update_layout(coloraxis_colorbar=dict(title='Pearson Correlation'))
    #Add labels to x and y axis
    fig.update_xaxes(title_text='ROI ID')
    fig.update_yaxes(title_text='ROI ID')
    fig.show()


def peak_event_synchrony_per_peak(peaks, window_size=15):
    """
    Calculate the synchrony between ROIs on a per-peak basis.

    Parameters:
    - peaks: numpy array of shape (N, T), where N is the number of ROIs and T is the number of timepoints.
             Values should be 0 (no peak) or 1 (peak).
    - window_size: int, the time window (±window_size) around each peak to check for coincidences.

    Returns:
    - synchrony_matrix: numpy array of shape (N, N), representing the proportion of peaks in each ROI
                        that have a corresponding peak in another ROI within the given window.
    """
    N, T = peaks.shape
    synchrony_matrix = np.zeros((N, N))
    
    for i in range(N):  # Iterate over each ROI
        for j in range(N):  # Compare with every other ROI
            roi_i_peaks = np.where(peaks[i] == 1)[0]  # Get peak indices for ROI i
            total_peaks_i = len(roi_i_peaks)
            if total_peaks_i == 0:  # Skip if ROI i has no peaks
                synchrony_matrix[i, j] = 0
                continue
            
            # Check for coinciding peaks within the window for each peak in ROI i
            coinciding_peaks = 0
            for peak_idx in roi_i_peaks:
                # Define the time window
                start_idx = max(0, peak_idx - window_size)
                end_idx = min(T, peak_idx + window_size + 1)
                # Check if ROI j has a peak in this window
                if np.any(peaks[j, start_idx:end_idx] == 1):
                    coinciding_peaks += 1
            
            # Calculate the proportion of peaks in ROI i with a corresponding peak in ROI j
            synchrony_matrix[i, j] = coinciding_peaks / total_peaks_i

    return synchrony_matrix



def synchrony_calculation(df, array, data_output_dir, experiment_id):
    # N: number of neurons, T: number of time points
    N = array.shape[1]
    T = array.shape[0]

    #Make a matrix with 1s where peaks are and 0 where they are not
    PeakRegions = np.zeros((N, T))
    for _, row in df.iterrows():
        roi_id = int(row['roi_id'].rsplit('ROI_')[1])
        #Check if the peak is not None
        if not pd.isna(row['peak_time']):
            PeakRegions[roi_id, int(row['peak_time'])] = 1

    aggregated_corr = peak_event_synchrony_per_peak(PeakRegions, window_size=15)

    #Identify what value is on the diagonal and scale the matrix to be between 1 and 0 where the largest value is 1
    for i in range(N):
        diag_value = aggregated_corr[i, i]
        if diag_value > 0:
            aggregated_corr = aggregated_corr / diag_value
            break
    
    #Save the aggregated_corr as csv
    aggregated_corr_df = pd.DataFrame(aggregated_corr)
    aggregated_corr_df.to_csv(os.path.join(data_output_dir, experiment_id + '_aggregated_corr.csv'), index=False)

    # Use plotly to plot the heatmap
    fig = go.Figure(data=go.Heatmap(z=aggregated_corr, colorscale='Inferno'))
    #set colorscale range from 0 to 1
    fig.update_traces(zmin=0, zmax=1)

    fig.update_layout(width=800, height=800)
    #make the plot dark
    fig.update_layout(template='plotly_dark')
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    #Add axis labels
    fig.update_layout(xaxis_title='ROI ID')
    fig.update_layout(yaxis_title='ROI ID')
    
    #save as html and as pdf
    fig.write_html(os.path.join(data_output_dir, experiment_id + '_aggregated_corr.html'))
    fig.write_image(os.path.join(data_output_dir, experiment_id + '_aggregated_corr.pdf'), format='pdf')


def full_pipeline(experiment_tif, experiment_outdir, yaml_file):
    #get the base filename withou txt extension
    experiment_id = os.path.splitext(os.path.basename(experiment_tif))[0]

    with suppress_output():
        try:
            raw_data_to_df_f(experiment_tif, yaml_file, experiment_outdir)
        except Exception:
            print(f"Exception occurred during processing of {experiment_id}")
    
    #Identify csv file in outdir
    try:
        csv_file = [f for f in os.listdir(experiment_outdir) if f.endswith('.csv')][0]  
    except:
        print(f'No ROIs extracted from {experiment_id}')
        return
    array, df = load_data(join(experiment_outdir, csv_file))

    roi_ids = list(df.columns)

    experiment_df_columns = ['roi_id', 'peak_time', 'peak_absolute_amplitude', 'peak_relative_amplitude']
    experiment_df = pd.DataFrame(columns=experiment_df_columns)

    for roi, roi_id in enumerate(roi_ids):
        trace = array[:, roi] #selece just one roi 
        peaks, peak_heights, height_threshold, smoothed_trace = peak_finder(trace)
        if len(peaks) == 0:
            peaks = [None]
            peak_heights = np.array([[None, None]])
        
        #Add peaks to experiment_df
        roi_df = pd.DataFrame(columns=experiment_df_columns)
        roi_df['roi_id'] = [roi_id for _ in range(len(peaks))]
        roi_df['peak_time'] = peaks
        roi_df['peak_absolute_amplitude'] = peak_heights[:, 0]
        roi_df['peak_relative_amplitude'] = peak_heights[:, 1]

        experiment_df = pd.concat([experiment_df, roi_df], ignore_index=True)
        plot_peaks(roi_id, trace, smoothed_trace, peaks, peak_heights[:, 0], height_threshold, experiment_outdir)

    experiment_df['noise_level'] = experiment_df['roi_id'].apply(lambda x: calc_noise_levels(array, int(x.rsplit('ROI_')[1])))
    plot_noise_level_histogram(experiment_df, experiment_outdir)

    #Save experiment_df as csv
    experiment_df.to_csv(os.path.join(experiment_outdir, experiment_id + '_experiment_df.csv'), index=False)

    recordings_duration = len(array) / 1000 #in seconds
    peak_to_peak_distance(experiment_df, recordings_duration, experiment_outdir, experiment_id)

    synchrony_calculation(experiment_df, array, experiment_outdir, experiment_id)


#A function that takes a dark hex color and returns a palette of n colors that go from pastel to dark
def generate_palette(dark_hex: str, n: int) -> list:
    """
    Generate a palette of `n` colors transitioning from pastel to the given dark color.
    
    Parameters:
        dark_hex (str): A dark hex color (e.g., '#123456').
        n (int): Number of colors in the palette.
        
    Returns:
        list: A list of `n` hex color strings.
    """
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convert a hex color string to an RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    def rgb_to_hex(rgb: tuple) -> str:
        """Convert an RGB tuple to a hex color string."""
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    
    # Convert dark hex color to RGB and then to HLS
    dark_rgb = hex_to_rgb(dark_hex)
    dark_hls = colorsys.rgb_to_hls(*dark_rgb)
    
    # Generate pastel color by increasing lightness
    pastel_hls = (dark_hls[0], min(dark_hls[1] + 0.5, 1.0), dark_hls[2])
    pastel_rgb = colorsys.hls_to_rgb(*pastel_hls)
    
    # Interpolate between pastel and dark colors
    palette = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0  # Avoid division by zero
        interpolated_hls = (
            dark_hls[0],  # Keep hue the same
            pastel_hls[1] * (1 - t) + dark_hls[1] * t,  # Interpolate lightness
            pastel_hls[2] * (1 - t) + dark_hls[2] * t   # Interpolate saturation
        )
        interpolated_rgb = colorsys.hls_to_rgb(*interpolated_hls)
        palette.append(rgb_to_hex(interpolated_rgb))
    
    return palette


def plot_peak_amplitudes(experiments_amplitude_df, group_to_eid):
    experiments_amplitude_df = experiments_amplitude_df.copy(deep=True)
    color_options = ['#00312F', '#1D2D46', '#46000D', '#5F3920']
    groups = list(group_to_eid.keys())
    group_palette = {}
    experiment_palette = {}
    for i, gr in enumerate(groups):
        eids = group_to_eid[gr]
        pastel_palette_arr = generate_palette(color_options[i], len(eids) + 1)
        group_palette[gr] = pastel_palette_arr[0]
        group_dict = {e : pastel_palette_arr[j + 1] for j, e in enumerate(eids)}
        #concatenate group_palette and group_dict
        experiment_palette.update(group_dict)

    # Remove entries with experiment id not among eids in the dictionary
    legal_eids = list(group_to_eid.values())
    #Make a flat list
    legal_eids = [item for sublist in legal_eids for item in sublist]

    #Remove all experiments that are not among legal eids
    experiments_amplitude_df = experiments_amplitude_df[experiments_amplitude_df['experiment_id'].isin(legal_eids)]

    eid_to_group = {eid : group for group, eids in group_to_eid.items() for eid in eids}

    #Add a group column to the dataframe
    experiments_amplitude_df['group'] = experiments_amplitude_df['experiment_id'].apply(lambda x: eid_to_group[x])

    # Remove ROIS that are beyond 2 stds from noise mean
    mean_noise = experiments_amplitude_df['noise_level'].mean()
    std_noise = experiments_amplitude_df['noise_level'].std()
    #Get a number of unique experiment id and roi pairs before filtering
    num_rois_before_filtering = experiments_amplitude_df.set_index(['experiment_id', 'roi_id']).index.nunique()
    min_noise_thresh = mean_noise - 0.5 * std_noise
    max_noise_thresh = mean_noise + 0.5 * std_noise

    #Create a dataframe with ROIs that need to be removed due to high noise and associated experiment
    rois_to_remove = experiments_amplitude_df.copy(deep=True)
    rois_to_remove = rois_to_remove[(rois_to_remove['noise_level'] > max_noise_thresh) | (rois_to_remove['noise_level'] < min_noise_thresh)]
    rois_to_remove = rois_to_remove.drop_duplicates()
    rois_to_remove = rois_to_remove[['experiment_id', 'roi_id']]

    #Remove rois that match rois_to_remove from experiments_amplitude_df
    experiments_amplitude_df = experiments_amplitude_df[
        ~experiments_amplitude_df.set_index(['experiment_id', 'roi_id']).index.isin(
            rois_to_remove.set_index(['experiment_id', 'roi_id']).index
        )
    ]

    num_rois_after_filtering = experiments_amplitude_df.set_index(['experiment_id', 'roi_id']).index.nunique()
    print(f"Keeping {num_rois_after_filtering} ROIs out of {num_rois_before_filtering}")
    print(f"Noise cutoffs are {min_noise_thresh} and {max_noise_thresh}")

    #For each experiment, check if it still has at least 3 different ROIs
    experiments_with_enough_rois = experiments_amplitude_df.groupby('experiment_id').filter(lambda x: x['roi_id'].nunique() >= 3)
    #Get a list of experiments that have less than 3 ROIs
    experiments_with_few_rois = experiments_amplitude_df[~experiments_amplitude_df['experiment_id'].isin(experiments_with_enough_rois['experiment_id'].unique())]
    if len(experiments_with_few_rois) > 0:
        print(f"Experiments with less than 3 ROIs: {experiments_with_few_rois['experiment_id'].unique()}")
        #If an experiment has less than 3 ROIs, remove it from the dataframe
        #Add these experiments and associated ROIs to rois_to_remove
        rois_to_remove = pd.concat([rois_to_remove, experiments_with_few_rois[['experiment_id', 'roi_id']]], ignore_index=True)
        experiments_amplitude_df = experiments_amplitude_df[
            ~experiments_amplitude_df.set_index(['experiment_id', 'roi_id']).index.isin(
                rois_to_remove.set_index(['experiment_id', 'roi_id']).index
            )
        ]

    #Make a new dataframe with average values for each ROI
    experiment_avg_peak_amplitudes = experiments_amplitude_df.groupby('experiment_id').agg({'peak_absolute_amplitude': 'mean', 'peak_relative_amplitude' : 'mean' }).reset_index()
    experiment_avg_peak_amplitudes['group'] = experiment_avg_peak_amplitudes['experiment_id'].apply(lambda x: eid_to_group[x])

    roi_avg_peak_amplitudes = experiments_amplitude_df.groupby(['experiment_id', 'roi_id']).agg({'peak_absolute_amplitude': 'mean', 'peak_relative_amplitude' : 'mean' }).reset_index()
    roi_avg_peak_amplitudes['group'] = roi_avg_peak_amplitudes['experiment_id'].apply(lambda x: eid_to_group[x])

    #Sort experiments_amplitude_df and experiment_avg_peak_amplitudes to make sure control conditions are first
    experiments_amplitude_df = experiments_amplitude_df.sort_values(['group', 'experiment_id'], ascending=False)
    experiment_avg_peak_amplitudes = experiment_avg_peak_amplitudes.sort_values(['group', 'experiment_id'], ascending=False)
    roi_avg_peak_amplitudes = roi_avg_peak_amplitudes.sort_values(['group', 'experiment_id'], ascending=False)

    #Use pingouin to run ANOVA on plot_peak_amplitudes to check if there are significant differences between groups
    aov = pg.anova(data=experiment_avg_peak_amplitudes, dv='peak_absolute_amplitude', between='group', detailed=True)
    print(aov)

    # Create the plot
    plt.figure(figsize=(len(groups), 4))
    #Make the plot dark
    plt.style.use('dark_background')
    # Boxplot with group colors
    sns.boxplot(data=experiment_avg_peak_amplitudes, x='group', y='peak_absolute_amplitude', palette=group_palette)
    # Swarmplot with experiment_id colors
    sns.swarmplot(data=experiment_avg_peak_amplitudes, x='group', y='peak_absolute_amplitude', hue='experiment_id', palette=experiment_palette)
    # Adjust legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # Labels and title
    plt.xlabel('Group')
    plt.ylabel('Peak Absolute Amplitude')
    plt.title('Peak Absolute Amplitude by Group (average per experiment)')
    #If there are more than 2 groups, hide legend
    if len(groups) > 2:
        plt.legend().remove()
    # Layout and save as PDF
    plt.tight_layout()
    plt.savefig('Peak_Absolute_Amplitude_by_Group.pdf', format='pdf')
    plt.show()


    #Use seaborn to plot a violin plot of the peak relative amplitudes
    plt.figure(figsize=(len(groups), 6))
    sns.violinplot(data=roi_avg_peak_amplitudes, x='group', y='peak_absolute_amplitude', palette=group_palette)
    #Add swarmplot
    if len(groups) <= 2:
        sns.swarmplot(data=roi_avg_peak_amplitudes, x='group', y='peak_absolute_amplitude', hue='experiment_id', palette=experiment_palette)
    #put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Group')
    plt.ylabel('Peak Absolute Amplitude')
    plt.title('Peak Absolute Amplitude by Group (average per ROI)')
    #If there are more than 2 groups, hide legend
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig('Peak_Absolute_Amplitude_by_ROI.pdf', format='pdf')
    plt.show()

    #Use seaborn to plot a violin plot of the peak relative amplitudes
    plt.figure(figsize=(len(groups), 4))
    sns.violinplot(data=experiments_amplitude_df, x='group', y='peak_absolute_amplitude', palette=group_palette)
    #Add swarmplot
    # sns.swarmplot(data=experiments_amplitude_df, x='group', y='peak_absolute_amplitude', hue='experiment_id')
    plt.xlabel('Group')
    plt.ylabel('Peak Absolute Amplitude')
    plt.title('Peak Absolute Amplitude by Group')
    plt.tight_layout()
    #Seve as pdf
    plt.savefig('Peak_Absolute_Amplitude.pdf', format='pdf')
    plt.show()

    return rois_to_remove, group_palette, experiment_palette, experiment_avg_peak_amplitudes


def plot_frequencies(experiments_frequency_df, group_to_eid, rois_to_remove, group_palette, experiment_palette):
    experiments_frequency_df = experiments_frequency_df.copy(deep=True)
    groups = list(group_to_eid.keys())
    # Remove entries with experiment id not among eids in the dictionary
    legal_eids = list(group_to_eid.values())
    #Make a flat list
    legal_eids = [item for sublist in legal_eids for item in sublist]

    #Remove all experiments that are not among legal eids
    experiments_frequency_df = experiments_frequency_df[experiments_frequency_df['experiment_id'].isin(legal_eids)]

    eid_to_group = {eid : group for group, eids in group_to_eid.items() for eid in eids}
    
    experiments_frequency_df['group'] = experiments_frequency_df['experiment_id'].apply(lambda x: eid_to_group[x])

    # Remove ROIS that are beyond 2 stds from noise mean

    #Remove rois that match rois_to_remove from experiments_frequency_df
    experiments_frequency_df = experiments_frequency_df[
        ~experiments_frequency_df.set_index(['experiment_id', 'roi_id']).index.isin(
            rois_to_remove.set_index(['experiment_id', 'roi_id']).index
        )
    ]

    #Make a new dataframe with average values for each experiment
    experiment_avg_firing_frequency = experiments_frequency_df.groupby('experiment_id').agg({'mean_firing_frequency[Hz]': 'mean', 'mean_peak_to_peak_distance[ms]' : 'mean'}).reset_index()
    experiment_avg_firing_frequency['group'] = experiment_avg_firing_frequency['experiment_id'].apply(lambda x: eid_to_group[x])

    #Sort experiment_avg_firing_frequency and experiments_frequency_df to make sure control conditions are first
    experiment_avg_firing_frequency = experiment_avg_firing_frequency.sort_values(['group', 'experiment_id'], ascending=False)
    experiments_frequency_df = experiments_frequency_df.sort_values(['group', 'experiment_id'], ascending=False)

    #Use pingouin to run ANOVA on experiment_avg_firing_frequency to check if there are significant differences between groups
    aov = pg.anova(data=experiment_avg_firing_frequency, dv='mean_firing_frequency[Hz]', between='group', detailed=True)
    print(aov)

    #Use seaborn to plot a boxplot of the peak absolute amplitudes
    plt.figure(figsize=(len(groups), 4))
    sns.boxplot(data=experiment_avg_firing_frequency, x='group', y='mean_firing_frequency[Hz]', palette=group_palette)
    #Add swarmplot
    sns.swarmplot(data=experiment_avg_firing_frequency, x='group', y='mean_firing_frequency[Hz]', hue='experiment_id', palette=experiment_palette)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Experiment ID')
    plt.ylabel('Mean Firing Frequency [Hz]')
    plt.title('Mean Firing Frequency by Experiment')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig('Mean_Firing_Frequency_by_Experiment.pdf', format='pdf')
    plt.show()

    #Use seaborn to plot a boxplot of the peak absolute amplitudes
    plt.figure(figsize=(len(groups), 4))
    sns.boxplot(data=experiment_avg_firing_frequency, x='group', y='mean_peak_to_peak_distance[ms]', palette=group_palette)
    #Add swarmplot
    sns.swarmplot(data=experiment_avg_firing_frequency, x='group', y='mean_peak_to_peak_distance[ms]', hue='experiment_id', palette=experiment_palette)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Experiment ID')
    plt.ylabel('Mean peak to peak distance [ms]')
    plt.title('Mean Firing Frequency by Experiment')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig('Mean_Peak_to_Peak_by_Experiment.pdf', format='pdf')
    plt.show()

    #Use seaborn to plot a violin plot of the firing frequency with individual ROIs
    plt.figure(figsize=(len(groups), 4))
    sns.violinplot(data=experiments_frequency_df, x='group', y='mean_firing_frequency[Hz]', palette=group_palette)
    #Add swarmplot
    # sns.swarmplot(data=experiments_frequency_df, x='group', y='mean_firing_frequency[Hz]', hue='experiment_id', palette=experiment_palette)
    #put legend outside the plot
    plt.xlabel('Group')
    plt.ylabel('Mean Firing Frequency [Hz]')
    plt.title('Mean Firing Frequency by Group')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig('Mean_Firing_Frequency.pdf', format='pdf')
    plt.show()

    #Use seaborn to plot a violin plot of the firing frequency with individual ROIs
    plt.figure(figsize=(4, 4))
    sns.violinplot(data=experiments_frequency_df, x='group', y='mean_peak_to_peak_distance[ms]', palette=group_palette)
    #Add swarmplot
    # sns.swarmplot(data=experiments_frequency_df, x='group', y='mean_firing_frequency[Hz]', hue='experiment_id', palette=experiment_palette)
    #put legend outside the plot
    plt.xlabel('Group')
    plt.ylabel('Mean peak to peak distance [ms]')
    plt.title('Mean Firing Frequency by Group')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig('Mean_Peak_to_Peak.pdf', format='pdf')
    plt.show()

    return experiment_avg_firing_frequency

def plot_synchronie(experimets_synchrony_df, group_to_eid, rois_to_remove, group_palette, experiment_palette):
    experimets_synchrony_df = experimets_synchrony_df.copy(deep=True)
    groups = list(group_to_eid.keys())
    # Remove entries with experiment id not among eids in the dictionary
    legal_eids = list(group_to_eid.values())
    #Make a flat list
    legal_eids = [item for sublist in legal_eids for item in sublist]

    #Remove all experiments that are not among legal eids
    experimets_synchrony_df = experimets_synchrony_df[experimets_synchrony_df['experiment_id'].isin(legal_eids)]

    eid_to_group = {eid : group for group, eids in group_to_eid.items() for eid in eids}

    experimets_synchrony_df['group'] = experimets_synchrony_df['experiment_id'].apply(lambda x: eid_to_group[x])

    #If either of the ROIs is in the list of ROIs to remove, remove the row
    experimets_synchrony_df = experimets_synchrony_df[
        ~experimets_synchrony_df.set_index(['experiment_id', 'ROI_a']).index.isin(
            rois_to_remove.set_index(['experiment_id', 'roi_id']).index
        )
    ]
    experimets_synchrony_df = experimets_synchrony_df[
        ~experimets_synchrony_df.set_index(['experiment_id', 'ROI_b']).index.isin(
            rois_to_remove.set_index(['experiment_id', 'roi_id']).index
        )
    ]

    #Calculate mean synchrony for each experiment
    mean_synchrony_df = experimets_synchrony_df.groupby(['experiment_id', 'group']).agg({'synchrony': 'mean'}).reset_index()

    #sort the dataframe by group
    mean_synchrony_df = mean_synchrony_df.sort_values(['group', 'experiment_id'], ascending=False)

    #Use pingouin to run ANOVA on mean_synchrony_df to check if there are significant differences between groups
    aov = pg.anova(data=mean_synchrony_df, dv='synchrony', between='group', detailed=True)
    print(aov)

    #Use seaborn to plot a boxplot of the synchrony
    plt.figure(figsize=(4, 4))
    sns.boxplot(data=mean_synchrony_df, x='group', y='synchrony', palette=group_palette)
    #Add swarmplot
    sns.swarmplot(data=mean_synchrony_df, x='group', y='synchrony', hue='experiment_id', palette=experiment_palette)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Group')
    plt.ylabel('Mean Synchrony')
    plt.title('Mean Synchrony by Group')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig('Mean_Synchrony_by_Group.pdf', format='pdf')
    plt.show()
    return mean_synchrony_df


def process_dataset(input_dir, output_dir):
    #Get all the folders in the input directory
    conditions = [f for f in listdir(input_dir) if isdir(join(input_dir, f))]

    #Get the yaml file location
    yaml_file = join(input_dir, [f for f in listdir(input_dir) if f.endswith('.yaml')][0])

    for condition in conditions:
        condition_dir = join(input_dir, condition)
        #Get all the experiments in the condition directory
        experiments = [f for f in listdir(condition_dir) if isdir(join(condition_dir, f))]
        for experiment in experiments:
            experiment_dir = join(condition_dir, experiment)
            #Identify the tif file
            try:
                tif_file = [f for f in listdir(experiment_dir) if f.endswith('.tif')][0]
            except:
                print(f'No tif file found in {experiment_dir}')
                continue
            #Create the corresponding output directory
            experiment_output_dir = join(output_dir, condition, experiment)
            if not os.path.exists(experiment_output_dir):
                os.makedirs(experiment_output_dir)
            #Run the fulll pipeline
            full_pipeline(join(experiment_dir, tif_file), experiment_output_dir, yaml_file)