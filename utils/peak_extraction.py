import numpy as np
import sys
from contextlib import contextmanager

import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import caiman as cm

#load txt files as csv into numpy arrays
import kaleido #required
kaleido.__version__ #0.2.1

import plotly
plotly.__version__ #5.5.0

import plotly.graph_objects as go
from scipy.signal import find_peaks
import os
from os.path import isdir, join
from os import listdir
import pandas as pd

import cv2

try:
    cv2.setNumThreads(0)
except():
    pass

from utils.analyze_data import *
from utils.caiman_wrapper import *


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


# Define the task wrapper at the module level
def task_wrapper(target_function, *args, **kwargs):
    return target_function(*args, **kwargs)

def run_with_timeout(target_function, timeout=300, *args, **kwargs):
    """
    Run a function with a timeout. If it exceeds the timeout, stop the caiman server and restart the function.

    Args:
        target_function (callable): The function to execute.
        timeout (int): The maximum time (in seconds) to allow the function to run.
        *args: Positional arguments to pass to the target_function.
        **kwargs: Keyword arguments to pass to the target_function.

    Returns:
        result: The result of the target function if it completes in time.
    """
    while True:  # Keep retrying until the function completes successfully
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                # Pass the function and its arguments to the executor
                future = executor.submit(task_wrapper, target_function, *args, **kwargs)
                
                # Wait for the task to complete or timeout
                result = future.result(timeout=timeout)
                return result  # If successful, return the result

        except multiprocessing.TimeoutError:
            print(f"Function timed out after {timeout} seconds. Restarting...")
            # Stop the caiman cluster
            cm.stop_server(dview='cluster')
            time.sleep(1)  # Optional pause before restarting

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            cm.stop_server(dview='cluster')
            raise  # Re-raise the exception for debugging

            
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


def precise_peak_locs(smoothed_peaks, trace):
    heights = []
    peaks = []
    for peak in smoothed_peaks:
        #smoothed peak location might not the actual peak due to smoothing
        #Get the position within original trace corresponding to the largest value in the window
        true_peak = np.argmax(trace[peak-15:peak+15]) + peak - 15 #Due to smoothing the peak is shifted to the right
        peaks.append(true_peak)
        absolute_height = trace[true_peak]
        heights.append(absolute_height)
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
    peaks, heights = precise_peak_locs(smoothed_peaks, trace)
    
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

    #Check if this experiment has already been processed
    if os.path.exists(join(experiment_outdir, experiment_id + '_settings.yaml')):
        print(f'{experiment_id} has already been processed')
        return

    # settings_dict =  raw_data_to_df_f(experiment_tif, yaml_file, experiment_outdir, experiment_id)
    with suppress_output():
        try:
            settings_dict = run_with_timeout(raw_data_to_df_f, 300, experiment_tif, yaml_file, experiment_outdir, experiment_id)
        except Exception:
            print(f"The following exception occurred during processing of {experiment_id}: {Exception}")

    #Clean up all temp files in caiman temp folder
    if "CAIMAN_DATA" in os.environ: 
        caiman_dir =  join(os.environ["CAIMAN_DATA"], 'temp')
    else: 
        caiman_dir = join(os.path.expanduser("~"), "caiman_data", "temp")
    for file in os.listdir(caiman_dir):
        file_path = join(caiman_dir, file)
        #Delete the file
        if os.path.isfile(file_path):
            os.remove(file_path)
            
    #Check if any ROIs were extracted
    if settings_dict['status_message']!='Successful ROI extraction':
        save_settings_to_yaml(join(experiment_outdir, experiment_id + "_settings.yaml"), settings_dict)
    
    #Identify csv file in outdir
    try:
        csv_file = [f for f in os.listdir(experiment_outdir) if f.endswith('.csv')][0]  
    except:
        print(f'No ROIs extracted from {experiment_id}')
        return
    array, df = load_data(join(experiment_outdir, csv_file))

    roi_ids = list(df.columns)

    experiment_df_columns = ['roi_id', 'peak_time', 'peak_absolute_amplitude']
    experiment_df = pd.DataFrame(columns=experiment_df_columns)

    for roi, roi_id in enumerate(roi_ids):
        trace = array[:, roi] #selece just one roi 
        peaks, peak_heights, height_threshold, smoothed_trace = peak_finder(trace)
        if len(peaks) == 0:
            peaks = [None]
            peak_heights = np.array([None])
        
        #Add peaks to experiment_df
        roi_df = pd.DataFrame(columns=experiment_df_columns)
        roi_df['roi_id'] = [roi_id for _ in range(len(peaks))]
        roi_df['peak_time'] = peaks
        roi_df['peak_absolute_amplitude'] = peak_heights

        experiment_df = pd.concat([experiment_df, roi_df], ignore_index=True)
        plot_peaks(roi_id, trace, smoothed_trace, peaks, peak_heights, height_threshold, experiment_outdir)

    experiment_df['noise_level'] = experiment_df['roi_id'].apply(lambda x: calc_noise_levels(array, int(x.rsplit('ROI_')[1])))
    plot_noise_level_histogram(experiment_df, experiment_outdir)

    #Save experiment_df as csv
    experiment_df.to_csv(os.path.join(experiment_outdir, experiment_id + '_experiment_df.csv'), index=False)

    recordings_duration = len(array) / 1000 #in seconds
    peak_to_peak_distance(experiment_df, recordings_duration, experiment_outdir, experiment_id)

    synchrony_calculation(experiment_df, array, experiment_outdir, experiment_id)

    settings_dict['status_message'] = 'Full pipeline success'
    save_settings_to_yaml(join(experiment_outdir, experiment_id + "_settings.yaml"), settings_dict)


def process_dataset(input_dir, output_dir):
    set_seed()
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

    #Analyse all the processed data
    experiments_amplitude_df, experiments_frequency_df, experimets_synchrony_df = aggregate_data(output_dir)

    rois_to_remove, group_palette, experiment_palette, experiment_avg_peak_amplitudes = plot_peak_amplitudes(experiments_amplitude_df, output_dir)
    experiment_avg_firing_frequency = plot_frequencies(experiments_frequency_df, rois_to_remove, group_palette, experiment_palette, output_dir)
    mean_synchrony_df = plot_synchrony(experimets_synchrony_df, rois_to_remove, group_palette, experiment_palette, output_dir)