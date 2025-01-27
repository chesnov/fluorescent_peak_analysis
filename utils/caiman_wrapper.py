import numpy as np
import yaml
import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf, params
from skimage import measure
from skimage.draw import polygon
import seaborn as sns
import numpy as np
from scipy.ndimage.filters import percentile_filter

#load txt files as csv into numpy arrays
import numpy as np
import kaleido #required
kaleido.__version__ #0.2.1

import plotly
plotly.__version__ #5.5.0

#now this works:
import os
from os.path import join

import seaborn as sns
import matplotlib.pyplot as plt

import cv2
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

try:
    cv2.setNumThreads(0)
except():
    pass

import bokeh.plotting as bpl
import cv2
import holoviews as hv
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import os

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
        # get_ipython().run_line_magic('matplotlib', 'qt')  #uncomment to run in qt mode
except NameError:
    pass

try:
    cv2.setNumThreads(0)
except:
    pass

bpl.output_notebook()
hv.notebook_extension('bokeh')
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

pipeline_version = '0.1.0'

def load_yaml_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return config


# Function to set up a cluster for parallel processing
def setup_cluster():
    _, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', n_processes=None, ignore_preexisting=False);
    return cluster, n_processes


# Function to perform motion correction
def perform_motion_correction(movie_path, cluster, parameters, pw_rigid):
    mot_correct = MotionCorrect([movie_path], dview=cluster, **parameters.get_group('motion'));
    mot_correct.motion_correct(save_movie=True);
    fname_mc = mot_correct.fname_tot_els if pw_rigid else mot_correct.fname_tot_rig
    if pw_rigid:
        bord_px = np.ceil(np.maximum(np.max(np.abs(mot_correct.x_shifts_els)),
                                      np.max(np.abs(mot_correct.y_shifts_els)))).astype(int)
    else:
        bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)
    return fname_mc, bord_px


# Function to load memory-mapped file
def load_memmap(fname_mc, bord_px):
    fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C', border_to_0=bord_px)
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')
    return images


# Function to extract ROIs using CNMF-E
def extract_rois(images, cluster, parameters, n_processes):
    cnmfe_model = cnmf.CNMF(n_processes=n_processes, dview=cluster, params=parameters);
    cnmfe_model.fit(images);
    return cnmfe_model


# Function to evaluate and retain components
def evaluate_components(cnmfe_model, images, cluster, config):
    cnmfe_model.params.change_params(params_dict={'min_SNR': config['cnmfe_params']['min_SNR'], 
                                                  'rval_thr' : config['cnmfe_params']['rval_thr'], 
                                                  'use_cnn': config['cnmfe_params']['use_cnn']})
    cnmfe_model.estimates.evaluate_components(images, cnmfe_model.params, dview=cluster)
    retained_components = cnmfe_model.estimates.idx_components
    return retained_components, cnmfe_model


# Function to generate masks and contours
def generate_masks_and_contours(A, parameters, retained_components, config):
    A2 = A.toarray().reshape(parameters.data['dims'] + (-1,), order='F').transpose([2, 0, 1])
    contours, new_mask = [], np.zeros_like(A2)
    is_to_delete = []
    for i in range(A2.shape[0]):
        if i in retained_components:
            raw_contours = measure.find_contours(A2[i], level=config['roi_contours']['level'])
            if len(raw_contours) == 0:
                is_to_delete.append(i)
                continue
            raw_contours = raw_contours[0]
            # Extract x and y coordinates
            r = raw_contours[:, 0]  # Row indices
            c = raw_contours[:, 1]  # Column indices
            
            # Convert contour into a filled polygon
            rr, cc = polygon(r, c, A2[0].shape)
            new_mask[i, rr, cc] = True

            #Calculate percentage of pixels for this component in A2 that are >0
            percentage_nonzero = np.count_nonzero(A2[i] > 0) / A2[i].size
            if percentage_nonzero < config['roi_contours']['percentage_nonzero']: #Assume that very large components are not valid         
                # Add each contour as an array to the list
                contours.extend([raw_contours])
            else:
                is_to_delete.append(i)
        else:
            is_to_delete.append(i)
    new_mask = np.delete(new_mask, is_to_delete, axis=0)
    return new_mask, contours

# Function to extract fluorescence traces
def extract_traces(images, new_mask):
    traces = []
    for i in range(new_mask.shape[0]):
        roi_mask = new_mask[i].astype(bool)
        if np.any(roi_mask):
            trace = images[:, roi_mask].mean(axis=1)
            traces.append(trace)
    return np.array(traces)


def detrend_df_f_rois(traces, outpath, quantile_min=8, frames_window=250):
    """
    Compute DF/F signal for extracted ROI traces.

    Args:
        traces: ndarray
            Fluorescence signals (n_rois x n_frames).
        
        quantile_min: float
            Quantile used to estimate the baseline (values in [0, 100]).
        
        frames_window: int
            Number of frames for computing running quantile.

    Returns:
        F_df: ndarray
            Detrended ΔF/F for each ROI (n_rois x n_frames).
    """
    _, n_frames = traces.shape

    # Compute the fluorescence baseline using a moving quantile
    if frames_window is None or frames_window > n_frames:
        # Fixed quantile across all frames
        baseline = np.percentile(traces, quantile_min, axis=1)
        baseline = baseline[:, None]  # Expand dimensions for broadcasting
    else:
        # Exact moving quantile
        baseline = np.array([
            percentile_filter(trace, quantile_min, size=frames_window)
            for trace in traces
        ])

    # Subtract the baseline (detrend)
    F_detrended = traces - baseline
    F_df = F_detrended / baseline

    #Export the traces to a csv file
    np.savetxt(outpath, F_df, delimiter=',')

    return F_df


def save_average_image_with_roi_contours(CI, contours, pdf_save_path):
    """
    Save an average image with ROI contours overlaid in different colors as a PDF.

    Returns:
        None. Saves the PDF to the same directory as the .tif file.
    """
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display the average image
    ax.imshow(CI, cmap="gray", interpolation="nearest")
    ax.set_title("Average Image with ROI Contours")
    ax.axis("off")

    colors = sns.color_palette("hsv", len(contours))
    np.random.shuffle(colors)
    
    for i, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color=colors[i])
        #Find rightmost point of the contour
        cont_right = np.argmax(contour[:, 1])
        #Add a label with the ROI number to the rightmost center point
        ax.text(contour[cont_right, 1], contour[cont_right, 0], str(i), color=colors[i], fontsize=6)
    
    # Save the plot as a PDF
    plt.tight_layout()
    plt.savefig(pdf_save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)  # Close the plot to free memory


#A function that saves all the settings used into a yaml file
def save_settings_to_yaml(yaml_file, settings_dict):
    with open(yaml_file, "w") as file:
        yaml.dump(settings_dict, file)


# Main processing workflow
def raw_data_to_df_f(movie_path, yaml_file, outdir, experiment_id):
    filename = Path(movie_path).stem
    config = load_yaml_config(yaml_file)
    cluster, n_processes = setup_cluster()

    mc_dict = {
        'fnames': [movie_path],
        'fr': config['motion_corr']['fr'],
        'decay_time': config['motion_corr']['decay_time'],
        'pw_rigid': config['motion_corr']['pw_rigid'],
        'max_shifts': (config['motion_corr']['max_shifts'], config['motion_corr']['max_shifts']),
        'gSig_filt': (config['motion_corr']['gSig_filt'], config['motion_corr']['gSig_filt']),
        'strides': (config['motion_corr']['strides'], config['motion_corr']['strides']),
        'overlaps': (config['motion_corr']['overlaps'], config['motion_corr']['overlaps']),
        'max_deviation_rigid': config['motion_corr']['max_deviation_rigid'],
        'border_nan': config['motion_corr']['border_nan']
    }

    parameters = params.CNMFParams(params_dict=mc_dict);
    fname_mc, bord_px = perform_motion_correction(movie_path, cluster, parameters, config['motion_corr']['pw_rigid'])
    images = load_memmap(fname_mc, bord_px);

    parameters.change_params(params_dict={'method_init': config['src_extr_deconv']['method_init'],  # use this for 1 photon
                                'K': config['src_extr_deconv']['K'],
                                'gSig': np.array([config['src_extr_deconv']['gSig'], config['src_extr_deconv']['gSig']]),
                                'gSiz': 2*np.array([config['src_extr_deconv']['gSig'], config['src_extr_deconv']['gSig']]) + 1, # half-width of bounding box created around neurons during initialization
                                'merge_thr': config['src_extr_deconv']['merge_thr'],
                                'p': config['src_extr_deconv']['p'],
                                'tsub': config['src_extr_deconv']['tsub'],
                                'ssub': config['src_extr_deconv']['ssub'],
                                'rf': config['src_extr_deconv']['rf'],
                                'stride': config['src_extr_deconv']['stride'],
                                'only_init': config['src_extr_deconv']['only_init'], # set it to True to run CNMF-E
                                'nb': config['src_extr_deconv']['nb'],
                                'nb_patch': config['src_extr_deconv']['nb_patch'],
                                'method_deconvolution': config['src_extr_deconv']['method_deconvolution'], # could use 'cvxpy' alternatively
                                'low_rank_background': config['src_extr_deconv']['low_rank_background'],
                                'update_background_components': config['src_extr_deconv']['update_background_components'], # sometimes setting to False improve the results
                                'min_corr': config['src_extr_deconv']['min_corr'],
                                'min_pnr': config['src_extr_deconv']['min_pnr'],
                                'normalize_init': config['src_extr_deconv']['normalize_init'], # just leave as is
                                'center_psf': config['src_extr_deconv']['center_psf'], # True for 1p
                                'ssub_B': config['src_extr_deconv']['ssub_B'],
                                'ring_size_factor': config['src_extr_deconv']['ring_size_factor'],
                                'del_duplicates': config['src_extr_deconv']['del_duplicates'], # whether to remove duplicates from initialization
                                'border_pix': bord_px}); # number of pixels to not consider in the borders)

    cnmfe_model = extract_rois(images, cluster, parameters, n_processes);

    # Visualize results
    CI = cm.local_correlations(images.transpose(1, 2, 0))

    #Check if there is no components
    if len(cnmfe_model.estimates.C) == 0:
        contours = []
        retained_components = []
        status_massage = 'No components were extracted'
    else:
        retained_components, cnmfe_model = evaluate_components(cnmfe_model, images, cluster, config)
    
    #Check if "retained_components" variable exists
    if len(retained_components) == 0:
        contours = []
        status_massage = 'No components were retained'
    else:
        new_mask, contours = generate_masks_and_contours(cnmfe_model.estimates.A, parameters, retained_components, config)

    #Create a dictionary with the settings used
    settings_dict = config
    settings_dict['pipeline_version'] = pipeline_version
    settings_dict['retained_components'] = len(retained_components)
    settings_dict['input_movie'] = movie_path
    settings_dict['output_dir'] = outdir

    save_average_image_with_roi_contours(CI, contours, pdf_save_path=join(outdir, experiment_id + "_roi_contours.pdf"))

    # Stop cluster
    cm.stop_server(dview='cluster')

    if len(contours) == 0:
        print("No contours were extracted")
        status_massage = 'No contours were extracted'
    else:
        traces = extract_traces(images, new_mask)
        detrend_df_f_rois(traces, outpath=join(outdir, experiment_id + "_DFF_traces.csv"))
        status_massage = 'Successful ROI extraction'
        print(f"Retained {len(traces)} components")
    
    settings_dict['status_message'] = status_massage

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

    return settings_dict
