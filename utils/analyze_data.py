import pandas as pd
import numpy as np
import os
from os.path import join, isfile
from tqdm import tqdm
import matplotlib.pyplot as plt
import colorsys
import seaborn as sns
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from utils.caiman_wrapper import *
from utils.peak_extraction import *


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


def plot_peak_amplitudes(experiments_amplitude_df, output_dir, config):
    experiments_amplitude_df = experiments_amplitude_df.copy(deep=True)
    color_options = ['#00312F', '#1D2D46', '#46000D', '#5F3920', '#573844', '#424313']
    groups = list(experiments_amplitude_df['condition'].unique())
    group_palette = {}
    experiment_palette = {}
    for i, gr in enumerate(groups):
        eids = experiments_amplitude_df.loc[experiments_amplitude_df['condition'] == gr, 'experiment_id'].unique()
        pastel_palette_arr = generate_palette(color_options[i], len(eids) + 1)
        group_palette[gr] = pastel_palette_arr[0]
        group_dict = {gr + e : pastel_palette_arr[j + 1] for j, e in enumerate(eids)}
        #concatenate group_palette and group_dict
        experiment_palette.update(group_dict)

    # Remove ROIS that are beyond 2 stds from noise mean
    mean_noise = experiments_amplitude_df['noise_level'].mean()
    std_noise = experiments_amplitude_df['noise_level'].std()
    #Get a number of unique experiment id and roi pairs before filtering
    num_rois_before_filtering = experiments_amplitude_df.set_index(['experiment_id', 'roi_id']).index.nunique()
    min_noise_thresh = mean_noise - config['peak_extraction']['num_noise_std_thresh'] * std_noise
    max_noise_thresh = mean_noise + config['peak_extraction']['num_noise_std_thresh'] * std_noise

    #Create a dataframe with ROIs that need to be removed due to high noise and associated experiment
    original_conditions = experiments_amplitude_df[['experiment_id', 'roi_id', 'condition']].drop_duplicates()
    rois_to_remove = experiments_amplitude_df.copy(deep=True)

    #The better approach:
    # rois_to_remove = rois_to_remove[(rois_to_remove['noise_level'] > max_noise_thresh) | (rois_to_remove['noise_level'] < min_noise_thresh)]

    #Only do this if you remember that including outlier low noise ROIs artificially inflates frequency!!!!
    rois_to_remove = rois_to_remove[(rois_to_remove['noise_level'] > max_noise_thresh)]

    rois_to_remove['reason'] = ['outside noise threshold' for i in range(rois_to_remove.shape[0])]
    #Remove silent ROIs (no detectable peaks found)
    if not config['peak_extraction']['include_silent_rois']:
        silent_rois = experiments_amplitude_df[experiments_amplitude_df['peak_absolute_amplitude'].isna()]
        silent_rois['reason'] = ['silent ROI' for i in range(silent_rois.shape[0])]
        rois_to_remove = pd.concat([rois_to_remove, silent_rois])
    rois_to_remove = rois_to_remove[['experiment_id', 'roi_id', 'condition', 'reason']]
    rois_to_remove = rois_to_remove.drop_duplicates()
    

    #Remove rois that match rois_to_remove from experiments_amplitude_df
    experiments_amplitude_df = experiments_amplitude_df[
        ~experiments_amplitude_df.set_index(['experiment_id', 'roi_id', 'condition']).index.isin(
            rois_to_remove.set_index(['experiment_id', 'roi_id', 'condition']).index
        )
    ].reset_index()

    num_rois_after_filtering = experiments_amplitude_df.set_index(['experiment_id', 'roi_id']).index.nunique()
    print(f"Keeping {num_rois_after_filtering} ROIs out of {num_rois_before_filtering}")
    print(f"Noise cutoffs are {min_noise_thresh} and {max_noise_thresh}")

    #For each experiment, check if it still has at least min_num_rois different ROIs
    experiments_with_enough_rois = experiments_amplitude_df.groupby(['experiment_id', 'condition']).filter(lambda x: x['roi_id'].nunique() >= config['peak_extraction']['min_num_rois'])
    #Get a list of experiments that have less than min_num_rois ROIs
    experiments_with_few_rois = experiments_amplitude_df[~experiments_amplitude_df['experiment_id'].isin(experiments_with_enough_rois['experiment_id'].unique())]
    if len(experiments_with_few_rois) > 0:
        print(f"Experiments with less than {config['peak_extraction']['min_num_rois']} ROIs: {experiments_with_few_rois['experiment_id'].unique()}")
        #If an experiment has less than min_num_rois ROIs, remove it from the dataframe
        #Add these experiments and associated ROIs to rois_to_remove
        remove_few_rois = experiments_with_few_rois[['experiment_id', 'roi_id', 'condition']]
        remove_few_rois['reason'] = [f'Less than {config['peak_extraction']['min_num_rois']} ROIs' for i in range(remove_few_rois.shape[0])]
        rois_to_remove = pd.concat([rois_to_remove, remove_few_rois], ignore_index=True)
        experiments_amplitude_df = experiments_amplitude_df[
            ~experiments_amplitude_df.set_index(['experiment_id', 'roi_id', 'condition']).index.isin(
                rois_to_remove.set_index(['experiment_id', 'roi_id', 'condition']).index
            )
        ]
    retained_conditions = experiments_amplitude_df[['experiment_id', 'roi_id', 'condition']].drop_duplicates()
    removed_conditions = original_conditions[
            ~original_conditions.set_index(['experiment_id', 'condition']).index.isin(
                retained_conditions.set_index(['experiment_id', 'condition']).index
            )
        ]
    removed_conditions['agg_condition'] = removed_conditions['experiment_id'] + removed_conditions['condition']
    print(f"Removed the following experiments completely: {removed_conditions['agg_condition'].unique()}")

    #Make a new dataframe with average values for each ROI
    experiment_avg_peak_amplitudes = experiments_amplitude_df.groupby(['experiment_id', 'condition']).agg({'peak_absolute_amplitude': 'mean'}).reset_index()
    roi_avg_peak_amplitudes = experiments_amplitude_df.groupby(['experiment_id', 'roi_id', 'condition']).agg({'peak_absolute_amplitude': 'mean'}).reset_index()

    #Sort experiments_amplitude_df and experiment_avg_peak_amplitudes to make sure control conditions are first
    experiments_amplitude_df = experiments_amplitude_df.sort_values(['condition', 'experiment_id'], ascending=False)
    experiment_avg_peak_amplitudes = experiment_avg_peak_amplitudes.sort_values(['condition', 'experiment_id'], ascending=False)
    roi_avg_peak_amplitudes = roi_avg_peak_amplitudes.sort_values(['condition', 'experiment_id'], ascending=False)

    #Use pingouin to run ANOVA on plot_peak_amplitudes to check if there are significant differences between groups
    aov = pg.anova(data=experiment_avg_peak_amplitudes, dv='peak_absolute_amplitude', between='condition', detailed=True)
    #Save aov as a csv
    aov.to_csv(join(output_dir, 'ANOVA_peak_absolute_amplitude.csv'))

    # Check if ANOVA is significant
    if aov['p-unc'][0] < 0.05:  # Assuming significance level of 0.05
        
        # Perform Tukey HSD test
        tukey = pairwise_tukeyhsd(
            endog=experiment_avg_peak_amplitudes['peak_absolute_amplitude'],
            groups=experiment_avg_peak_amplitudes['condition'],
            alpha=0.05
        )
        # Save Tukey HSD results to a text file
        with open(join(output_dir, 'Tukey_peak_absolute_amplitude.txt'), 'w') as f:
            f.write(str(tukey))

    # Create the plot
    plt.figure(figsize=(len(groups), 4))
    #Make the plot dark
    plt.style.use('dark_background')
    # Boxplot with group colors
    sns.boxplot(data=experiment_avg_peak_amplitudes, x='condition', y='peak_absolute_amplitude', palette=group_palette)
    # Swarmplot with experiment_id colors
    experiment_avg_peak_amplitudes['condition_experiment_id'] = experiment_avg_peak_amplitudes['condition'] + experiment_avg_peak_amplitudes['experiment_id']
    sns.swarmplot(data=experiment_avg_peak_amplitudes, x='condition', y='peak_absolute_amplitude', hue='condition_experiment_id', palette=experiment_palette)
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
    plt.savefig(join(output_dir, 'Peak_Absolute_Amplitude_by_Group.pdf'), format='pdf')

    #Use seaborn to plot a violin plot of the peak relative amplitudes
    plt.figure(figsize=(len(groups), 6))
    sns.violinplot(data=roi_avg_peak_amplitudes, x='condition', y='peak_absolute_amplitude', palette=group_palette)
    #Add swarmplot
    if len(groups) <= 2:
        roi_avg_peak_amplitudes['condition_experiment_id'] = roi_avg_peak_amplitudes['condition'] + roi_avg_peak_amplitudes['experiment_id']
        sns.swarmplot(data=roi_avg_peak_amplitudes, x='condition', y='peak_absolute_amplitude', hue='condition_experiment_id', palette=experiment_palette)
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
    plt.savefig(join(output_dir,'Peak_Absolute_Amplitude_by_ROI.pdf'), format='pdf')

    #Use seaborn to plot a violin plot of the peak relative amplitudes
    plt.figure(figsize=(len(groups), 4))
    sns.violinplot(data=experiments_amplitude_df, x='condition', y='peak_absolute_amplitude', palette=group_palette)
    #Add swarmplot
    # sns.swarmplot(data=experiments_amplitude_df, x='group', y='peak_absolute_amplitude', hue='experiment_id')
    plt.xlabel('Group')
    plt.ylabel('Peak Absolute Amplitude')
    plt.title('Peak Absolute Amplitude by Group')
    plt.tight_layout()
    #Seve as pdf
    plt.savefig(join(output_dir, 'Peak_Absolute_Amplitude.pdf'), format='pdf')

    #Save all the dataframes to csvs
    experiments_amplitude_df.to_csv(join(output_dir, 'experiments_amplitude_df.csv'), index=False)
    experiment_avg_peak_amplitudes.to_csv(join(output_dir, 'experiment_avg_peak_amplitudes.csv'), index=False)
    roi_avg_peak_amplitudes.to_csv(join(output_dir, 'roi_avg_peak_amplitudes.csv'), index=False)
    rois_to_remove.to_csv(join(output_dir, 'rois_to_remove.csv'), index=False)

    return rois_to_remove, group_palette, experiment_palette, experiment_avg_peak_amplitudes


def plot_frequencies(experiments_frequency_df, rois_to_remove, group_palette, experiment_palette, output_dir):
    experiments_frequency_df = experiments_frequency_df.copy(deep=True)
    groups = list(experiments_frequency_df['condition'].unique())

    #Remove rois that match rois_to_remove from experiments_frequency_df
    experiments_frequency_df = experiments_frequency_df[
        ~experiments_frequency_df.set_index(['experiment_id', 'roi_id', 'condition']).index.isin(
            rois_to_remove.set_index(['experiment_id', 'roi_id', 'condition']).index
        )
    ]

    #Make a new dataframe with average values for each experiment
    experiment_avg_firing_frequency = experiments_frequency_df.groupby(['experiment_id', 'condition']).agg({'mean_firing_frequency[Hz]': 'mean', 'mean_peak_to_peak_distance[ms]' : 'mean'}).reset_index()

    #Sort experiment_avg_firing_frequency and experiments_frequency_df to make sure control conditions are first
    experiment_avg_firing_frequency = experiment_avg_firing_frequency.sort_values(['condition', 'experiment_id'], ascending=False)
    experiments_frequency_df = experiments_frequency_df.sort_values(['condition', 'experiment_id'], ascending=False)

    #Use pingouin to run ANOVA on experiment_avg_firing_frequency to check if there are significant differences between groups
    aov = pg.anova(data=experiment_avg_firing_frequency, dv='mean_firing_frequency[Hz]', between='condition', detailed=True)
    aov.to_csv(join(output_dir, 'ANOVA_mean_firing_frequency.csv'))

    # Check if ANOVA is significant
    if aov['p-unc'][0] < 0.05:  # Assuming significance level of 0.05
        
        # Perform Tukey HSD test
        tukey = pairwise_tukeyhsd(
            endog=experiment_avg_firing_frequency['mean_firing_frequency[Hz]'],
            groups=experiment_avg_firing_frequency['condition'],
            alpha=0.05
        )
        # Save Tukey HSD results to a text file
        with open(join(output_dir, 'Tukey_mean_firing_frequency.txt'), 'w') as f:
            f.write(str(tukey))

    #Use seaborn to plot a boxplot of the peak absolute amplitudes
    plt.figure(figsize=(len(groups), 4))
    sns.boxplot(data=experiment_avg_firing_frequency, x='condition', y='mean_firing_frequency[Hz]', palette=group_palette)
    #Add swarmplot
    experiment_avg_firing_frequency['condition_experiment_id'] = experiment_avg_firing_frequency['condition'] + experiment_avg_firing_frequency['experiment_id']
    sns.swarmplot(data=experiment_avg_firing_frequency, x='condition', y='mean_firing_frequency[Hz]', hue='condition_experiment_id', palette=experiment_palette)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Experiment ID')
    plt.ylabel('Mean Firing Frequency [Hz]')
    plt.title('Mean Firing Frequency by Experiment')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig(join(output_dir, 'Mean_Firing_Frequency_by_Experiment.pdf'), format='pdf')

    #Use seaborn to plot a boxplot of the peak absolute amplitudes
    plt.figure(figsize=(len(groups), 4))
    sns.boxplot(data=experiment_avg_firing_frequency, x='condition', y='mean_peak_to_peak_distance[ms]', palette=group_palette)
    #Add swarmplot
    sns.swarmplot(data=experiment_avg_firing_frequency, x='condition', y='mean_peak_to_peak_distance[ms]', hue='condition_experiment_id', palette=experiment_palette)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Experiment ID')
    plt.ylabel('Mean peak to peak distance [ms]')
    plt.title('Mean Firing Frequency by Experiment')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig(join(output_dir, 'Mean_Peak_to_Peak_by_Experiment.pdf'), format='pdf')

    #Use seaborn to plot a violin plot of the firing frequency with individual ROIs
    plt.figure(figsize=(len(groups), 4))
    sns.violinplot(data=experiments_frequency_df, x='condition', y='mean_firing_frequency[Hz]', palette=group_palette)
    #Add swarmplot
    # sns.swarmplot(data=experiments_frequency_df, x='condition', y='mean_firing_frequency[Hz]', hue='experiment_id', palette=experiment_palette)
    #put legend outside the plot
    plt.xlabel('Group')
    plt.ylabel('Mean Firing Frequency [Hz]')
    plt.title('Mean Firing Frequency by Group')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig(join(output_dir, 'Mean_Firing_Frequency.pdf'), format='pdf')

    #Use seaborn to plot a violin plot of the firing frequency with individual ROIs
    plt.figure(figsize=(4, 4))
    sns.violinplot(data=experiments_frequency_df, x='condition', y='mean_peak_to_peak_distance[ms]', palette=group_palette)
    #Add swarmplot
    # sns.swarmplot(data=experiments_frequency_df, x='condition', y='mean_firing_frequency[Hz]', hue='experiment_id', palette=experiment_palette)
    #put legend outside the plot
    plt.xlabel('Group')
    plt.ylabel('Mean peak to peak distance [ms]')
    plt.title('Mean Firing Frequency by Group')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig(join(output_dir, 'Mean_Peak_to_Peak.pdf'), format='pdf')

    #Save all the dataframes to csvs
    experiments_frequency_df.to_csv(join(output_dir, 'experiments_frequency_df.csv'), index=False)
    experiment_avg_firing_frequency.to_csv(join(output_dir, 'experiment_avg_firing_frequency.csv'), index=False)

    return experiment_avg_firing_frequency


def plot_synchrony(experimets_synchrony_df, rois_to_remove, group_palette, experiment_palette, output_dir):
    experimets_synchrony_df = experimets_synchrony_df.copy(deep=True)
    groups = list(experimets_synchrony_df['condition'].unique())

    #If either of the ROIs is in the list of ROIs to remove, remove the row
    experimets_synchrony_df = experimets_synchrony_df[
        ~experimets_synchrony_df.set_index(['experiment_id', 'ROI_a', 'condition']).index.isin(
            rois_to_remove.set_index(['experiment_id', 'roi_id', 'condition']).index
        )
    ]
    experimets_synchrony_df = experimets_synchrony_df[
        ~experimets_synchrony_df.set_index(['experiment_id', 'ROI_b', 'condition']).index.isin(
            rois_to_remove.set_index(['experiment_id', 'roi_id', 'condition']).index
        )
    ]

    #Calculate mean synchrony for each experiment
    mean_synchrony_df = experimets_synchrony_df.groupby(['experiment_id', 'condition']).agg({'synchrony': 'mean'}).reset_index()

    #sort the dataframe by condition
    mean_synchrony_df = mean_synchrony_df.sort_values(['condition', 'experiment_id'], ascending=False)

    #Use pingouin to run ANOVA on mean_synchrony_df to check if there are significant differences between groups
    aov = pg.anova(data=mean_synchrony_df, dv='synchrony', between='condition', detailed=True)
    aov.to_csv(join(output_dir, 'ANOVA_synchrony.csv'))

    # Check if ANOVA is significant
    if aov['p-unc'][0] < 0.05:  # Assuming significance level of 0.05
        
        # Perform Tukey HSD test
        tukey = pairwise_tukeyhsd(
            endog=mean_synchrony_df['synchrony'],
            groups=mean_synchrony_df['condition'],
            alpha=0.05
        )
        # Save Tukey HSD results to a text file
        with open(join(output_dir, 'Tukey_synchrony.txt'), 'w') as f:
            f.write(str(tukey))

    #Use seaborn to plot a boxplot of the synchrony
    plt.figure(figsize=(4, 4))
    sns.boxplot(data=mean_synchrony_df, x='condition', y='synchrony', palette=group_palette)
    #Add swarmplot
    mean_synchrony_df['condition_experiment_id'] = mean_synchrony_df['condition'] + mean_synchrony_df['experiment_id']
    sns.swarmplot(data=mean_synchrony_df, x='condition', y='synchrony', hue='condition_experiment_id', palette=experiment_palette)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Group')
    plt.ylabel('Mean Synchrony')
    plt.title('Mean Synchrony by Group')
    if len(groups) > 2:
        plt.legend().remove()
    plt.tight_layout()
    #Seve as pdf
    plt.savefig(join(output_dir, 'Mean_Synchrony_by_Group.pdf'), format='pdf')

    #Save all the dataframes to csvs
    experimets_synchrony_df.to_csv(join(output_dir, 'experimets_synchrony_df.csv'), index=False)
    mean_synchrony_df.to_csv(join(output_dir, 'mean_synchrony_df.csv'), index=False)
    
    return mean_synchrony_df


def aggregate_data(output_dir):
    #Compile a dataframe with all experiments
    experiments_amplitude_df = pd.DataFrame(columns=['experiment_id'])
    experiments_frequency_df = pd.DataFrame(columns=['experiment_id'])
    experimets_synchrony_df = pd.DataFrame(columns=['experiment_id'])

    #Get folder names in output_dir directory
    conditions = [f for f in os.listdir(output_dir) if os.path.isdir(join(output_dir, f))]
    experiments_tuples = []
    for condition in conditions:
        condition_path = join(output_dir, condition)
        experiment_ids = [f for f in os.listdir(condition_path) if os.path.isdir(join(condition_path, f))]
        for experiment_id in experiment_ids:
            experiment_path = join(condition_path, experiment_id)
            experiment_files = [f for f in os.listdir(experiment_path) if isfile(join(experiment_path, f))]
            for f in experiment_files:
                #Only add files if yaml file status_message field says "Full pipeline success"
                if f.endswith('.yaml'):
                    output_config = load_yaml_config(join(experiment_path, f))
                    if output_config['status_message'] == 'Full pipeline success':
                        experiments_tuples.append([condition, experiment_id])

    for condition, experiment_id in tqdm(experiments_tuples):
        experiment_path = join(output_dir, condition, experiment_id)
        #Find _experiment_df.csv file
        experiment_df_loc = [join(experiment_path, f) for f in os.listdir(experiment_path) if f.endswith('_experiment_df.csv')][0]
        #Find _firing_frequency.csv file
        experiment_firing_frequency_loc = [join(experiment_path, f) for f in os.listdir(experiment_path) if f.endswith('_firing_frequency.csv')][0]
        #Find _aggregated_corr.csv file
        experiment_aggregated_corr_loc = [join(experiment_path, f) for f in os.listdir(experiment_path) if f.endswith('_aggregated_corr.csv')][0]

        #load the experiment_df
        experiment_df = pd.read_csv(experiment_df_loc)
        #load the experiment_firing_frequency
        experiment_firing_frequency = pd.read_csv(experiment_firing_frequency_loc)
        #Load correlation into numpy array without row/column names
        experiment_aggregated_corr = pd.read_csv(experiment_aggregated_corr_loc).to_numpy()
                                        
        experiment_df['experiment_id'] = experiment_id
        experiment_df['condition'] = condition
        experiments_amplitude_df = pd.concat([experiments_amplitude_df, experiment_df], ignore_index=True)

        experiment_firing_frequency['experiment_id'] = experiment_id
        experiment_firing_frequency['condition'] = condition
        experiments_frequency_df = pd.concat([experiments_frequency_df, experiment_firing_frequency], ignore_index=True)

        single_experiment_synchrony = pd.DataFrame(columns = ['experiment_id', 'ROI_a', 'ROI_b', 'synchrony'])
        #Flatten numpy array into a single dimension
        experiment_aggregated_corr_1d = experiment_aggregated_corr.flatten()
        #Create an array of ROI pairs corresponding to flattened array
        roi_pairs = np.array([['ROI_'+str(i), 'ROI_'+str(j)] for i in range(experiment_aggregated_corr.shape[0]) for j in range(experiment_aggregated_corr.shape[0])])
        single_experiment_synchrony['experiment_id'] = [experiment_id for _ in range(len(experiment_aggregated_corr_1d))]
        single_experiment_synchrony['ROI_a'] = roi_pairs[:, 0]
        single_experiment_synchrony['ROI_b'] = roi_pairs[:, 1]
        single_experiment_synchrony['synchrony'] = experiment_aggregated_corr_1d
        single_experiment_synchrony['condition'] = condition
        experimets_synchrony_df = pd.concat([experimets_synchrony_df, single_experiment_synchrony], ignore_index=True)  
    return [experiments_amplitude_df, experiments_frequency_df, experimets_synchrony_df]