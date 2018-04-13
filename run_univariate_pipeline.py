import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


# ========== HYPERPARAMETERS ==========
NUM_SAMPLES_AVAILABLE_TO_MODEL = np.linspace(10,250,num=20)
NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE = 5
NUM_SYN_SAMPLES_TO_GENERATE = 25000
UNIVARIATE_DISTRIBUTIONS = ['gaussian_0', 'gaussian_1', 'chi_square_9', 'exp_9', 'gaussian_mixture']

# ========== OUTPUT DIRECTORIES ==========
OUTPUT_DIR = 'OUTPUT/'
MODELS_OUTPUT_DIR = OUTPUT_DIR + 'MODELS/'
SYN_DATA_OUTPUT_DIR = OUTPUT_DIR + 'SYN_DATA/'
REAL_DATA_OUTPUT_DIR = OUTPUT_DIR + 'REAL_DATA/'
POWER_DIR = OUTPUT_DIR + 'POWER/'

RESULTS_DIR = 'RESULTS/'

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(MODELS_OUTPUT_DIR)
os.makedirs(SYN_DATA_OUTPUT_DIR)
os.makedirs(REAL_DATA_OUTPUT_DIR)
os.makedirs(POWER_DIR)

os.makedirs(RESULTS_DIR)

# ========== RUN PIPELINE ==========
def output_dirs(dist, n, k):
    model_tag = '[{0}]_[n={1}]_[k={2}]'.format(dist, n, k)
    model_dir = '{0}{1}/'.format(MODELS_OUTPUT_DIR, model_tag)
    syn_data_dir = '{0}{1}/'.format(SYN_DATA_OUTPUT_DIR, model_tag)
    real_data_dir = '{0}{1}/'.format(REAL_DATA_OUTPUT_DIR, model_tag)
    return model_dir, syn_data_dir, real_data_dir

def train_and_generate_samples(num_samples_available_to_model):
    # Generate real and synthetic samples:
    for k in range(NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE):
        for dist in UNIVARIATE_DISTRIBUTIONS:
            # Set up output directories
            n = int(num_samples_available_to_model)
            model_dir, syn_data_dir, real_data_dir = output_dirs(dist, n, k)

            # Set up commands
            train_cmd = 'python3 train_gan.py {0} {1} {2}'.format(dist, n, model_dir)
            generate_syn_cmd = 'python3 generate_gan.py {0} {1} {2}'.format(model_dir+'generator', NUM_SYN_SAMPLES_TO_GENERATE, syn_data_dir)
            generate_real_cmd = 'python3 generate.py {0} {1} {2}'.format(dist, n, real_data_dir)

            # Run commands
            os.system(train_cmd)
            os.system(generate_syn_cmd)
            os.system(generate_real_cmd)

def compute_power(num_samples_available_to_model):
    # Compute power for every combination of distributions generated at this sample size:
    for i in range(len(univariate_distributions)):
        for j in range(i, len(univariate_distributions)):
            dist_1 = univariate_distributions[i]
            dist_2 = univariate_distributions[j]

            t_real_power = []
            t_syn_power = []

            for k in range(NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE):
                # Retrieve data directories
                n = int(num_samples_available_to_model)
                _, syn_data_1_dir, real_data_1_dir = output_dirs(dist_1, n, k)
                _, syn_data_2_dir, real_data_2_dir = output_dirs(dist_2, n, k)

                # Set up compute_power_cmd
                real_dataset_1 = real_data_1_dir + 'data.npy'
                syn_dataset_1 = syn_data_1_dir + 'data.npy'
                real_dataset_2 = real_data_2_dir + 'data.npy'
                syn_dataset_2 = syn_data_2_dir + 'data.npy'
                power_dir = '{0}[{1}*{2}]_[n={3}]_[k={4}]/'.format(POWER_DIR, dist_1, dist_2, n, k)
                compute_power_cmd = 'python3 compute_power.py {0} {1} {2} {3} {4}'.format(real_dataset_1, syn_dataset_1, real_dataset_2, syn_dataset_2, power_dir)

                # Run power computation
                os.system(compute_power_cmd)

                # Collect results
                results = open(power_dir+'results.txt').readlines()[0].split(',')
                t_real_power.append(float(results[0]))
                t_syn_power.append(float(results[1]))

            # Average power results across all models trained at this sample size:
            t_real_power = np.mean(t_real_power)
            t_syn_power = np.mean(t_syn_power)

            # Save power results:
            results_pth = '{0}[{1}*{2}]/'.format(RESULTS_DIR, dist_1, dist_2)
            real_results_pth = results_pth+'real.txt'
            syn_results_pth = results_pth+'syn.txt'
            if not os.path.exists(results_pth):
                os.makedirs(results_pth)
                t_real_power_for_sample_size_for_dist1_dist2 = np.array([])
                t_syn_power_for_sample_size_for_dist1_dist2 = np.array([])
            else:
                t_real_power_for_sample_size_for_dist1_dist2 = np.loadtxt(real_results_pth)
                t_syn_power_for_sample_size_for_dist1_dist2 = np.loadtxt(syn_results_pth)

            t_real_power_for_sample_size_for_dist1_dist2 = np.append(t_real_power_for_sample_size_for_dist1_dist2, t_real_power)
            t_syn_power_for_sample_size_for_dist1_dist2 = np.append(t_syn_power_for_sample_size_for_dist1_dist2, t_syn_power)

            np.savetxt(real_results_pth, t_real_power_for_sample_size_for_dist1_dist2)
            np.savetxt(syn_results_pth, t_syn_power_for_sample_size_for_dist1_dist2)

def clear_output_dirs():
    shutil.rmtree(MODELS_OUTPUT_DIR)
    shutil.rmtree(SYN_DATA_OUTPUT_DIR)
    shutil.rmtree(REAL_DATA_OUTPUT_DIR)
    shutil.rmtree(POWER_DIR)

for n in NUM_SAMPLES_AVAILABLE_TO_MODEL:
    train_and_generate_samples(n)
    compute_power(n)
    clear_output_dirs()

# ========== VISUALIZATION ==========
for i in range(len(univariate_distributions)):
    for j in range(i, len(univariate_distributions)):
        dist_1 = univariate_distributions[i]
        dist_2 = univariate_distributions[j]

        results_pth = '{0}[{1}*{2}]/'.format(RESULTS_DIR, dist_1, dist_2)
        real_results_pth = results_pth+'real.txt'
        syn_results_pth = results_pth+'syn.txt'

        t_real_power_for_sample_size_for_dist1_dist2 = np.loadtxt(real_results_pth)
        t_syn_power_for_sample_size_for_dist1_dist2 = np.loadtxt(syn_results_pth)

        plt.plot(NUM_SAMPLES_AVAILABLE_TO_MODEL, t_real_power_for_sample_size_for_dist1_dist2, label='T Test Real')
        plt.plot(NUM_SAMPLES_AVAILABLE_TO_MODEL, t_syn_power_for_sample_size_for_dist1_dist2, label='T Test Syn')
        plt.title('{0} vs {1}'.format(dist_1, dist_2))
        plt.xlabel('Real Samples')
        plt.ylabel('Power')
        plt.ylim([-0.1, 1.1])
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig('{0}real_sample_size_vs_power.png'.format(results_pth))
