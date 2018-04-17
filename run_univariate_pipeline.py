import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import shutil


# ========== HYPERPARAMETERS ==========
NUM_SAMPLES_AVAILABLE_TO_MODEL = np.geomspace(10,250,num=20)
NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE = 5
NUM_SYN_SAMPLES_TO_GENERATE = 25000
UNIVARIATE_DISTRIBUTIONS = ['gaussian_0', 'gaussian_0_1', 'gaussian_1', 'chi_square_9', 'exp_9', 'gaussian_mixture']

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
            model_dir, syn_data_dir, real_data_dir = output_dirs(dist, num_samples_available_to_model, k)

            # Set up commands
            train_cmd = 'python3 train_gan.py {0} {1} {2}'.format(dist, num_samples_available_to_model, model_dir)
            generate_syn_cmd = 'python3 generate_gan.py {0} {1} {2}'.format(model_dir+'generator', NUM_SYN_SAMPLES_TO_GENERATE, syn_data_dir)
            generate_real_cmd = 'python3 generate.py {0} {1} {2}'.format(dist, num_samples_available_to_model, real_data_dir)

            # Run commands
            os.system(train_cmd)
            os.system(generate_syn_cmd)
            os.system(generate_real_cmd)

def compute_power(num_samples_available_to_model):
    # Compute power for every combination of distributions generated at this sample size:
    for i in range(len(UNIVARIATE_DISTRIBUTIONS)):
        for j in range(i, len(UNIVARIATE_DISTRIBUTIONS)):
            dist_1 = UNIVARIATE_DISTRIBUTIONS[i]
            dist_2 = UNIVARIATE_DISTRIBUTIONS[j]

            t_real_power = []
            t_syn_power = []

            for k in range(NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE):
                # Retrieve data directories
                _, syn_data_1_dir, real_data_1_dir = output_dirs(dist_1, num_samples_available_to_model, k)
                _, syn_data_2_dir, real_data_2_dir = output_dirs(dist_2, num_samples_available_to_model, k)

                # Set up compute_power_cmd
                real_dataset_1 = real_data_1_dir + 'data.npy'
                syn_dataset_1 = syn_data_1_dir + 'data.npy'
                real_dataset_2 = real_data_2_dir + 'data.npy'
                syn_dataset_2 = syn_data_2_dir + 'data.npy'
                power_dir = '{0}[{1}*{2}]_[n={3}]_[k={4}]/'.format(POWER_DIR, dist_1, dist_2, num_samples_available_to_model, k)
                compute_power_cmd = 'python3 compute_power.py {0} {1} {2} {3} {4}'.format(real_dataset_1, syn_dataset_1, real_dataset_2, syn_dataset_2, power_dir)

                # Run power computation
                os.system(compute_power_cmd)

                # Collect results
                results = open(power_dir+'results.txt').readlines()[0].split(',')
                t_real_power.append(float(results[0]))
                t_syn_power.append(float(results[1]))

            # Save power results:
            results_pth = '{0}[{1}*{2}]/'.format(RESULTS_DIR, dist_1, dist_2)
            real_results_pth = results_pth+'real.npy'
            syn_results_pth = results_pth+'syn.npy'
            if not os.path.exists(results_pth):
                os.makedirs(results_pth)
                t_real_power_for_sample_size_for_dist1_dist2 = []
                t_syn_power_for_sample_size_for_dist1_dist2 = []
            else:
                t_real_power_for_sample_size_for_dist1_dist2 = np.load(real_results_pth).tolist()
                t_syn_power_for_sample_size_for_dist1_dist2 = np.load(syn_results_pth).tolist()

            t_real_power_for_sample_size_for_dist1_dist2.append(t_real_power)
            t_syn_power_for_sample_size_for_dist1_dist2.append(t_syn_power)

            np.save(real_results_pth, np.array(t_real_power_for_sample_size_for_dist1_dist2))
            np.save(syn_results_pth, np.array(t_syn_power_for_sample_size_for_dist1_dist2))

def clear_output_dirs():
    shutil.rmtree(MODELS_OUTPUT_DIR)
    shutil.rmtree(SYN_DATA_OUTPUT_DIR)
    shutil.rmtree(REAL_DATA_OUTPUT_DIR)
    shutil.rmtree(POWER_DIR)

for i in range(NUM_SAMPLES_AVAILABLE_TO_MODEL.shape[0]):
    n = int(NUM_SAMPLES_AVAILABLE_TO_MODEL[i])
    train_and_generate_samples(n)
    compute_power(n)
    clear_output_dirs()

# ========== VISUALIZATION ==========
for i in range(len(UNIVARIATE_DISTRIBUTIONS)):
    for j in range(i, len(UNIVARIATE_DISTRIBUTIONS)):
        dist_1 = UNIVARIATE_DISTRIBUTIONS[i]
        dist_2 = UNIVARIATE_DISTRIBUTIONS[j]

        results_pth = '{0}[{1}*{2}]/'.format(RESULTS_DIR, dist_1, dist_2)
        real_results_pth = results_pth+'real.npy'
        syn_results_pth = results_pth+'syn.npy'

        t_real_power_for_sample_size_for_dist1_dist2 = np.load(real_results_pth).T
        t_syn_power_for_sample_size_for_dist1_dist2 = np.load(syn_results_pth).T

        plt.figure()
        sns.tsplot(data=t_real_power_for_sample_size_for_dist1_dist2, time=NUM_SAMPLES_AVAILABLE_TO_MODEL, color='blue', condition='T Test Real')
        sns.tsplot(data=t_syn_power_for_sample_size_for_dist1_dist2, time=NUM_SAMPLES_AVAILABLE_TO_MODEL, color='orange', condition='T Test Syn')
        plt.title('{0} vs {1}'.format(dist_1, dist_2))
        plt.xlabel('Real Samples')
        plt.ylabel('Power')
        plt.ylim([-0.1, 1.1])
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig('{0}true_sample_size_vs_power.png'.format(results_pth))
        plt.close()
