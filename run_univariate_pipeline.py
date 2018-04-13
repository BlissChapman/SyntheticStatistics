import numpy as np
import os
import shutil


# ========== HYPERPARAMETERS ==========
NUM_SAMPLES_AVAILABLE_TO_MODEL = np.linspace(10,11,num=1)#np.linspace(10, 510, num=500)
NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE = 1
NUM_SYN_SAMPLES_TO_GENERATE = 25000

# ========== OUTPUT DIRECTORIES ==========
OUTPUT_DIR = 'OUTPUT/'
MODELS_OUTPUT_DIR = OUTPUT_DIR + 'MODELS/'
SYN_DATA_OUTPUT_DIR = OUTPUT_DIR + 'SYN_DATA/'
REAL_DATA_OUTPUT_DIR = OUTPUT_DIR + 'REAL_DATA/'
POWER_DIR = OUTPUT_DIR + 'POWER/'

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(MODELS_OUTPUT_DIR)
os.makedirs(SYN_DATA_OUTPUT_DIR)
os.makedirs(REAL_DATA_OUTPUT_DIR)
os.makedirs(POWER_DIR)

# ========== RUN PIPELINE ==========
def output_dirs(dist, n, k):
    model_tag = '[{0}]_[n={1}]_[k={2}]'.format(dist, n, k)
    model_dir = '{0}{1}/'.format(MODELS_OUTPUT_DIR, model_tag)
    syn_data_dir = '{0}{1}/'.format(SYN_DATA_OUTPUT_DIR, model_tag)
    real_data_dir = '{0}{1}/'.format(REAL_DATA_OUTPUT_DIR, model_tag)
    # results_dir = '{0}{1}/'.format(RESULTS_DIR, model_tag)
    return model_dir, syn_data_dir, real_data_dir

univariate_distributions = ['gaussian_0', 'gaussian_1', 'chi_square_9', 'exp_9', 'gaussian_mixture']

for n in NUM_SAMPLES_AVAILABLE_TO_MODEL:
    # Generate real and synthetic samples:
    for k in range(NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE):
        for dist in univariate_distributions:
            # Set up output directories
            n = int(n)
            model_dir, syn_data_dir, real_data_dir = output_dirs(dist, n, k)

            # Set up commands
            train_cmd = 'python3 train_gan.py {0} {1} {2}'.format(dist, n, model_dir)
            generate_syn_cmd = 'python3 generate_gan.py {0} {1} {2}'.format(model_dir+'generator', NUM_SYN_SAMPLES_TO_GENERATE, syn_data_dir)
            generate_real_cmd = 'python3 generate.py {0} {1} {2}'.format(dist, n, real_data_dir)

            # Run commands
            os.system(train_cmd)
            os.system(generate_syn_cmd)
            os.system(generate_real_cmd)


    # Compute power for every combination of distributions generated at this sample size:
    for dist_1 in univariate_distributions:
        for dist_2 in univariate_distributions:
            for k in range(NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE):
                # Retrieve data directories
                n = int(n)
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

    # Collect results:

    # Save averaged results

    # Remove trained models

    # Remove synthetic data

    # Remove real data
