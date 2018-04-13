import numpy as np
import os
import shutil


# ========== HYPERPARAMETERS ==========
NUM_SAMPLES_AVAILABLE_TO_MODEL = np.linspace(10,11,num=1)#np.linspace(10, 510, num=500)
NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE = 5
NUM_SYN_SAMPLES_TO_GENERATE = 25000

# ========== OUTPUT DIRECTORIES ==========
OUTPUT_DIR = 'OUTPUT/'
MODELS_OUTPUT_DIR = OUTPUT_DIR + 'MODELS/'
SYN_DATA_OUTPUT_DIR = OUTPUT_DIR + 'SYN_DATA/'
REAL_DATA_OUTPUT_DIR = OUTPUT_DIR + 'REAL_DATA/'
VISUALIZATIONS_DIR = OUTPUT_DIR + 'VISUALIZATIONS/'

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(MODELS_OUTPUT_DIR)
os.makedirs(SYN_DATA_OUTPUT_DIR)
os.makedirs(REAL_DATA_OUTPUT_DIR)
os.makedirs(VISUALIZATIONS_DIR)

# ========== RUN PIPELINE ==========
def output_dirs(dist, n, k):
    model_tag = '{0}_{1}_{2}'.format(dist, n, k)
    model_dir = '{0}{1}/'.format(MODELS_OUTPUT_DIR, model_tag)
    syn_data_dir = '{0}{1}/'.format(SYN_DATA_OUTPUT_DIR, model_tag)
    real_data_dir = '{0}{1}/'.format(REAL_DATA_OUTPUT_DIR, model_tag)
    vis_dir = '{0}{1}/'.format(VISUALIZATIONS_DIR, model_tag)
    return model_dir, syn_data_dir, real_data_dir, vis_dir

univariate_distributions = ['gaussian_0', 'gaussian_1', 'chi_square_9', 'exp_9', 'gaussian_mixture']

for dist in univariate_distributions:
    for n in NUM_SAMPLES_AVAILABLE_TO_MODEL:
        for k in range(NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE):
            # Set up output directories
            n = int(n)
            model_dir, syn_data_dir, real_data_dir, vis_dir = output_dirs(dist, n, k)

            # Build commands
            train_cmd = 'python3 train_gan.py {0} {1} {2}'.format(dist, n, model_dir)
            generate_syn_cmd = 'python3 generate_gan.py {0} {1} {2}'.format(model_dir+'generator', NUM_SYN_SAMPLES_TO_GENERATE, syn_data_dir)
            generate_real_cmd = 'python3 generate.py {0} {1} {2}'.format(dist, n, real_data_dir)

            # Run commands
            os.system(train_cmd)
            os.system(generate_syn_cmd)
            os.system(generate_real_cmd)

        # Collect results:

        # Save averaged results

        # Remove trained models

        # Remove synthetic data

        # Remove real data
