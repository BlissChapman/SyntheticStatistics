import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import shutil

from utils.multiple_comparison import multivariate_power_calculation

# ========== HYPERPARAMETERS ==========
# ***RESEARCHER BEWARE***
# Total number of GANs =
#   |NUM_SAMPLES_AVAILABLE_TO_MODEL| * NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE * |MULTIVARIATE_DISTRIBUTIONS| * 2
NUM_SAMPLES_AVAILABLE_TO_MODEL = np.geomspace(10, 250, num=2)
NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE = 3
NUM_SYN_SAMPLES_TO_GENERATE = 25000
MULTIVARIATE_DISTRIBUTIONS = ['m_gaussian_0_0', 'm_gaussian_1_1']  # 'gaussian_1'

# ========== OUTPUT DIRECTORIES ==========
OUTPUT_DIR = 'OUTPUT/'
MODELS_OUTPUT_DIR = OUTPUT_DIR + 'MODELS/'
SYN_DATA_OUTPUT_DIR = OUTPUT_DIR + 'SYN_DATA/'
REAL_DATA_OUTPUT_DIR = OUTPUT_DIR + 'REAL_DATA/'

RESULTS_DIR = 'RESULTS/'

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(MODELS_OUTPUT_DIR)
os.makedirs(SYN_DATA_OUTPUT_DIR)
os.makedirs(REAL_DATA_OUTPUT_DIR)

os.makedirs(RESULTS_DIR)

# ========== RUN PIPELINE ==========
def output_dirs(dist, n, k):
    model_tag_base = '[{0}]_[n={1}]_[k={2}]'.format(dist, n, k)
    model_1_tag = model_tag_base + '_[v=1]'
    model_2_tag = model_tag_base + '_[v=2]'

    model_1_dir = '{0}{1}/'.format(MODELS_OUTPUT_DIR, model_1_tag)
    model_2_dir = '{0}{1}/'.format(MODELS_OUTPUT_DIR, model_2_tag)

    syn_data_1_dir = '{0}{1}/'.format(SYN_DATA_OUTPUT_DIR, model_1_tag)
    syn_data_2_dir = '{0}{1}/'.format(SYN_DATA_OUTPUT_DIR, model_2_tag)

    real_data_1_dir = '{0}{1}/'.format(REAL_DATA_OUTPUT_DIR, model_1_tag)
    real_data_2_dir = '{0}{1}/'.format(REAL_DATA_OUTPUT_DIR, model_2_tag)

    return model_1_dir, model_2_dir, syn_data_1_dir, syn_data_2_dir, real_data_1_dir, real_data_2_dir


def train_and_generate_samples(num_samples_available_to_model):
    # Generate real and synthetic samples:
    for k in range(NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE):
        for dist in MULTIVARIATE_DISTRIBUTIONS:
            # Set up output directories
            model_1_dir, model_2_dir, syn_data_1_dir, syn_data_2_dir, real_data_1_dir, real_data_2_dir = output_dirs(dist, num_samples_available_to_model, k)

            # Set up commands
            train_cmd_1 = 'python3 train_prob_gan.py {0} {1} {2}'.format(dist, num_samples_available_to_model, model_1_dir)
            train_cmd_2 = 'python3 train_prob_gan.py {0} {1} {2}'.format(dist, num_samples_available_to_model, model_2_dir)

            generate_syn_cmd_1 = 'python3 generate_prob_gan.py {0} {1} {2}'.format(model_1_dir + 'generator', NUM_SYN_SAMPLES_TO_GENERATE, syn_data_1_dir)
            generate_syn_cmd_2 = 'python3 generate_prob_gan.py {0} {1} {2}'.format(model_2_dir + 'generator', NUM_SYN_SAMPLES_TO_GENERATE, syn_data_2_dir)

            generate_real_cmd_1 = 'python3 sample_prob_dist.py {0} {1} {2}'.format(dist, num_samples_available_to_model, real_data_1_dir)
            generate_real_cmd_2 = 'python3 sample_prob_dist.py {0} {1} {2}'.format(dist, num_samples_available_to_model, real_data_2_dir)

            # Run commands
            os.system(train_cmd_1)
            os.system(train_cmd_2)
            os.system(generate_syn_cmd_1)
            os.system(generate_syn_cmd_2)
            os.system(generate_real_cmd_1)
            os.system(generate_real_cmd_2)


def compute_power_between_distributions(num_samples_available_to_model, dist_1, dist_2):
    t_real_power = []
    t_syn_power = []

    for k in range(NUM_MODELS_TO_TRAIN_PER_SAMPLE_SIZE):
        # Retrieve data directories
        # Note syn_data_1_dir contains data generated by v=1 model.
        #      syn_data_2_dir contains data generated by v=2 model.
        # This strategy ensures that null tests aren't testing two synthetic samples
        # that were generated by the same GAN.
        _, _, syn_data_1_dir, _, real_data_1_dir, _ = output_dirs(dist_1, num_samples_available_to_model, k)
        _, _, _, syn_data_2_dir, _, real_data_2_dir = output_dirs(dist_2, num_samples_available_to_model, k)

        syn_data_1 = np.load(syn_data_1_dir + 'data.npy')
        real_data_1 = np.load(real_data_1_dir + 'data.npy')
        syn_data_2 = np.load(syn_data_2_dir + 'data.npy')
        real_data_2 = np.load(real_data_2_dir + 'data.npy')

        real_power = multivariate_power_calculation(real_data_1, real_data_2, len(real_data_1), len(real_data_2), alpha=0.05, k=1)
        syn_power = multivariate_power_calculation(syn_data_1, syn_data_2, len(syn_data_1), len(syn_data_2), alpha=0.05, k=1)

        t_real_power.append(real_power)
        t_syn_power.append(syn_power)

    # Save power results:
    results_pth = '{0}[{1}*{2}]/'.format(RESULTS_DIR, dist_1, dist_2)
    real_results_pth = results_pth + 'real.npy'
    syn_results_pth = results_pth + 'syn.npy'
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


def compute_all_power_tests(num_samples_available_to_model):
    # For every combination of distributions, compute the power
    #  of a t test distinguishing between real and synthetic samples respectively
    #  and save the results.
    for i in range(len(MULTIVARIATE_DISTRIBUTIONS)):
        for j in range(i, len(MULTIVARIATE_DISTRIBUTIONS)):
            dist_1 = MULTIVARIATE_DISTRIBUTIONS[i]
            dist_2 = MULTIVARIATE_DISTRIBUTIONS[j]
            compute_power_between_distributions(num_samples_available_to_model, dist_1, dist_2)


def clear_output_dirs():
    shutil.rmtree(MODELS_OUTPUT_DIR)
    shutil.rmtree(SYN_DATA_OUTPUT_DIR)
    shutil.rmtree(REAL_DATA_OUTPUT_DIR)


# ========== MAIN ==========
for i in range(NUM_SAMPLES_AVAILABLE_TO_MODEL.shape[0]):
    n = int(NUM_SAMPLES_AVAILABLE_TO_MODEL[i])
    train_and_generate_samples(n)
    compute_all_power_tests(n)
    clear_output_dirs()

# ========== VISUALIZATION ==========
for i in range(len(MULTIVARIATE_DISTRIBUTIONS)):
    for j in range(i, len(MULTIVARIATE_DISTRIBUTIONS)):
        dist_1 = MULTIVARIATE_DISTRIBUTIONS[i]
        dist_2 = MULTIVARIATE_DISTRIBUTIONS[j]

        results_pth = '{0}[{1}*{2}]/'.format(RESULTS_DIR, dist_1, dist_2)
        real_results_pth = results_pth + 'real.npy'
        syn_results_pth = results_pth + 'syn.npy'

        t_real_power_for_sample_size_for_dist1_dist2 = np.load(real_results_pth).T
        t_syn_power_for_sample_size_for_dist1_dist2 = np.load(syn_results_pth).T

        plt.figure()
        sns.tsplot(data=t_real_power_for_sample_size_for_dist1_dist2, time=NUM_SAMPLES_AVAILABLE_TO_MODEL, ci=[68, 95], color='blue', condition='T Test Real')
        sns.tsplot(data=t_syn_power_for_sample_size_for_dist1_dist2, time=NUM_SAMPLES_AVAILABLE_TO_MODEL, ci=[68, 95], color='orange', condition='T Test Syn')
        plt.title('{0} vs {1}'.format(dist_1, dist_2))
        plt.xlabel('Real Samples')
        plt.ylabel('Power')
        plt.ylim([-0.1, 1.1])
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig('{0}true_sample_size_vs_power.png'.format(results_pth))
        plt.close()
