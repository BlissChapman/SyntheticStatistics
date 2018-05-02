import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import shutil


# =========== HYPERPARAMETERS ==========
UNIVARIATE_DISTRIBUTIONS = ['gaussian_0', 'gaussian_1', 'chi_square_9', 'exp_9', 'gaussian_mixture']
NUM_SAMPLES = 20000
NUM_TRIALS = 5

# ========== OUTPUT DIRECTORIES ==========
OUTPUT_DIR = 'OUTPUT/'
MODELS_OUTPUT_DIR = OUTPUT_DIR + 'MODELS/'
SYN_DATA_OUTPUT_DIR = OUTPUT_DIR + 'SYN_DATA/'
REAL_DATA_OUTPUT_DIR = OUTPUT_DIR + 'REAL_DATA/'
POWER_OUTPUT_DIR = OUTPUT_DIR + 'POWER/'

RESULTS_DIR = 'RESULTS/'

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(MODELS_OUTPUT_DIR)
os.makedirs(SYN_DATA_OUTPUT_DIR)
os.makedirs(REAL_DATA_OUTPUT_DIR)
os.makedirs(POWER_OUTPUT_DIR)

os.makedirs(RESULTS_DIR)

# ========== RUN PIPELINE ==========
def generate_real_cmd(dist, num_samples, output_dir):
    return 'python3 sample_prob_dist.py {0} {1} {2}/'.format(dist, num_samples, output_dir)

def train_gan_cmd(real_data_dir, output_dir):
    return 'python3 train_prob_gan.py {0}data.npy {1}'.format(real_data_dir, output_dir)

def generate_syn_cmd(gen_dir, num_samples, output_dir):
    return 'python3 generate_prob_gan.py {0}generator {1} {2}'.format(gen_dir, num_samples, output_dir)

def power_analysis_cmd(real_data_1_dir, real_data_2_dir, syn_data_1_dir, syn_data_2_dir, output_dir):
    return 'python3 univariate_power_analysis.py {0}data.npy {1}data.npy {2}data.npy {3}data.npy {4}'.format(real_data_1_dir, syn_data_1_dir, real_data_2_dir, syn_data_2_dir, output_dir)

def output_dirs(dist, k):
    model_tag_base = '[{0}]_[k={1}]'.format(dist, k)
    model_1_tag = model_tag_base + '_[v=1]'
    model_2_tag = model_tag_base + '_[v=2]'

    real_data_1_dir = '{0}{1}/'.format(REAL_DATA_OUTPUT_DIR, model_1_tag)
    real_data_2_dir = '{0}{1}/'.format(REAL_DATA_OUTPUT_DIR, model_2_tag)

    model_1_dir = '{0}{1}/'.format(MODELS_OUTPUT_DIR, model_1_tag)
    model_2_dir = '{0}{1}/'.format(MODELS_OUTPUT_DIR, model_2_tag)

    syn_data_1_dir = '{0}{1}/'.format(SYN_DATA_OUTPUT_DIR, model_1_tag)
    syn_data_2_dir = '{0}{1}/'.format(SYN_DATA_OUTPUT_DIR, model_2_tag)

    return real_data_1_dir, real_data_2_dir, model_1_dir, model_2_dir, syn_data_1_dir, syn_data_2_dir

def run_cmd_sequence(cmds):
    for cmd in cmds:
        os.system(cmd)

def generate_real_data_samples():
    for i in range(len(UNIVARIATE_DISTRIBUTIONS)):
        for k in range(NUM_TRIALS):
            dist_i = UNIVARIATE_DISTRIBUTIONS[i]
            real_data_1_dir, real_data_2_dir, _, _, _, _ = output_dirs(dist_i, k)
            sample_real_1 = generate_real_cmd(dist_i, NUM_SAMPLES, real_data_1_dir)
            sample_real_2 = generate_real_cmd(dist_i, NUM_SAMPLES, real_data_2_dir)
            run_cmd_sequence([sample_real_1, sample_real_2])

def train_gans():
    for i in range(len(UNIVARIATE_DISTRIBUTIONS)):
        for k in range(NUM_TRIALS):
            dist_i = UNIVARIATE_DISTRIBUTIONS[i]
            real_data_1_dir, real_data_2_dir, model_1_dir, model_2_dir, _, _ = output_dirs(dist_i, k)
            train_gan_1 = train_gan_cmd(real_data_1_dir, model_1_dir)
            train_gan_2 = train_gan_cmd(real_data_2_dir, model_2_dir)
            run_cmd_sequence([train_gan_1, train_gan_2])

def generate_syn_data_samples():
    for i in range(len(UNIVARIATE_DISTRIBUTIONS)):
        for k in range(NUM_TRIALS):
            dist_i = UNIVARIATE_DISTRIBUTIONS[i]
            _, _, model_1_dir, model_2_dir, syn_data_1_dir, syn_data_2_dir = output_dirs(dist_i, k)
            sample_syn_1 = generate_syn_cmd(model_1_dir, NUM_SAMPLES, syn_data_1_dir)
            sample_syn_2 = generate_syn_cmd(model_2_dir, NUM_SAMPLES, syn_data_2_dir)
            run_cmd_sequence([sample_syn_1, sample_syn_2])

def run_power_analyses():
    for i in range(len(UNIVARIATE_DISTRIBUTIONS)):
        for j in range(i, len(UNIVARIATE_DISTRIBUTIONS)):
            for k in range(NUM_TRIALS):
                dist_i = UNIVARIATE_DISTRIBUTIONS[i]
                dist_j = UNIVARIATE_DISTRIBUTIONS[j]

                real_data_1_dir_i, real_data_2_dir_i, _, _, syn_data_1_dir_i, syn_data_2_dir_i = output_dirs(dist_i, k)
                real_data_1_dir_j, real_data_2_dir_j, _, _, syn_data_1_dir_j, syn_data_2_dir_j = output_dirs(dist_j, k)

                output_dir = '{0}[{1}_VS_{2}]_[k={3}]/'.format(POWER_OUTPUT_DIR, dist_i, dist_j, k)

                cmd = power_analysis_cmd(real_data_1_dir_i, real_data_2_dir_j, syn_data_1_dir_i, syn_data_2_dir_j, output_dir)
                run_cmd_sequence([cmd])

def visualize():
    for i in range(len(UNIVARIATE_DISTRIBUTIONS)):
        for j in range(i, len(UNIVARIATE_DISTRIBUTIONS)):
            figure, axes = plt.subplots(nrows=2, ncols=1)

            n = None
            t_test_real_power = []
            mmd_test_real_power = []
            t_test_syn_power = []
            mmd_test_syn_power = []

            for k in range(NUM_TRIALS):
                dist_i = UNIVARIATE_DISTRIBUTIONS[i]
                dist_j = UNIVARIATE_DISTRIBUTIONS[j]
                power_dir_k = '{0}[{1}_VS_{2}]_[k={3}]/'.format(POWER_OUTPUT_DIR, dist_i, dist_j, k)

                if n is None:
                    n = np.load(power_dir_k+'n.npy')
                t_test_real_power.append(np.load(power_dir_k+'t_test_real_power.npy'))
                mmd_test_real_power.append(np.load(power_dir_k+'mmd_test_real_power.npy'))
                t_test_syn_power.append(np.load(power_dir_k+'t_test_syn_power.npy'))
                mmd_test_syn_power.append(np.load(power_dir_k+'mmd_test_syn_power.npy'))

            n = np.array(n)
            t_test_real_power = np.array(t_test_real_power)
            mmd_test_real_power = np.array(mmd_test_real_power)
            t_test_syn_power = np.array(t_test_syn_power)
            mmd_test_syn_power = np.array(mmd_test_syn_power)

            # Plot curve of n vs power
            sns.tsplot(data=t_test_real_power, time=n, ci=[68, 95], color='blue', condition='Real', ax=axes[0])
            sns.tsplot(data=t_test_syn_power, time=n, ci=[68, 95], color='orange', condition='Synthetic', ax=axes[0])
            axes[0].set_title('Sample Size vs T Test Power')
            axes[0].set_xlabel('Sample Size')
            axes[0].set_ylabel('Power')
            axes[0].set_ylim([-0.1, 1.1])
            axes[0].legend(loc="upper right")

            sns.tsplot(data=mmd_test_real_power, time=n, ci=[68, 95], color='blue', condition='Real', ax=axes[1])
            sns.tsplot(data=mmd_test_syn_power, time=n, ci=[68, 95], color='orange', condition='Synthetic', ax=axes[1])
            axes[1].set_title('Sample Size vs MMD Test Power')
            axes[1].set_xlabel('Sample Size')
            axes[1].set_ylabel('Power')
            axes[1].set_ylim([-0.1, 1.1])
            axes[1].legend(loc="upper right")

            # Save results
            figure.tight_layout()
            figure.savefig('{0}{1}_VS_{2}.png'.format(RESULTS_DIR, dist_i, dist_j))


# ========== MAIN ==========
generate_real_data_samples()
train_gans()
generate_syn_data_samples()
run_power_analyses()
visualize()
