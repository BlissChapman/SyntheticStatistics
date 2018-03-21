import matplotlib.pyplot as plt


class Plot:

    def plot_samples(real_data, noise, synthetic_data, output_file):
        figure = plt.figure(figsize=(10, 10))

        real_data = real_data.flatten()
        noise = noise.flatten()
        synthetic_data = synthetic_data.flatten()

        # Real Data
        real_ax = plt.subplot(6, 1, 1)
        real_ax.set_title('REAL')
        real_ax.hist(real_data)
        real_ax.set_xlim(-1, 5)

        # Noise
        noise_ax = plt.subplot(6, 1, 3)
        noise_ax.set_title('NOISE')
        noise_ax.hist(noise)
        noise_ax.set_xlim(-1, 5)

        # Synthetic
        synthetic_ax = plt.subplot(6, 1, 5)
        synthetic_ax.set_title('SYNTHETIC')
        synthetic_ax.hist(synthetic_data)
        synthetic_ax.set_xlim(-1, 5)

        figure.savefig(output_file)
        plt.close()

    def plot_histories(histories, titles, output_path):
        plt.figure(figsize=(30, 20))
        for history in histories:
            plt.plot(history)
        plt.legend(titles)
        plt.savefig(output_path)
        plt.close()
