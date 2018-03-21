import matplotlib.pyplot as plt


class Plot:

    def plot_samples(real_data, noise, synthetic_data, output_file):
        figure = plt.figure(figsize=(10, 10))

        # Real Data
        real_ax = plt.subplot(6, 1, 1)
        real_ax.hist(real_data[0])

        real_ax_ln = plt.subplot(6, 1, 2)
        real_ax_ln.plot(real_data[0])

        # Noise
        noise_ax = plt.subplot(6, 1, 3)
        noise_ax.hist(noise[0])

        noise_ax_ln = plt.subplot(6, 1, 4)
        noise_ax_ln.plot(noise[0])

        # Synthetic
        synthetic_ax = plt.subplot(6, 1, 5)
        synthetic_ax.hist(synthetic_data[0])

        synth_ax_ln = plt.subplot(6, 1, 6)
        synth_ax_ln.plot(synthetic_data[0])

        figure.savefig(output_file)
        plt.close()

    def plot_histories(histories, titles, output_path):
        plt.figure(figsize=(30, 20))
        for history in histories:
            plt.plot(history)
        plt.legend(titles)
        plt.savefig(output_path)
        plt.close()
