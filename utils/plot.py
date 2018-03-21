import matplotlib.pyplot as plt


class Plot:

    def plot_samples(real_data, synthetic_data, output_file):
        figure = plt.figure(figsize=(10, 10))

        real_ax = plt.subplot(4, 1, 1)
        real_ax.hist(real_data[0])

        real_ax_ln = plt.subplot(4, 1, 2)
        real_ax_ln.plot(real_data[0])

        synthetic_ax = plt.subplot(4, 1, 3)
        synthetic_ax.hist(synthetic_data[0])

        synth_ax_ln = plt.subplot(4, 1, 4)
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
