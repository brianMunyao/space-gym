import matplotlib.pyplot as plt


def plotScores(x, scores,  filename):
    fig = plt.figure()
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax2.plot(x, scores, color="C0")
    ax2.set_xlabel("Game", color="C0")
    ax2.tick_params(axis='x', colors="C0")
    ax2.set_ylabel("Scores", color="C0")
    ax2.tick_params(axis='y', colors="C0")

    plt.savefig(filename)
