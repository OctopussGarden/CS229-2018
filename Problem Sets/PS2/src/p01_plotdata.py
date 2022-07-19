import numpy as np
import util
import matplotlib.pyplot as plt


def plotdata(x, y, title):
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bo')
    plt.plot(x[y == -1, -2], x[y == -1, -1], 'rx')
    plt.suptitle(title, fontsize=12)
    plt.savefig('../output/'+str(title)+'.png')


x_a, y_a = util.load_csv('../data/ds1_a.csv', add_intercept=True)
x_b, y_b = util.load_csv('../data/ds1_b.csv', add_intercept=True)

plotdata(x_a, y_a, "Dataset A")
plotdata(x_b, y_b, "Dataset B")

