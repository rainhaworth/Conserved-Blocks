import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.family'] = 'sans'

csv_fns = glob.glob('./*.csv')
# hardcode again
shareds = [250, 500, 750, 1000]
lsrs = [1.2, 1.5, 2., 3., 4.]
difficulties = ['Hard', 'Simple']
metrics = ['Precision', 'Recall']


# grid params
nrow = 2
ncol = 2
# grid subplots
fig, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True)

# set min and max manually
minval = 0.5
maxval = 1.0

# draw all plots
print('plotting')
for i, dif in tqdm(enumerate(difficulties), total=len(difficulties)):
    for j, metric in enumerate(metrics):
        ax_curr = axs[i,j]
        
        # plot
        fn = dif.lower() + '-' + metric.lower() + '.csv'
        data = np.genfromtxt(fn, delimiter=',', usecols=range(len(lsrs)))

        sns.heatmap(data, vmin=minval, vmax=maxval, annot=True,
                    cbar=False, square=True, cmap='mako',
                    ax=ax_curr, xticklabels=lsrs, yticklabels=shareds) #type: ignore

        # hide ticks + labels
        ax_curr.tick_params(axis='both', which='both', bottom=False, left=False, labelsize='small')
                            #, labelbottom=False, labelleft=False)

        # labels
        if j == 0:
            ax_curr.set_ylabel(dif)
        if i == len(difficulties)-1:
            ax_curr.set_xlabel(metric)


fig.suptitle('Hash Metrics, Shared Region Size vs. Length:Shared Ratio')
plt.tight_layout()

# do this after tight_layout or figure breaks
fig.colorbar(axs[0, 0].collections[0], ax=axs)

plt.savefig('fig.png', dpi=200)