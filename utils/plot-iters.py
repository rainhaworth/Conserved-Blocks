import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


bszs = [64, 128, 256, 512, 1024]

# construct dataset
data = None
for bsz in bszs:
    filename = '/fs/nexus-scratch/rhaworth/output/batchsz-' + str(bsz) + '-iters.csv' 

    # read data
    newdata = pd.read_csv(filename, index_col='Iter')
    newdata['Time'] = newdata['Time'] / 60
    # add 0 at index -1
    newdata = pd.concat([pd.DataFrame([[0.0, 0, 0]], columns=['Time', 'Clusters', 'Seqs'], index=[-1]), newdata])
    newdata['Batch Size'] = [str(bsz)] * newdata.shape[0]

    if data is None:
        data = newdata
    else:
        data = pd.concat([data, newdata])


# plot clusters vs time
sns.lineplot(data, x='Time', y='Clusters', style='Batch Size', hue='Batch Size')

plt.title('Number of Clusters Recovered')
plt.xlabel('Time (min)')
plt.xlim(0, 120)
plt.ylim(0, data.max(axis=0)['Clusters'])
plt.tight_layout()
plt.savefig('iters-clusts.png', facecolor='white', dpi=100)

plt.clf()

# plot seqs vs time
sns.lineplot(data, x='Time', y='Seqs', style='Batch Size', hue='Batch Size')

plt.title('Number of Sequences Clustered')
plt.xlabel('Time (min)')
plt.xlim(0, 120)
plt.ylim(0, data.max(axis=0)['Seqs'])
plt.tight_layout()
plt.savefig('iters-seqs.png', facecolor='white', dpi=100)