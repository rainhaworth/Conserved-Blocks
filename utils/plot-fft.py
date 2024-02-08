import os
import glob
import pickle
import matplotlib.pyplot as plt

out_dir ='/fs/nexus-scratch/rhaworth/output' 
files = glob.glob(os.path.join(out_dir, '/block_cluster_*.pickle'))

fig, ax = plt.subplots()

for file in files:
    with open(file, 'rb') as f:
        string, support = pickle.load(f)

    plt.scatter(range(len(support)), support, alpha=0.7, s=10)

plt.xlim([0,4096])
plt.tight_layout()

fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.set_dpi(100)
fig.savefig('support.png', facecolor='white')

plt.show()
