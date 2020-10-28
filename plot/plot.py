from  exp_done_list import *
import matplotlib.pyplot as plt

"""
- Dataset
- Fuse Layer
- Depth
- M
- Candidate Ops
"""

Dataset = {'UCF':[UCF_arr, UCF_l],
            'HMDB':[HMDB_arr, HMDB_l],
            'SVW' : [SVW_arr, SVW_l],
            'Hollywood2':[Hollywood2_arr, Hollywood2_l]
        }

# -------- Fuse Layer ----------
# TC vs C
titles = [
    'UCF-101 | Fuse: TC vs C', # M=4 
    'HMDB-51 | Different Candidates', # M=3, C vs OLD
    'UCF-101 | M: 4 vs 7',    # TC
    'HMDB-51 | M: 3 vs 4',   # C
    'UCF-101 | Depth: 26 vs 50',
    'HMDB-51 | Depth: 26 vs 50 vs 101 (1)',
    'HMDB-51 | Depth: 26 vs 50 vs 101 (2)',
    'SVW | Depth: 26 vs 50', # M=4 TC
    'Hollywood2 | Depth: 26 vs 50 vs 101', # M=4 TC
    ]

reference_data = ['UCF', 'HMDB','UCF', 'HMDB',
                    'UCF', 'HMDB', 'HMDB','SVW', 'Hollywood2']
div = [2, 3, 2, 3, 
        3, 2, 2, 1, 1] # for coloring
indices = [[0,1, 2,3],
        [5,6,7, 8,9,10, 0,1],
        [2,3, 5,6],
        [8,9,10, 11,12,13, 0,1],

        [0,2,5, 1,3,6],
        [2,5, 3,6, 4,7, 0,1],
        [8,11, 9,12, 10,13, 0,1],
        [0,1],
        [0,1,2]
        ]

#------------------- plot ------------------------
colors = ['red',  'blue', 'black', 'grey', 'purple', 'green', 'cyan', 'magenta', 'orange', 'pink', 'yellow']
lstyle=['-', '-.', '--', ':']

w, h = 5, 5
# to plot each setting
for title in range(len(titles)):
    # valid accuracy
    plt.figure(figsize=(w,h))
    plt.title(titles[title])

    for j in range(len(indices[title])):
        arr, lb = Dataset[reference_data[title]]
        idx = indices[title][j]
        #x = [k for k in range(1,len(arr[indices[title][j]])+1)]
        x = [k for k in range(1,201)]
        plt.plot(x, arr[idx][:200], label=lb[idx], alpha=1., linestyle=lstyle[j % div[title]], color=colors[j // div[title]])
    plt.grid()
    plt.legend()
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(titles[title] + "_valid.png")
    plt.cla()
