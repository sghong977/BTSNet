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
    # M
    'UCF-101 | Comparing M',
    'UCF-101 | Comparing M (2)',

    # Depth
    'UCF-101 | Depth: 26 vs 50 vs 101',
    'HMDB-51 | Depth: 26 vs 50 vs 101',
    'SVW | Depth: 26 vs 50', # M=4 TC
    'Hollywood2 | Depth: 26 vs 50 vs 101', # M=4 TC

    #Pretrain
    'UCF-101 | Pretrained on MiT or not',   # M=4 TC O2 D50
    'Hollywood2 | Pretrained on MiT or not',   # M=4 TC O2 D50
    ]

reference_data = ['UCF', 'UCF',
                'UCF', 'HMDB', 'SVW', 'Hollywood2',
                'UCF', 'Hollywood2']
div = [1, 2,
        2, 2, 1, 1, 
        1, 1] # for coloring
indices = [
        [0,1,2,3, 4,7,10,13],
        [0,1,2,3, 4, 10, 13],

        [0,1, 2,3, 10,11,12],
        [2,5, 3,6, 4,7, 0,1],
        [0,1],
        [0,1,2],

        [11, 16],
        [1,3],
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
