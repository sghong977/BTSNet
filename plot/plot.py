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
            'Hollywood2':[Hollywood2_arr, Hollywood2_l],
            'EpicKitchen':[EpicKitchen_arr, EpicKitchen_l],
        }

# -------- Fuse Layer ----------
# TC vs C
titles = [
    # M
    'UCF-101 | Comparing M (O2-1)',
    'UCF-101 | Comparing M (O2-2)',
    'UCF-101 | Comparing M (O2-3)',
    'UCF-101 | Comparing M (O1)',
    'UCF-101 | Comparing M (O2-C)',
    # candidates
    'UCF-101 | O1 vs O2 (1)',
    'UCF-101 | O1 vs O2 (2)',
    'UCF-101 | O1 vs O2 (3)',
    # Depth-ucf
    'UCF-101 | Depth: 26 vs 50 vs 101 (O2-1)',
    'UCF-101 | Depth: 26 vs 50 vs 101 (O2-2)',
    'UCF-101 | Depth: 26 vs 50 vs 101 (O2-3)',
    'UCF-101 | Depth: 26 vs 50 vs 101 (O1)',
    'UCF-101 | Depth: 26 vs 50 (O2-C)',
    # depth-others
    'HMDB-51 | Depth: 26 vs 50 vs 101',
    'SVW | Depth: 26 vs 50', # M=4 TC
    'Hollywood2 | Depth: 26 vs 50 vs 101', # M=4 TC
    #Fuse Layer
    'UCF-101 | TC vs C (Depth 26)',
    'UCF-101 | TC vs C (Depth 50)',

    #Pretrain
    'UCF-101 | Pretrained on MiT or not',   # M=4 TC O2 D50
    'Hollywood2 | Pretrained on MiT or not',   # M=4 TC O2 D50

    #Just Result
    'EpicKitchen | noun',
    ]

reference_data = ['UCF', 'UCF','UCF','UCF','UCF',
                'UCF','UCF','UCF',
                'UCF','UCF','UCF','UCF','UCF', 'HMDB', 'SVW', 'Hollywood2',
                'UCF','UCF',  # Fuse 
                'UCF', 'Hollywood2',
                'EpicKitchen']
div = [1, 1, 1, 1,2,
        2,2,2,
        1, 1,2,1,2,   2, 1, 1, 
        2,2,   # Fuse
        1, 1,
        2] # for coloring
indices = [ # M
        [0,1,2,3, 4,7,10,13],
        [0,1,2,3, 5,8,11,14],
        [0,1,2,3, 9,12],
        [0,1,2,3, 26, 29],
        [0,1,2,3, 17,18,20,21],
        # candidates
        [0,1,2,3, 8,9, 27,28],
        [0,1,2,3, 7,10, 26,29],
        [0,1,2,3, 9,12, 28,31],
        #depth-ucf
        [0,1,2,3, 7,8,9],
        [0,1,2,3, 10,11,12],
        [0,1,2,3, 4,13, 5,14],
        [0,1,2,3, 26,27,28],
        [0,1,2,3, 17,20, 18,21],
        # depth-others
        [2,5, 3,6, 4,7, 0,1],
        [0,1],
        [0,1,2],
        # Fuse
        [0,1,2,3, 7,10, 17,20],
        [0,1,2,3, 8,11, 18,21],
        # last
        [11, 16],
        [1,3],

        [0,1, 2,3, 4]
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
    plt.close()
