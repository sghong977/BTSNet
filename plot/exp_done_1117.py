# create new file because of the different learning scheduler.

#UCF101
# Temporal-Channel = TC
# channel-wise = C
# temporal/spatial divide X = OLD
import os

UCF_l = [
        'ResNet-50', #0
        'ResNet-101',
        'SlowFast-50',
        'SlowFast-101',

        #--- TC, O2, M=2,3,4
        'TC-M2-BTS26-O2', #4
        'TC-M2-BTS50-O2',
        'TC-M2-BTS101-O2',   #X

        'TC-M3-BTS26-O2',  #7
        'TC-M3-BTS50-O2',
        'TC-M3-BTS101-O2',

        'TC-M4-BTS26-O2', # 10
        'TC-M4-BTS50-O2',
        'TC-M4-BTS101-O2',

        'TC-M7-BTS26-O2', # 13
        'TC-M7-BTS50-O2',
        'TC-M7-BTS101-O2',   #       X

        'TC-M4-BTS50-O2-PRET', #16
        #--- C
        'C-M3-BTS26-O2',  #17
        'C-M3-BTS50-O2',
        'C-M3-BTS101-O2',

        'C-M4-BTS26-O2', #20         
        'C-M4-BTS50-O2',
        'C-M4_BTS101-O2',

        #--------------- O1
        'TC-M2-BTS26-O1', #23
        'TC-M2-BTS50-O1',
        'TC-M2-BTS101-O1',
        
        'TC-M3-BTS26-O1',  #26
        'TC-M3-BTS50-O1', 
        'TC-M3-BTS101-O1',

        'TC-M4-BTS26-O1',  #29
        'TC-M4-BTS50-O1',
        'TC-M4-BTS101-O1',
        #-- O1, C
        'C-M3-BTS26-O1',        #32
        'C-M3-BTS50-O1',
        'C-M3-BTS101-O1',

        'C-M4-BTS26-O1',        #35
        'C-M4-BTS50-O1',
        'C-M4-BTS101-O1',
        ]
UCF = [
    ]


#HMDB-51
HMDB_l = [
    'ResNet50', #0
    'ResNet101',
    'Slowfast-50',
    'Slowfast-101',

    'TC-M4-BTS26-O2', 
    'TC-M4-BTS50-O2',
    'TC-M4-BTS101-O2',

]
HMDB = [
    'results/hmdb51_slowfast50plateau0.1_M4_O2_TC_20201117-145838',
    'results/hmdb51_slowfast101plateau0.1_M4_O2_TC_20201117-222214',

    'results/hmdb51_spnet26plateau0.1_M4_O2_TC_20201117-175035',



]

# SVW
# Channel-wise
SVW_l = [
    'ResNet50', #0
    'ResNet101',
    'Slowfast-50',
    'Slowfast-101',

    'TC-M4-BTS26-O2',
    'TC-M4-BTS50-O2'
]
SVW = [
    'results/SVW_slowfast50plateau0.1_M4_O2_TC_20201117-180305',
    'results/SVW_slowfast101plateau0.1_M4_O2_TC_20201118-040755',
    ]

EpicKitchen_l = [
    'ResNet-50',
    'ResNet-101',
    'SlowFast-50',
    'SlowFast-101',

    'TC-M4-BTS26-O2',
    'TC-M4-BTS50-O2',

]
# slowfast101 : starts with lr 0.01. (not converged well on lr 0.1)
EpicKitchen = [
    'results/epic_resnet50_M4_O2_TC_20201104-085926',
    'results/epic_resnet101_M4_O2_TC_20201116-014327', #'results/epic_resnet101_M4_O2_TC_20201106-155250',
    'results/epic_slowfast50_M4_O2_TC_20201104-032644',
    'results/epic_slowfast101_M4_O2_TC_20201111-014056',
    'results/epic_slowfast101_M4_O2_TC_20201113-004559', #'results/epic_slowfast101_M4_O2_TC_20201106-061910',
    
    'results/epic_spnet26_M4_O2_TC_20201106-171538',
    'results/epic_spnet50_M4_O2_TC_20201110-011514',
]


#--------------------------
path = '../'

# read data
UCF_arr = []
for j in range(len(UCF)):
    if UCF[j] == '':
        UCF_arr.append([])
    else:
        f2 = open(path+UCF[j]+"_val_acc.txt", 'r')
        a = f2.readline().split(' ')[0:-1]
        a = [float(i) for i in a]
        UCF_arr.append(a)
        f2.close()

EpicKitchen_arr = []
for j in range(len(EpicKitchen)):
    if EpicKitchen[j] == '':
        EpicKitchen_arr.append([])
    else:
        f2 = open(path+EpicKitchen[j]+"_val_acc.txt", 'r')
        a = f2.readline().split(' ')[0:-1]
        a = [float(i) for i in a]
        EpicKitchen_arr.append(a)
        f2.close()

HMDB_arr = []
for j in range(len(HMDB)):
    if HMDB[j] == '':
        HMDB_arr.append([])
    else:
        f2 = open(path+HMDB[j]+"_val_acc.txt", 'r')
        a = f2.readline().split(' ')[0:-1]
        a = [float(i) for i in a]
        HMDB_arr.append(a)
        f2.close()

SVW_arr = []
for j in range(len(SVW)):
    f2 = open(path+SVW[j]+"_val_acc.txt", 'r')
    a = f2.readline().split(' ')[0:-1]
    a = [float(i) for i in a]
    SVW_arr.append(a)
    f2.close()

Hollywood2_arr = []
for j in range(len(Hollywood2)):
    f2 = open(path+Hollywood2[j]+"_val_acc.txt", 'r')
    a = f2.readline().split(' ')[0:-1]
    a = [float(i) for i in a]
    Hollywood2_arr.append(a)
    f2.close()

UCF_print_idx =         [0,1,2,3, 17,18, 32,33]   #[0,1,2,3, 10,11,12]
#HMDB_print_idx = [0,1,    10,11,12]
#SVW_print_idx = [0,1]

print("UCF")
for i in range(len(UCF_print_idx)):
    print(UCF_l[UCF_print_idx[i]], round(UCF_arr[UCF_print_idx[i]][-1]*100, 5))
"""
print("HMDB")
for i in range(len(HMDB_print_idx)):
    print(HMDB_l[HMDB_print_idx[i]], HMDB_arr[HMDB_print_idx[i]][-1])
print("SVW")
for i in range(len(SVW_print_idx)):
    print(SVW_l[SVW_print_idx[i]], SVW_arr[SVW_print_idx[i]][-1])

"""