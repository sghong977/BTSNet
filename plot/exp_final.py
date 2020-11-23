""" multistep / default card=16
UCF101 : ep200 lr0.1
HMDB51 : ep250 lr0.1
SVW    : ep250 lr0.1 / slowfast ep200 0.01
epic   : ep200 lr 0.01
"""
import os

UCF_l = [
        'ResNet-50', #0
        'ResNet-101',
        'SlowFast-50', #2
        'SlowFast-101',
        'SlowFast-152',
        'SlowFast-200',

        'TC-M4-BTS26-O2', # 6
        'TC-M4-BTS50-O2',
        'TC-M4-BTS101-O2',
       ]
UCF = [
    'results/',
    'results/',
    'results/',
    'results/',
    'results/',
    'results/',

    'results/ucf101_btsnet26multistep0.1_M4_O2_TC_20201120-032353',
    'results/ucf101_btsnet50multistep0.1_M4_O2_TC_20201121-121614',
    'results/ucf101_btsnet101multistep0.1_M4_O2_TC_20201122-224317',
    ]


#HMDB-51
HMDB_l = [
        'ResNet-50', #0
        'ResNet-101',
        'SlowFast-50', #2
        'SlowFast-101',
        'SlowFast-152',
        'SlowFast-200',

        'TC-M4-BTS26-O2', # 6
        'TC-M4-BTS50-O2',
        'TC-M4-BTS101-O2',
]
HMDB = [
    'results/hmdb51_resnet50multistep0.1_M4_O2_TC_20201121-021448',
    'results/hmdb51_resnet101multistep0.1_M4_O2_TC_20201121-181923',

    'results/',
    'results/hmdb51_slowfast101multistep0.1_M4_O2_TC_20201120-130557',
    'results/',
    'results/',

    'results/hmdb51_btsnet26multistep0.1_M4_O2_TC_20201120-154336', #6
    'results/hmdb51_btsnet50multistep0.1_M4_O2_TC_20201121-081446',
    'results/hmdb51_btsnet101multistep0.1_M4_O2_TC_20201122-015538',
]

# SVW
# Channel-wise
SVW_l = [
        'ResNet-50', #0
        'ResNet-101',
        'SlowFast-50', #2
        'SlowFast-101',
        'SlowFast-152',
        'SlowFast-200',

        'TC-M4-BTS26-O2', # 6
        'TC-M4-BTS50-O2',
        'TC-M4-BTS101-O2',
]
SVW = [
    'results/SVW_resnet50multistep0.1_M4_O2_TC_20201121-060718',
    'results/SVW_resnet101multistep0.1_M4_O2_TC_20201121-220200',

    'results/',
    'results/SVW_slowfast101multistep0.01_M4_O2_TC_20201119-152318',
    'results/',
    'results/',

    'results/',
    'results/',
    'results/',
    ]

EpicKitchen_l = [
        'ResNet-50', #0
        'ResNet-101',
        'SlowFast-50', #2
        'SlowFast-101',
        'SlowFast-152',
        'SlowFast-200',

        'TC-M4-BTS26-O2', # 6
        'TC-M4-BTS50-O2',
        'TC-M4-BTS101-O2',
]
# slowfast101 : starts with lr 0.01. (not converged well on lr 0.1)
EpicKitchen = [
    'results/epic_resnet50_M4_O2_TC_20201118-030934',
    'results/epic_resnet101multistep0.01_M4_O2_TC_20201121-110645',
    
    '',
    'results/epic_slowfast101_M4_O2_TC_20201113-004559',
    '',
    '',

    'results/epic_btsnet26multistep0.01_M4_O2_TC_20201121-213122', #6
    '',
    '',
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