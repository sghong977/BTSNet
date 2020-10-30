#UCF101
# Temporal-Channel = TC
# channel-wise = C
# temporal/spatial divide X = OLD
UCF_l = [
        'TC-M4-SP26-O2', #0
        'TC-M4-SP50-O2',
        'TC-M4-SP101-O2',

        'TC-M7-SP26-O2', # 3
        'TC-M7-SP50-O2',

        'TC-M4-SP50-O2-PRET', #5
        ]
UCF = [
    'results/ucf101_sknet326_M4_20201007-062945',
    'results/ucf101_sknet350_M4_20201008-161223',
    'results/ucf101_sknet3101_M4_20201010-090646',

    'results/ucf101_sknet326_M7_20201014-174418',
    'results/ucf101_sknet350_M7_20201016-183956',

    'results/ucf101_sknet350_M4_20201028-185927',
    ]


#HMDB-51
HMDB_l = [
    'ResNet50', #0
    'ResNet101',

    'C-M2-SK26-O1', #2
    'C-M2-SK50-O1',
    'C-M2-SK101-O1',
    
    'C-M3-SK26-O1', #5
    'C-M3-SK50-O1',
    'C-M3-SK101-O1',
]
HMDB = [
    'olds/results/hmdb51_resnet50_M2_20200911-202451',
    'olds/results/hmdb51_resnet101_M2_20200912-150219',
    
    'olds/results/hmdb51_sknet26_M2_20200911-214316',
    'olds/results/hmdb51_sknet50_M2_20200912-183445',
    'olds/results/hmdb51_sknet101_M2_20200913-164034',

    'olds/results/hmdb51_sknet26_M3_20200911-231448',
    'olds/results/hmdb51_sknet50_M3_20200912-224731',
    'olds/results/hmdb51_sknet101_M3_20200913-215856',
]

# SVW
# Channel-wise
SVW_l = [
    'TC-M4-26-O2',
    'TC-M4-50-O2'
]
SVW = [
    'results/SVW_sknet326_M4_20201019-154545',
    'results/SVW_sknet350_M4_20201020-042004',
    ]

# Hollywood2
# 쓰레기실험 하나 있음. class수 오류
Hollywood2_l = [
        'TC-M4-26-O2',
        'TC-M4-50-O2',
        'TC-M4-101-O2',

        'TC-M4-50-O2-PRET',
        ]
Hollywood2 = [
    'results/hollywood2_sknet326_M4_20201026-131234', #0
    'results/hollywood2_sknet350_M4_20201026-210727',
    'results/hollywood2_sknet3101_M4_20201027-054943',

    'results/hollywood2_sknet350_M4_20201029-030600',    #3, M4 TC O2 pret
    ]



#--------------------------
path = '../'

# read data
UCF_arr = []
for j in range(len(UCF)):
    f2 = open(path+UCF[j]+"_val_acc.txt", 'r')
    a = f2.readline().split(' ')[0:-1]
    a = [float(i) for i in a]
    UCF_arr.append(a)
    f2.close()

HMDB_arr = []
for j in range(len(HMDB)):
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

