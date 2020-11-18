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
    #---- 
    'results/ucf101_resnet50_M4_20201030-094546',
    'results/ucf101_resnet101_M4_20201031-165354',
    'results/ucf101_slowfast50_M4_20201030-015238',
    'results/ucf101_slowfast101_M4_20201031-124539',
    #-
    'results/ucf101_spnet26_M2_20201030-181325', #4
    'results/ucf101_spnet50_M2_20201101-060206',
    '',
    
    'results/ucf101_spnet26_M3_20201031-221721', #7
    'results/ucf101_spnet50_M3_20201102-083027',
    'results/ucf101_spnet101_M3_O2_TC_20201105-051134',

    'results/ucf101_sknet326_M4_20201007-062945', #10
    'results/ucf101_sknet350_M4_20201008-161223',
    'results/ucf101_sknet3101_M4_20201010-090646',

    'results/ucf101_sknet326_M7_20201014-174418', #13
    'results/ucf101_sknet350_M7_20201016-183956',
    '',

    'results/ucf101_sknet350_M4_20201028-185927',  #16
    # C-O2
    'results/ucf101_spnet26_M3_O2_C_20201107-074530', #17
    'results/ucf101_spnet50_M3_O2_C_20201108-140858',
    'results/ucf101_spnet101_M3_O2_C_20201110-094606',

    'results/ucf101_spnet26_M4_O2_C_20201107-101100', #20
    'results/ucf101_spnet50_M4_O2_C_20201108-210205',
    'results/ucf101_spnet101_M4_O2_C_20201110-153538',
    #----- O1
    'results/ucf101_spnet26_M2_O1_TC_20201112-045614', #23
    'results/ucf101_spnet50_M2_O1_TC_20201113-120429',
    'results/ucf101_spnet101_M2_O1_TC_20201114-184810',

    'results/ucf101_spnet26_M3_20201031-215149', #26
    'results/ucf101_spnet50_M3_20201102-073241',
    'results/ucf101_spnet101_M3_O1_TC_20201105-042017',

    'results/ucf101_spnet26_M4_20201101-031147',  #29
    'results/ucf101_spnet50_M4_O1_TC_20201109-095610',
    'results/ucf101_spnet101_M4_O1_TC_20201107-212702',

    'results/ucf101_spnet26_M3_O1_C_20201111-051051',    # 32
    'results/ucf101_spnet50_M3_O1_C_20201112-115727',
    'results/ucf101_spnet101_M3_O1_C_20201114-002959',

    'results/ucf101_spnet26_M4_O1_C_20201111-114854',   #35
    'results/ucf101_spnet50_M4_O1_C_20201112-230727',
    'results/ucf101_spnet101_M4_O1_C_20201114-171737',

    ]


#HMDB-51
HMDB_l = [
    'ResNet50', #0
    'ResNet101',
    'Slowfast-50',
    'Slowfast-101',

    'C-M2-BTS26-O1', #4
    'C-M2-BTS50-O1',
    'C-M2-BTS101-O1',
    
    'C-M3-BTS26-O1', #7
    'C-M3-BTS50-O1',
    'C-M3-BTS101-O1',

    #------ TC O1
    'TC-M3-BTS26-O1', #10
    'TC-M3-BTS50-O1',
    'TC-M3-BTS101-O1',

    #------ TC O2
    'TC-M3-BTS26-O2', #13
    'TC-M3-BTS50-O2', 
    'TC-M3-BTS101-O2',  #X

]
HMDB = [
    'olds/results/hmdb51_resnet50_M2_20200911-202451',
    'olds/results/hmdb51_resnet101_M2_20200912-150219',
    'results/hmdb51_slowfast50_M3_O2_TC_20201116-122127',
    'results/hmdb51_slowfast101_M3_O2_TC_20201116-193107',
    
    'olds/results/hmdb51_sknet26_M2_20200911-214316', #4
    'olds/results/hmdb51_sknet50_M2_20200912-183445',
    'olds/results/hmdb51_sknet101_M2_20200913-164034',

    'olds/results/hmdb51_sknet26_M3_20200911-231448', #7
    'olds/results/hmdb51_sknet50_M3_20200912-224731',
    'olds/results/hmdb51_sknet101_M3_20200913-215856',

    'results/hmdb51_spnet26_M3_O1_TC_20201111-141814', #10
    'results/hmdb51_spnet50_M3_O1_TC_20201112-054541',
    'results/hmdb51_spnet101_M3_O1_TC_20201113-004856',

    'results/hmdb51_spnet26_M3_O2_TC_20201116-140816', #13
    'results/hmdb51_spnet50_M3_O2_TC_20201117-014740',
    '',
    
]

# SVW
# Channel-wise
SVW_l = [
    'TC-M4-BTS26-O2',
    'TC-M4-BTS50-O2'
]
SVW = [
    'results/SVW_sknet326_M4_20201019-154545',
    'results/SVW_sknet350_M4_20201020-042004',
    ]

# Hollywood2
# 쓰레기실험 하나 있음. class수 오류
Hollywood2_l = [
        'TC-M4-BTS26-O2',
        'TC-M4-BTS50-O2',
        'TC-M4-BTS101-O2',

        'TC-M4-BTS50-O2-PRET',
        ]
Hollywood2 = [
    'results/hollywood2_sknet326_M4_20201026-131234', #0
    'results/hollywood2_sknet350_M4_20201026-210727',
    'results/hollywood2_sknet3101_M4_20201027-054943',

    'results/hollywood2_sknet350_M4_20201029-030600',    #3, M4 TC O2 pret
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
    'results/epic_resnet50_M4_O2_TC_20201118-030934',    #'results/epic_resnet50_M4_O2_TC_20201104-085926',
    'results/epic_resnet101_M4_O2_TC_20201116-014327', #'results/epic_resnet101_M4_O2_TC_20201106-155250',
    'results/epic_slowfast50_M4_O2_TC_20201104-032644',
    'results/epic_slowfast101_M4_O2_TC_20201111-014056',
    'results/epic_slowfast101_M4_O2_TC_20201113-004559', #'results/epic_slowfast101_M4_O2_TC_20201106-061910',
    
    'results/epic_spnet26_M4_O2_TC_20201117-181458',   #'results/epic_spnet26_M4_O2_TC_20201106-171538',
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