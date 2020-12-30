""" multistep / default card=16
UCF101 : ep200 lr0.1
HMDB51 : ep250 lr0.1
SVW    : ep250 lr0.1 / slowfast ep200 0.01
epic   : ep200 lr 0.01

card32 for resnext
"""
import os

UCF_l = [
        'ResNet-50', #0
        'ResNet-101',
        'ResNeXt-50',
        'ResNeXt-101',

        'SlowFast-50', #4
        'SlowFast-101',
        'SlowFast-152',
        'SlowFast-200',

        # card16
        'TC-M4-BTS26-O2-C16', # 8
        'TC-M4-BTS50-O2-C16',
        'TC-M4-BTS101-O2-C16',
        # card8
        'TC-M4-BTS26-O2-C8', # 11
        'TC-M4-BTS50-O2-C8',
        'TC-M4-BTS101-O2-C8',

        # -SGDR
        'SlowFast-50-SGDR', #14
        'SlowFast-101-SGDR',
        'SlowFast-152-SGDR',
        'SlowFast-200-SGDR',

        'TC-M4-BTS26-O2-C32', # 18
        'TC-M4-BTS50-O2-C32',

       ]
UCF = [
    'results/ucf101_resnet50_card16multistep0.1_M4_O2_TC_20201127-141256', #ucf101_resnet50_card16multistep0.1_M4_O2_TC_20201202-055852_val_acc
    'results/ucf101_resnet101_card16multistep0.1_M4_O2_TC_20201128-131955',
    'results/ucf101_class101_resnext50_card32multistep0.1_M4_O2_TC_20201204-140824',
    'results/ucf101_class101_resnext101_card32multistep0.1_M4_O2_TC_20201203-175021',

    'results/ucf101_slowfast50_card16multistep0.1_M4_O2_TC_20201201-111357',
    'results/ucf101_slowfast101_card16multistep0.1_M4_O2_TC_20201130-144454',  # bad results
    'results/ucf101_slowfast152_card16multistep0.1_M4_O2_TC_20201125-131232',
    'results/ucf101_slowfast200multistep0.1_M4_O2_TC_20201124-100349',

    # card 16
    'results/ucf101_btsnet26multistep0.1_M4_O2_TC_20201120-032353',
    'results/ucf101_btsnet50multistep0.1_M4_O2_TC_20201121-121614',
    'results/ucf101_btsnet101multistep0.1_M4_O2_TC_20201122-224317',
    #card 8
    'results/ucf101_btsnet26_card8multistep0.1_M4_O2_TC_20201124-054848',
    'results/ucf101_btsnet50_card8multistep0.1_M4_O2_TC_20201125-153920',
    'results/ucf101_btsnet101_card8multistep0.1_M4_O2_TC_20201127-091834',

    # SlowFast SGDR
    'results/ucf101_class101_slowfast50_card32SGDR0.01_M4_O2_TC_20201207-192901',
    'results/ucf101_class101_slowfast101_card32SGDR0.01_M4_O2_TC_20201208-173142',
    'results/ucf101_class101_slowfast152_card32SGDR0.01_M4_O2_TC_20201209-141836',
    'results/ucf101_class101_slowfast200_card32SGDR0.01_M4_O2_TC_20201210-132155',

    # C32
    'results/ucf101_class101_btsnet26_card32multistep0.1_M4_O2_TC_20201222-032619',
    'results/ucf101_class101_btsnet50_card32multistep0.1_M4_O2_TC_20201223-140701',
    ]


#HMDB-51
HMDB_l = [
        'ResNet-50', #0
        'ResNet-101',
        'ResNeXt-50',
        'ResNeXt-101',

        'SlowFast-50', #4
        'SlowFast-101',
        'SlowFast-152',
        'SlowFast-200',

        'TC-M4-BTS26-O2', # 8
        'TC-M4-BTS50-O2',
        'TC-M4-BTS101-O2',

        'TC-M4-BTS26-O2-C32', #11
        'TC-M4-BTS50-O2-C32',

        # SGDR
        'SlowFast-50-SGDR', #13
        'SlowFast-101-SGDR',
        'TC-M4-BTS26-O2-SGDR',
]
HMDB = [
    'results/hmdb51_resnet50multistep0.1_M4_O2_TC_20201121-021448',
    'results/hmdb51_resnet101multistep0.1_M4_O2_TC_20201121-181923',
    'results/hmdb51_resnext50_card32multistep0.1_M4_O2_TC_20201124-174912',
    'results/hmdb51_resnext101_card32multistep0.1_M4_O2_TC_20201125-070703',

    'results/hmdb51_slowfast50_M3_O2_TC_20201116-122127',  ##??
    'results/hmdb51_slowfast101multistep0.1_M4_O2_TC_20201120-130557', #4
    'results/hmdb51_slowfast152multistep0.1_M4_O2_TC_20201123-113932',
    'results/hmdb51_slowfast200_card16multistep0.1_M4_O2_TC_20201123-224857',

    'results/hmdb51_btsnet26multistep0.1_M4_O2_TC_20201120-154336', #8
    'results/hmdb51_btsnet50multistep0.1_M4_O2_TC_20201121-081446',
    'results/hmdb51_btsnet101multistep0.1_M4_O2_TC_20201122-015538',

    'results/hmdb51_class51_btsnet26_card32multistep0.1_M4_O2_TC_20201221-201736',
    'results/hmdb51_class51_btsnet50_card32multistep0.1_M4_O2_TC_20201222-201107',

    # SGDR
    'results/hmdb51_class51_slowfast50_card32SGDR0.1_M4_O2_TC_20201209-181623',
    'results/hmdb51_class51_slowfast101_card32SGDR0.1_M4_O2_TC_20201210-035057',
    'results/hmdb51_class51_btsnet26_card16SGDR0.1_M4_O2_TC_20201210-145533',
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

    '',   #2
    'results/SVW_slowfast101multistep0.01_M4_O2_TC_20201119-152318',
    'results/SVW_slowfast152multistep0.1_M4_O2_TC_20201123-145955',  # 58%.. not good
    'results/SVW_slowfast200_card16multistep0.1_M4_O2_TC_20201124-035243', #59%... not good

    'results/SVW_btsnet26_card16multistep0.1_M4_O2_TC_20201124-170418',  #6
    'results/SVW_btsnet50_card16multistep0.1_M4_O2_TC_20201125-084921',
    'results/SVW_btsnet101_card16multistep0.1_M4_O2_TC_20201126-024015',
    ]

EpicKitchen_l = [
        'ResNet-50', #0
        'ResNet-101',

        'ResNeXt-50-C32', #2
        'ResNeXt-101-C32',

        'SlowFast-101', #4
        'SlowFast-152',
        'SlowFast-200',

        'TC-M4-BTS26-O2-C16', # 7
        'TC-M4-BTS50-O2-C16',
        'TC-M4-BTS101-O2-C16',

        'TC-M4-BTS26-O2-C32', # 10
        'TC-M4-BTS50-O2-C32',
]
# slowfast101 : starts with lr 0.01. (not converged well on lr 0.1)
EpicKitchen = [
    'results/epic_resnet50multistep0.01_M4_O2_TC_20201123-171226', #epic_resnet50multistep0.01_M4_O2_TC_20201123-171226_val_acc
    'results/epic_resnet101multistep0.01_M4_O2_TC_20201121-110645',
    'results/epic_resnext50_card32multistep0.01_M4_O2_TC_20201203-003210',
    'results/epic_resnext101_card32multistep0.01_M4_O2_TC_20201130-141516',
    
    #multistep
    'results/epic_slowfast101_card16multistep0.01_M4_O2_TC_20201125-230609', #epic_slowfast101_M4_O2_TC_20201113-004559',
    'results/epic_slowfast152_card16multistep0.01_M4_O2_TC_20201128-143609',
    'results/epic_slowfast200_card16multistep0.01_M4_O2_TC_20201201-080109',

    'results/epic_btsnet26multistep0.01_M4_O2_TC_20201121-213122', #7
    'results/epic_btsnet50multistep0.01_M4_O2_TC_20201125-112512',
    'results/epic_btsnet101_card16multistep0.01_M4_O2_TC_20201130-002218',

    'results/epic_class352_btsnet26_card32multistep0.01_M4_O2_TC_20201223-225409',  # 높은거 쓴거라 사실 결과 두개임
    'results/epic_class352_btsnet50_card32multistep0.01_M4_O2_TC_20201229-220916', # 높은거 쓴거라 사실 결과 두개임

]

EpicKitchen_verb_l = [
        'ResNet-50', #0
        'ResNet-101',

        'ResNeXt-50-C32', #2
        'ResNeXt-101-C32',

        'SlowFast-50', #4
        'SlowFast-101',
        'SlowFast-200',

        'TC-M4-BTS26-O2-C32', # 7
        'TC-M4-BTS50-O2-C32',
        #'TC-M4-BTS101-O2-C16',
]
# slowfast101 : starts with lr 0.01. (not converged well on lr 0.1)
EpicKitchen_verb = [
    '', #'results/epic_class152_resnet50_card16multistep0.01_M4_O2_TC_20201205-071733',
    '', #'results/epic_class152_resnet101_card16multistep0.01_M4_O2_TC_20201209-031508',

    '', #'results/epic_class152_resnext50_card32multistep0.01_M4_O2_TC_20201208-221920',
    '', #'results/epic_class152_resnext101_card32multistep0.01_M4_O2_TC_20201205-212559',

    'results/epic_class125_slowfast50_card32multistep0.01_M4_O2_TC_20201220-083322',
    'results/epic_class125_slowfast101_card32multistep0.01_M4_O2_TC_20201222-133443',    
    '',

    'results/epic_class125_btsnet26_card32multistep0.01_M4_O2_TC_20201223-225453', #   epic_class152_btsnet26_card16multistep0.01_M4_O2_TC_20201205-204043',
    'results/epic_class125_btsnet50_card32multistep0.01_M4_O2_TC_20201228-012857', #epic_class152_btsnet50_card16multistep0.01_M4_O2_TC_20201209-195255',
    #'results/epic_class152_btsnet101_card16multistep0.01_M4_O2_TC_20201212-233658'
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

EpicKitchen_verb_arr = []
for j in range(len(EpicKitchen_verb)):
    if EpicKitchen_verb[j] == '':
        EpicKitchen_verb_arr.append([])
    else:
        f2 = open(path+EpicKitchen_verb[j]+"_val_acc.txt", 'r')
        a = f2.readline().split(' ')[0:-1]
        a = [float(i) for i in a]
        EpicKitchen_verb_arr.append(a)
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
    if SVW[j] == '':
        SVW_arr.append([])
    else:
        f2 = open(path+SVW[j]+"_val_acc.txt", 'r')
        a = f2.readline().split(' ')[0:-1]
        a = [float(i) for i in a]
        SVW_arr.append(a)
        f2.close()


UCF_print_idx =   [0,1,2,3, 4,5,6,7, 8,9,10, 11,12,13, 14,15,16,17, 18,19]   #[0,1,2,3, 10,11,12]
HMDB_print_idx = [0,1,2,3, 4,5,6,7, 8,9,10, 11,12] ##,13]
SVW_print_idx = [0,1, 3,4,5, 6,7,8]
EPIC_print_idx=[0,1,2,3, 4,5,6, 7,8,9, 10,11]
VERB_print_idx=[4,5,7,8]

print("UCF")
for i in range(len(UCF_print_idx)):
    print(UCF_l[UCF_print_idx[i]], round(UCF_arr[UCF_print_idx[i]][-1]*100, 5))
print("HMDB")
for i in range(len(HMDB_print_idx)):
    print(HMDB_l[HMDB_print_idx[i]], round(HMDB_arr[HMDB_print_idx[i]][-1]*100,5))
print("SVW")
for i in range(len(SVW_print_idx)):
    print(SVW_l[SVW_print_idx[i]], round(SVW_arr[SVW_print_idx[i]][-1]*100,5))
print("EPIC-noun")
for i in range(len(EPIC_print_idx)):
    print(EpicKitchen_l[EPIC_print_idx[i]], round(EpicKitchen_arr[EPIC_print_idx[i]][-1]*100,5))
print("EPIC-verb")
for i in range(len(VERB_print_idx)):
    data = EpicKitchen_verb_arr[VERB_print_idx[i]]
    print(EpicKitchen_verb_l[VERB_print_idx[i]], round(data[-1]*100,5))
