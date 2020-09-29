import matplotlib.pyplot as plt

"""
1. Entire Train/Validaion Accuray Graph
2. Comparison of M size (plot 26, 50, 101 separately)
3. Comparison of Layer size (plot M=2, 3, 4 separately)
* resnet baseline should be plotted for comparison
"""

titles = ['Entire Accuracy',
            'Comparison of M size on SKNet-26',
            'Comparison of M size on SKNet-50',
            'Comparison of M size on SKNet-101',
            'Comparison of Layer depth on SKNet M=2',
            'Comparison of Layer depth on SKNet M=3',
            ]
indices = [[0,1,2,3,4,5,6,7,8,9,10],  #1
            [0,1,2,3,8],  #2
            [0,1,4,5, 9],
            [0,1,6,7, 10],
            [0,1,2,4,6], #3
            [0,1,3,5,7],
            [0,1,8,9,10]
            ]
fnames = ['hmdb51_resnet50_M2_20200911-202451',
          'hmdb51_resnet101_M2_20200912-150219',
          'hmdb51_sknet26_M2_20200911-214316',
          'hmdb51_sknet26_M3_20200911-231448',

          'hmdb51_sknet50_M2_20200912-183445',
          'hmdb51_sknet50_M3_20200912-224731',
          'hmdb51_sknet101_M2_20200913-164034',
          'hmdb51_sknet101_M3_20200913-215856',

          'hmdb51_sknet226_M4_20200925-185718',
          'hmdb51_sknet250_M4_20200926-083813',
          'hmdb51_sknet2101_M4_20200927-010416',
          ]
#['ucf101_20200904-033403']
flabels = ['ResNet-50',
            'ResNet-101',
            'SKNet-26-M2',
            'SKNet-26-M3',
            'SKNet-50-M2',
            'SKNet-50-M3',
            'SKNet-101-M2',
            'SKNet-101-M3',
            'SKNet-26-M4',
            'SKNet-50-M4',
            'SKNet-101-M4',
            ]


train_acc = []
val_acc = []
loss = []
for j in range(len(fnames)):
    f = open(fnames[j]+"_train_acc.txt", 'r')
    a = f.readline().split(' ')[0:-1]
    a = [float(i) for i in a]
    train_acc.append(a)
    f.close()

    f2 = open(fnames[j]+"_val_acc.txt", 'r')
    a = f2.readline().split(' ')[0:-1]
    a = [float(i) for i in a]
    val_acc.append(a)
    f2.close()

    f3 = open(fnames[j]+"_loss.txt", 'r')
    a = f3.readline().split(' ')[0:-1]
    a = [float(i) for i in a]
    loss.append(a)
    f3.close()


#------------------- plot ------------------------
colors = ['red',  'blue', 'black', 'grey', 'purple', 'green', 'cyan', 'magenta', 'orange', 'pink', 'yellow']
lstyle=['-', '-.', '--', '.']

# to plot each setting
for i in range(len(titles)):
    # train accuracy
    plt.figure(figsize=(5,5))
    plt.title(titles[i]+ " (train)")
    for j in range(len(indices[i])):
        x = [k for k in range(1,len(train_acc[indices[i][j]])+1)]
        plt.plot(x, train_acc[indices[i][j]], label=flabels[indices[i][j]], alpha=1., linestyle=lstyle[0], color=colors[j])
    plt.grid()
    plt.legend()
    plt.savefig(titles[i] + "_train.png")
    plt.cla()

    # valid accuracy
    plt.figure(figsize=(5,5))
    plt.title(titles[i]+ " (validation)")
    for j in range(len(indices[i])):
        x = [k for k in range(1,len(val_acc[indices[i][j]])+1)]
        plt.plot(x, val_acc[indices[i][j]], label=flabels[indices[i][j]], alpha=1., linestyle=lstyle[0], color=colors[j])
    plt.grid()
    plt.legend()
    plt.savefig(titles[i] + "_valid.png")
    plt.cla()

"""
for i in range(len(train_acc)):
    # to plot each setting
    plt.figure(figsize=(5,5))
    plt.title(fsettings[i]+" Accuracy")
    x = [k for k in range(1,len(train_acc[i])+1)]
    plt.plot(x, train_acc[i], label="train acc", alpha=1., linestyle=lstyle[i%3], color=colors[0])
    x = [k for k in range(1,len(val_acc[i])+1)]
    plt.plot(x, val_acc[i], label="valid acc", alpha=1., linestyle=lstyle[i%3], color=colors[1])

    plt.grid()
    plt.legend()
    plt.savefig(fsettings[i] + ".png")
    plt.cla()

    plt.figure(figsize=(5,5))
    plt.title(fsettings[i]+" loss")
    x = [k for k in range(1,len(loss[i])+1)]
    plt.plot(x, loss[i], label="loss", alpha=1., linestyle=lstyle[i%3], color=colors[2])
    plt.grid()
    plt.legend()
    plt.savefig(fsettings[i] + "_loss.png")
    plt.cla()
"""