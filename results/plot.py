import matplotlib.pyplot as plt


fnames = ['ucf101_20200904-033403']
fsettings = ['3D-SK-Net']


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
colors = ['red',  'blue', 'black', 'grey', 'purple', 'green', 'cyan', 'magenta']
lstyle=['-', '-.', '--']

for i in range(len(train_acc)):
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
