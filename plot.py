import matplotlib.pyplot as plt
import re

log_file = 'log/epoch_50_lr_0.0002/log_total.txt'
epoch =[]
g_loss = []
d_loss = []
val_loss = []
with open(log_file, 'r') as log:
    lines = log.readlines()
for line in lines:
    if 'train' in line:
        line = re.split('\[|\]|,|:|/',line)
        epoch.append(int(line[1]))
        g_loss.append(float(line[6]))
        d_loss.append(float(line[8]))
        val_loss.append(float(line[10]))
fig = plt.figure()
plt.plot(epoch, g_loss)
plt.plot(epoch, d_loss)
plt.plot(epoch, val_loss)
plt.legend(['g_loss', 'd_loss', 'val_loss'])
plt.savefig(fname='total.png')