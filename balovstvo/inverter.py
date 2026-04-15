from matplotlib import pyplot as pypl

file1 = 'balovstvo/scattering_Si-crop.txt'
file2 = 'balovstvo/scattering_SiO2-crop.txt'
r = 40
l = 640

Xs1 = []
Ys1 = []
with open(file1, 'r') as f:
    read = f.readline()
    while read != '' and read != ' ':
        xy = read.strip('\n').split(', ')
        Xs1.append(r*l/(float)(xy[0]))
        Ys1.append((float)(xy[1]))
        read = f.readline()

Xs2 = []
Ys2 = []
with open(file2, 'r') as f:
    read = f.readline()
    while read != '' and read != ' ':
        xy = read.strip('\n').split(', ')
        Xs2.append(r*l/(float)(xy[0]))
        Ys2.append((float)(xy[1]))
        read = f.readline()

pypl.plot(Xs1, Ys1, label='Si')
pypl.plot(Xs2, Ys2, label='SiO2')
pypl.xlabel('R')
pypl.ylabel('Scattering cross section')
pypl.show()
