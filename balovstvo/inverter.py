from matplotlib import pyplot as pypl

file = 'balovstvo/scattering1-crop.txt'
r = 40
l = 640

Xs = []
Ys = []
with open(file, 'r') as f:
    read = f.readline()
    while read != '' and read != ' ':
        xy = read.strip('\n').split(', ')
        Xs.append(r*l/(float)(xy[0]))
        Ys.append((float)(xy[1]))
        read = f.readline()

pypl.plot(Xs, Ys)
pypl.xlabel('R')
pypl.ylabel('Scattering cross section')
pypl.show()
