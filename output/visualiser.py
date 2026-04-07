import matplotlib.pyplot as plt

Xs = []
Ys1 = []
Ys2 = []

with open('output/' + 'a.txt', 'r') as file:
    line = file.readline()
    while line != '':
        output = line.split(';')
        Xs.append(output[0])
        Ys1.append(output[2])
        Ys2.append(output[1])
        line = file.readline()

    
plt.plot(Xs, Ys1)
plt.show()