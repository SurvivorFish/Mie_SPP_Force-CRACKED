import matplotlib.pyplot as plt

Xs = []
Ys = []

with open('output/' + 'a.txt', 'r') as file:
    line = file.readline()
    while line != '':
        output = line.split(';')
        Xs.append(output[0])
        Ys.append(output[1])
        line = file.readline()

    
plt.scatter(Xs, Ys)
# plt.savefig("output/hehehe.png", dpi=350)
plt.show()