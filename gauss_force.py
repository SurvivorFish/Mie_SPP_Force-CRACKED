from fields import gaussian_beam
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

wavelen = 640E-9
waist = 450E-9

x_min, x_max = -2*wavelen, 2*wavelen
z_min, z_max = -2*wavelen, 2*wavelen


def normE(x, z):
    E = gaussian_beam(wl=wavelen, alpha=0, amplitude=1, eps_interp='Air', point=(x,0,z), w0=waist, z_beam=0)[0]
    return np.sqrt(np.abs(E[0])**2 + np.abs(E[1])**2 + np.abs(E[2])**2)[0]

def Ex(x, z):
    E = gaussian_beam(wl=wavelen, alpha=0, amplitude=1, eps_interp='Air', point=(x,0,z), w0=waist, z_beam=0)[0]
    return np.abs(E[0])[0]

def Ez(x, z):
    E = gaussian_beam(wl=wavelen, alpha=0, amplitude=1, eps_interp='Air', point=(x,0,z), w0=waist, z_beam=0)[0]
    return np.abs(E[2])[0]


#  point   | Ex (code)       | abs(Ebx) (comsol)
# (0, 0)     0.9982153015247032   1
# (0, 50nm): 0.9982153015247034, 0.99874
# (100, 0)   0.9509594372090185, 0.95182
# (1000, 0)  0.0077457745365089  0.007167
# (0, 1000)  0.9982153015247033, 0.70498

#  point   | normE (code)       | normE (comsol)
# (0, 0)     0.9982153015247032   0.85747
# (0, 50nm): 0.8432306842472972, 0.85483
# (100, 0)   0.7908565756340074, 0.82924

num_x = 20
num_z = 20
x_vals = np.linspace(x_min, x_max, num_x)
z_vals = np.linspace(z_min, z_max, num_z)


Z = np.zeros((num_z, num_x))


for i in tqdm(range(num_x), desc="Столбцы X"):
    for j in range(num_z):
        Z[j, i] = normE(x_vals[i], z_vals[j])

# Визуализация
plt.figure(figsize=(8, 6))
plt.contourf(x_vals, z_vals, Z, levels=50, cmap='viridis')
plt.colorbar(label='Значение')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Скалярное поле (вычислено через циклы)')
plt.show()