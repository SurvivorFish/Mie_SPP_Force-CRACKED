import numpy as np
from gauss_force import F
import frenel
import pint
import matplotlib.pyplot as plt

Zgrid = np.linspace(100E-9, 1000E-9, 20)
Fzvals = np.zeros_like(Zgrid)

ureg=pint.UnitRegistry()
WAVELEN = 640 * ureg.nanometer
no_subs = lambda wl: 1+0*1j 
Si_data = frenel.get_interpolate('SiO2')


for i in range(len(Zgrid)):
    Fzvals[i] = F(wl=WAVELEN,
                 eps_Au=no_subs,
                 point=np.array([0, 0, 0]),
                 R=70*ureg.nanometer,
                 eps_Si=Si_data,
                 w0=450*ureg.nanometer,
                 amplitude=1,
                 z0=Zgrid[i]*ureg.nanometer,
                 stop=45,
                 full_output=False)[2]

plt.plot(Zgrid, Fzvals)
plt.show()
