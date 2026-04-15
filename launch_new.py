from numpy import pi
import pint
from simulation import SimulationConfig, OpticalForceCalculator, DipoleCalculator, SweepRunner
from comsol_data import parse_file
import numpy as np
import matplotlib.pyplot as plt
ureg = pint.UnitRegistry()

wavelen = 640
rad = 10
baseConfig = SimulationConfig(
    wl=wavelen * ureg.nanometer,
    R=rad * ureg.nanometer,
    dist=1 * ureg.nanometer,
    angle=np.deg2rad(0),
    psi=pi/2,
    chi=pi/4,
    substrate='Air',
    particle='SiO2',
    amplitude=1,
    show_warnings=False,
    initial_field_type='custom',
    z_beam= 1E-9 * ureg.nanometer,
    w0=450*ureg.nanometer,
    x_part=0*ureg.nanometer
)

# dips = OpticalForceCalculator(baseConfig).compute()

# R_arr = np.linspace(10, 170, 50) * ureg.nanometer
# z_beam_arr = np.linspace(-2*wavelen, 2*wavelen, 80) * ureg.nanometer

# res, _, _ = SweepRunner(baseConfig, 'z_beam', z_beam_arr, compute_force=True).run()

x_part_arr = np.linspace(-wavelen, wavelen, 20) * ureg.nanometer
res, _, _ = SweepRunner(baseConfig, 'x_part', x_part_arr, compute_force=True).run()

with open('code_fx_wiki.csv', 'w') as f:
    f.write("x_part,Fx\n")
    for i in range(len(res.Fx)):
        f.write(str(x_part_arr[i]))
        f.write(',')
        f.write(str(res.Fx[i]))
        f.write('\n')




# plt.plot(res.x_part, res.Fx, label='Fx (code)')

# datax = parse_file("/home/uspensky/ComsolData/wl640nm_fx_SiO2_r10-120nm_nosubs.csv")

# i = rad // 10 - 1
# plt.plot(1E9*datax[i]["x_part (m)"], datax[i]["F_x (N)"], label=f"Fx (comsol) ({datax[i]["% r_part (m)"].iloc[0]*1E9} nm)")

# plt.xlabel("x_part (nm)")
# plt.ylabel("Fx (N)")
# plt.legend()
# plt.title(f'R={rad}nm')
# plt.grid()
# plt.show()

