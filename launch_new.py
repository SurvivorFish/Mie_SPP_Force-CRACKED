from numpy import pi
import pint
from simulation import SimulationConfig, OpticalForceCalculator, DipoleCalculator, SweepRunner
from comsol_data import parse_file
import pandas as pd
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
    amplitude=0.1444,
    show_warnings=False,
    initial_field_type='custom',
    z_beam= 1E-12 * ureg.nanometer,
    w0=450*ureg.nanometer,
    xp=0*ureg.nanometer
)

# Для x
# x_part_arr = np.linspace(-wavelen, wavelen, 40) * ureg.nanometer
# res, _, _ = SweepRunner(baseConfig, 'xp', x_part_arr, compute_force=True).run()
# data_our = parse_file("/home/uspensky/ComsolData/wl640nm_x_SiO2_r10-100nm_nosubs.csv")
# i = rad // 10 - 1
# plt.plot(x_part_arr, res.Fx, label=f'Fx (code) ({rad} nm)')
# plt.plot(1E9*data_our[i]["xp (m)"], data_our[i]["Fx (N)"], label=f"Fx (comsol) ({round(data_our[i]["% r (m)"].iloc[0]*1E9)} nm)")
# plt.xlabel("x_part (nm)")
# plt.ylabel("Fx (N)")

# Для z
i = rad // 10 - 1
data_mph = pd.read_csv("/home/uspensky/ComsolData/wl640nm_z_SiO2_r10nm_nosubs.csv", header=4)
plt.plot(-1E9*data_mph["f (m)"], data_mph["Fz (N)"], label=f"Fz (comsol) (10 nm)")
z_beam_arr = np.linspace(-2*wavelen, wavelen, 40) * ureg.nanometer
res, _, _ = SweepRunner(baseConfig, 'z_beam', z_beam_arr, compute_force=True).run()
plt.plot(-z_beam_arr, res.Fz, label=f'Fz (code) ({rad} nm)')
# plt.plot(-1E9*data_mph[i]["zp (m)"], data_mph[i]["Fz (N)"], label=f"Fz (comsol) ({round(data_mph[i]["% r (m)"].iloc[0]*1E9)} nm)")

plt.xlabel("zp (nm)")
plt.ylabel("Fz (N)")

# data_mph = parse_file("/home/uspensky/ComsolData/wl640nm_z_SiO2_r10-90nm_nosubs.csv")
# for j in range(3):
#     plt.plot(1E9*data_mph[j]["zp (m)"], data_mph[j]["Fz (N)"], label=f"Fz (comsol) ({round(data_mph[j]["% r (m)"].iloc[0]*1E9)} nm)")
    

plt.legend()
plt.title(f'R={rad}nm')
plt.grid()
plt.show()
