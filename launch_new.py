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
    z_beam= 0 * ureg.nanometer,
    w0=wavelen/2*ureg.nanometer
)

# dips = OpticalForceCalculator(baseConfig).compute()

# R_arr = np.linspace(10, 170, 50) * ureg.nanometer
z_beam_arr = np.linspace(-2*wavelen, 2*wavelen, 80) * ureg.nanometer

res, _, _ = SweepRunner(baseConfig, 'z_beam', z_beam_arr, compute_force=True).run()

plt.plot(-res.z_beam, res.Fz, label='Fz (code)')
print(res.px)

dataz = parse_file("/home/uspensky/ComsolData/wl640nm_fz_SiO2_r10-100nm_nosubs.csv")

i = rad // 10 - 1
# plt.plot(-1E9*dataz[i]["z0 (m)"], dataz[i]["F_z (N)"], label=f"Fz (comsol) ({dataz[i]["% r_part (m)"].iloc[0]*1E9} nm)")

plt.xlabel("z_part (nm)")
plt.ylabel("Fz (N)")
plt.legend()
plt.title(f'R={rad}nm')
plt.grid()
plt.show()

