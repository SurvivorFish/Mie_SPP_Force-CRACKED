from numpy import pi
import pint
from simulation import SimulationConfig, OpticalForceCalculator, DipoleCalculator


ureg = pint.UnitRegistry()

for r in range(10, 170, 10):
    cfg = SimulationConfig(
        wl=640 * ureg.nanometer,
        R=r * ureg.nanometer,
        dist=10 * ureg.nanometer,
        angle=45,
        psi=pi/2,
        chi=pi/4,
        substrate='Vacuum',
        particle='SiO2',
        amplitude=1
    )
    forces = OpticalForceCalculator(config=cfg).compute()
    
    # dipoles = DipoleCalculator(config=cfg).compute()
    with open('output/' + 'a.txt', 'a') as file:
        file.write((str)(r) + ';' + (str)(forces.Fx[0]) + ';' + (str)(forces.Fz[0]) + '\n')


