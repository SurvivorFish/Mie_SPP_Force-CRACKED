from numpy import pi
import pint
from simulation import SimulationConfig, OpticalForceCalculator, DipoleCalculator


ureg = pint.UnitRegistry()

cfg = SimulationConfig(
    wl=640 * ureg.nanometer,
    R=20 * ureg.nanometer,
    dist=10 * ureg.nanometer,
    angle=45,
    psi=pi/2,
    chi=pi/4,
    substrate='Au',
    particle='Si',
    amplitude=1
)

forces = OpticalForceCalculator(config=cfg).compute()
# dipoles = DipoleCalculator(config=cfg).compute()
print(forces)
