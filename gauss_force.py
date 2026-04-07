import numpy as np
from dipoles import alpha_v2
import green_func_v2
from force import field_dx, field_dz, cached_green_functions

# ureg = pint.UnitRegistry()

# with open("output/a.txt", 'w') as file:
#     file.write('')

# for z_b in range(100, 2000, 100):
#     cfg = SimulationConfig(
#         wl=640 * ureg.nanometer,
#         R= 50 * ureg.nanometer,ulationConfig, OpticalForceCalculator
# from force import F
#         dist=0,
#         angle=0,
#         z0=z_b * ureg.nanometer,
#         psi=0,
#         chi=0,
#         substrate='Vacuum',
#         particle='SiO2',
#         amplitude=1,
#         initial_field_type='custom'
#     )
#     forces = OpticalForceCalculator(config=cfg).compute()
    
#     # dipoles = DipoleCalculator(config=cfg).compute()
#     print(forces)
#     with open("output/a.txt", 'a') as file:
#         file.write(str(z_b) + ';' + str(forces.Fz[0]) + '\n')


c_const = 3E8
eps0_const = 1 / (4 * np.pi * c_const ** 2) * 1e7
mu0_const = 4 * np.pi * 1e-7


def gaussian_beam(wl, w0, amplitude, point, z_beam):
    k = 2*np.pi/wl
    omega = 2*np.pi*c_const/wl
    
    zR = np.pi*w0**2 / wl

    def w(z_coord): return w0*np.sqrt(1 + (z_coord / zR)**2)
    def R(z_coord): return z_coord * (1 + (zR / z_coord)**2)
    def psi(z_coord): return np.arctan(z_coord / zR)

    def E(at_point):
        x0, y0, z0 = at_point
        r0 = np.sqrt(x0**2 + y0**2)
        Ex = amplitude * w0/w(z0 - z_beam) * \
        np.exp( -r0**2 / w(z0 - z_beam)**2 ) * \
        np.exp( 1j* ( k*(z0 - z_beam) + k * r0**2 / 2 / R(z0 - z_beam) - psi(z0 - z_beam) ) )

        return np.array([Ex, 0, 0], dtype=complex)\

    def H(at_point):
        step = 1E-9

        x0, y0, z0 = at_point
        pxp = (x0 + step, y0, z0)
        pxm = (z0 - step, y0, z0)

        pyp = (x0, y0 + step, z0)
        pym = (x0, y0 - step, z0)

        pzp = (x0, y0, z0 + step)
        pzm = (x0, y0, z0 - step)


        _, Ey_pxp, Ez_pxp = E(pxp)
        _, Ey_pxm, Ez_pxm = E(pxm)

        Ex_pyp, _, Ez_pyp = E(pyp)
        Ex_pym, _, Ez_pym = E(pym)

        Ex_pzp, Ey_pzp, _ = E(pzp)
        Ex_pzm, Ey_pzm, _ = E(pzm)
        
        dEy_dx = (Ey_pxp - Ey_pxm) / (2 * step)
        dEz_dx = (Ez_pxp - Ez_pxm) / (2 * step)

        dEx_dy = (Ex_pyp - Ex_pym) / (2 * step)
        dEz_dy = (Ez_pyp - Ez_pym) / (2 * step)

        dEx_dz = (Ex_pzp - Ex_pzm) / (2 * step)
        dEy_dz = (Ey_pzp - Ey_pzm) / (2 * step)

        curl_x = dEz_dy - dEy_dz
        curl_y = dEx_dz - dEz_dx
        curl_z = dEy_dx - dEx_dy

        return np.array([curl_x, curl_y, curl_z], dtype=complex)*(-1j) / omega /mu0_const
    
    resE = E(point)
    resH = H(point)

    return resE, resH


def calc_dipoles_v2(wl, eps_Au, point, R, eps_Si, w0, amplitude, z0, green_func_type=None):
    mu = 1
    eps = 1
    k = 2*np.pi/wl/1e-9
    omega = 2*np.pi*c_const/wl/1e-9
    _, _, z0 = point
    alpha_e, alpha_m = alpha_v2(wl, R, eps_Si)
    
    
    G_ref_E, rot_G_ref_H, G_ref_H, rot_G_ref_E = green_func_v2.getG(wl, eps_Au, 2*z0, 0, 0, green_func_type)
    # (G_ref_E, G_ref_H, rot_G_ref_E, rot_G_ref_H) = cached_green_functions(
    #     wl, z0, eps_Au)
    # if initial_field_type == 'plane_wave':
    #     E0, H0 = initial_field(wl, alpha, amplitude, eps_Au, point, phase, a_angle)
    # elif initial_field_type == 'two_beam':
    #     E0, H0 = field_two_beam_setup(wl, alpha, amplitude, eps_Au, point, phase, a_angle)
    # elif initial_field_type == 'custom':
    #     E0, H0 = custom_field(wl, alpha, amplitude, eps_Au, point, phase, a_angle, z0)
    # else:
    #     raise ValueError("Invalid initial_field_type. Choose from 'plane_wave', 'two_beam', or 'custom'.")
        
    E0, H0 = gaussian_beam(wl, w0, amplitude, point, z0)

    G_ee = mu*k**2/eps0_const * G_ref_E
    G_em = 1j*omega*mu*mu0_const * rot_G_ref_H
    G_me = -1j*omega*rot_G_ref_E
    G_mm = eps*mu*k**2*G_ref_H

    I = np.eye(3, dtype=np.complex128)

    A = I - eps0_const * alpha_e * G_ee - eps0_const * alpha_e * \
        G_em @ np.linalg.inv(I - alpha_m * G_mm) * alpha_m @ G_me
    B = I - alpha_m * G_mm - alpha_m * \
        G_me @ np.linalg.inv(I - eps0_const * alpha_e *
                             G_ee) * eps0_const * alpha_e @ G_em

    Am1 = np.linalg.inv(A)
    Bm1 = np.linalg.inv(B)

    alpha_ee = Am1 * alpha_e
    alpha_mm = Bm1 * alpha_m

    alpha_em = Am1 * eps0_const * \
        alpha_e @ G_em @ np.linalg.inv(I - alpha_m * G_mm) * alpha_m
    alpha_me = Bm1 * \
        alpha_m @ G_me @ np.linalg.inv(I - eps0_const *
                                       alpha_e * G_ee) * eps0_const * alpha_e

    p = eps0_const * alpha_ee @ E0 + alpha_em @ H0
    m = alpha_mm @ H0 + alpha_me @ E0

    return p, m


def F(wl, eps_Au, point, R, eps_Si, w0, amplitude, z0, stop, full_output=False, effective_dipoles_in_air=False, effective_dipoles_substrate = None):
    mu=1
    eps=1
    k = 2*np.pi/wl/1e-9
    omega = 2*np.pi*c_const/wl/1e-9
    _,_,z0=point
    
    if effective_dipoles_in_air == True:
        dip = calc_dipoles_v2(wl=wl,
                              eps_Au=effective_dipoles_substrate,
                              point=point,
                              R=R,
                              eps_Si=eps_Si,
                              w0=w0,
                              amplitude=amplitude,
                              z0=z0)
    else:
        dip = calc_dipoles_v2(wl=wl,
                              eps_Au=eps_Au,
                              point=point,
                              R=R,
                              eps_Si=eps_Si,
                              w0=w0,
                              amplitude=amplitude,
                              z0=z0)
        
    p = dip[0][:,0]
    m = dip[1][:,0]
    
    # if initial_field_type == 'two_beam':
    #     field0 = dipoles.field_two_beam_setup
    # elif initial_field_type == 'plane_wave':
    #     field0 = dipoles.initial_field
    # elif initial_field_type == 'custom':
    #     field0 = dipoles.custom_field
    # else:
    #     raise ValueError("Invalid initial_field_type. Choose from 'plane_wave', 'two_beam', or 'custom'.")

    dE_dx, dH_dx = field_dx(gaussian_beam, wl, 0, amplitude, eps_Au, point, 0, 0, z0)
    dE_dz, dH_dz = field_dz(gaussian_beam, wl, 0, amplitude, eps_Au, point, 0, 0, z0)


    (dx_G_E, dx_G_H, dx_rot_G_E, dx_rot_G_H,
     dy_G_E, dy_G_H, dy_rot_G_E, dy_rot_G_H,
     dz_G_E, dz_G_H, dz_rot_G_E, dz_rot_G_H) = cached_green_functions(wl, z0, eps_Au, stop)
    
    F_crest = - k**4/(12*np.pi*c_const*eps0_const) * np.real( np.cross(p, np.conj(m)))
    
    Fy_e1 = 0.5*mu*k**2/eps0_const * np.real( np.conj(p) @ (dy_G_E @ p))
    Fy_e2 = 0.5*omega*mu*mu0_const * np.real( 1j* np.conj(p) @ (dy_rot_G_H @ m))
    Fy_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real( np.conj(m) @ (dy_G_H @ m))
    Fy_m2 = - 0.5 * mu*mu0_const * omega * np.real(1j* np.conj(m) @ (dy_rot_G_E @ p) )
    F_y = Fy_e1 + Fy_e2 + Fy_m1 + Fy_m2 + F_crest[1]
    
    Fx_e0 = 0.5*np.real(np.conj(p) @ dE_dx)
    Fx_e1 = 0.5*mu*k**2/eps0_const * np.real(np.conj(p) @ (dx_G_E @ p))
    Fx_e2 = 0.5*omega*mu*mu0_const * np.real(1j*np.conj(p) @ (dx_rot_G_H @ m))
    Fx_m0 = 0.5*np.real(np.conj(m) @ dH_dx)*mu*mu0_const    
    Fx_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real(np.conj(m)@ (dx_G_H @ m))
    Fx_m2 = -0.5*mu*mu0_const*omega*np.real(1j*np.conj(m) @ (dx_rot_G_E @ p))
    F_x = Fx_e1 + Fx_e2 + Fx_e0 + Fx_m0 + Fx_m1 + Fx_m2 + F_crest[0]
    
    
    
    Fz_e1 = 0.5*mu*k**2/eps0_const * np.real(np.conj(p) @ (dz_G_E @ p))
    Fz_e2 = 0.5*omega*mu*mu0_const * np.real(1j*np.conj(p) @ (dz_rot_G_H @ m))
    Fz_e0 = 0.5*np.real(np.conj(p) @ dE_dz)
    Fz_m1 = 0.5*mu**2*mu0_const*eps*k**2 * np.real(np.conj(m)@ (dz_G_H @ m))
    Fz_m2 = -0.5*mu*mu0_const*omega*np.real(1j*np.conj(m) @ (dz_rot_G_E @ p))
    Fz_m0 = 0.5*np.real(np.conj(m) @ dH_dz)*mu*mu0_const
    F_z = Fz_e1 + Fz_e2 + Fz_e0 + Fz_m0 + Fz_m1 + Fz_m2 + F_crest[2]
    
    if full_output==False:
        return F_x, F_y, F_z
    else:
        Fx = [F_x, Fx_e0, Fx_e1, Fx_e2, Fx_m0, Fx_m1, Fx_m2, F_crest[0]]
        Fy = [F_y, 0, Fy_e1, Fy_e2, 0, Fy_m1, Fy_m2, F_crest[1]]
        Fz = [F_z, Fz_e0, Fz_e1, Fz_e2, Fz_m0, Fz_m1, Fz_m2, F_crest[2]]
        return np.array([Fx, Fy, Fz])
