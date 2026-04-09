import dipoles, green_func_v2
import numpy as np
import frenel
from scipy.integrate import quad
from numpy import sin, cos
import green_func_v2

c_const = 299792458
eps0_const = 1/(4*np.pi*c_const**2)*1e7
mu0_const = 4*np.pi * 1e-7
sqrtpi = np.sqrt(np.pi)


def get_field(wl, eps_interp, alpha, phase, a_angle, eps_particle, R,   r, phi, z, z0, field_type = None, amplitude=1, initial_field_type=None, 
              z_beam = None, w0 = None):
    
    assert z>= 0, "z should be >=0"
    assert z0>0, "z0 should be >0"
    
    k = 2*np.pi/wl*1e9
    omega = k*c_const
    
    GEres =np.zeros((3,3), dtype=complex)
    rotGHres = np.zeros_like(GEres)
    GHres = np.zeros_like(GEres)
    rotGEres = np.zeros_like(GEres)
    
    p,m = dipoles.calc_dipoles_v2(wl, eps_interp, [0,0,z0], R, eps_particle, alpha, amplitude, phase, a_angle, initial_field_type=initial_field_type, w0=w0, z_beam=z_beam)

    G0, rotG0 = green_func_v2.G0(wl, z0, r, phi, z)
    GE_spp, rotGH_spp, GH_spp, rotGE_spp = green_func_v2.getG(wl, eps_interp, z+z0, r, phi, 'spp')
    GE_reg, rotGH_reg, GH_reg, rotGE_reg = green_func_v2.getG(wl, eps_interp, z+z0, r, phi, 'reg')
    # GE, rotGH, GH, rotGE = green_func_v2.getG(wl, eps_interp, z+z0, r, phi)
    
    if field_type == 'spp':
        GEres, rotGHres, GHres, rotGEres = GE_spp, rotGH_spp, GH_spp, rotGE_spp
    elif field_type == 'sc':
        GEres, rotGHres, GHres, rotGEres = GE_reg, rotGH_reg, GH_reg, rotGE_reg
    elif field_type == 'air':
        GEres, rotGHres, GHres, rotGEres = G0, rotG0, G0, rotG0
    elif field_type == 'reg':
        GEres, rotGHres, GHres, rotGEres = GE_reg+G0, rotGH_reg+rotG0, GH_reg+G0, rotGE_reg+rotG0
    else:
        GEres, rotGHres, GHres, rotGEres = GE_spp+GE_reg+G0, rotGH_reg+rotGH_spp+rotG0, GH_reg+GH_spp+G0, rotGE_reg+rotGE_spp+rotG0


    E =  k**2/eps0_const * GEres @ p + 1j*omega*mu0_const* rotGHres @m
    H =  k**2 * GHres  @ m - 1j*omega*rotGEres @ p
    return E[:,0],H[:,0]


def gaussian_beam(wl, alpha, amplitude, eps_interp, point, w0, z_beam):
    # rp, rs = frenel.reflection_coeff_v2(wl, eps_interp, alpha)
    rp = 0
    k = 2*np.pi/wl
    # kx = k * np.sin(alpha)
    # kz = k * np.cos(alpha)
    
    omega = 2*np.pi*c_const/wl

    def electric_field(wl, point):
        x0, _, z0 = point

        f = z_beam
        # def integrand_re(k_x, sign):
        #     return (kz*(1-sign) + k_x*sign ) / k * np.exp(-k_x**2 * w0**2 / 4) * (cos(kz*f)*(  cos(kz*z0)*cos(k_x*x0) + sin(kz*z0)*sin(k_x*x0)  ) + \
        #                                                     sin(kz*f)*(  sin(kz*z0)*cos(k_x*x0) - cos(kz*z0)*sin(k_x*x0)  )  + \
        #                                                     (-1)**(sign)*rp * (cos(kz*f)*(  -cos(kz*z0)*cos(k_x*x0) + sin(kz*z0)*sin(k_x*x0)  ) + \
        #                                                           sin(kz*f)*(  sin(kz*z0)*cos(k_x*x0) + cos(kz*z0)*sin(k_x*x0)  )))
        
        # def integrand_im(k_x, sign):
        #     return (kz*(1-sign) + k_x*sign ) / k * np.exp(-k_x**2 * w0**2 / 4) * ( -cos(kz*f)*(  sin(kz*z0)*cos(k_x*x0) + cos(kz*z0)*sin(k_x*x0)  ) + \
        #                                                        sin(kz*f) * (  cos(kz*z0)*cos(k_x*x0) + sin(kz*z0)*sin(k_x*x0)  ) + \
        #                                                        (-1)**(sign)*rp * (cos(kz*f) * (  -sin(kz*z0)*cos(k_x*x0) + cos(kz*z0)*sin(k_x*x0)  ) + \
        #                                                              sin(kz*f) * (  -sin(kz*z0)*cos(k_x*x0) + cos(kz*z0)*sin(k_x*x0)  )))

        def integrand_re(k_x, sign):
            kz = np.sqrt(k**2 - k_x**2)
            return (kz*(1-sign) + k_x*sign ) / k * np.exp(-k_x**2 * w0**2 / 4) * \
                np.real( np.exp(1j*(kz*f + k_x*x0)) * (np.exp(-1j*kz*z0) - (-1)**sign*rp * np.exp(1j*kz*z0)) )
        
        def integrand_im(k_x, sign):
            kz = np.sqrt(k**2 - k_x**2)
            return (kz*(1-sign) + k_x*sign ) / k * np.exp(-k_x**2 * w0**2 / 4) * \
                np.imag( np.exp(1j*(kz*f + k_x*x0)) * (np.exp(-1j*kz*z0) - (-1)**sign*rp * np.exp(1j*kz*z0)) )

        Ex = amplitude * w0 / (2 * sqrtpi) * (quad(integrand_re, -k, k, args=(0))[0] + 1j * quad(integrand_im, -k, k, args=(0))[0])
        Ez = amplitude * w0 / (2 * sqrtpi) * (quad(integrand_re, -k, k, args=(1))[0] + 1j * quad(integrand_im, -k, k, args=(1))[0])
        
        E0 = np.array([Ex, 0, Ez], dtype=complex)
        
        return E0
    
    def magnetic_field(electric_field, wl, point, step_nm=1):
        
        omega = 2*np.pi*c_const/wl/1e-9
        
        h_m = step_nm * 1e-9
        x0, y0, z0 = point

        def E(p):
            return electric_field(wl, p).flatten()

        pxp = (x0 + step_nm, y0, z0)
        pxm = (x0 - step_nm, y0, z0)

        pyp = (x0, y0 + step_nm, z0)
        pym = (x0, y0 - step_nm, z0)

        pzp = (x0, y0, z0 + step_nm)
        pzm = (x0, y0, z0 - step_nm)


        _, Ey_pxp, Ez_pxp = E(pxp)
        _, Ey_pxm, Ez_pxm = E(pxm)

        Ex_pyp, _, Ez_pyp = E(pyp)
        Ex_pym, _, Ez_pym = E(pym)

        Ex_pzp, Ey_pzp, _ = E(pzp)
        Ex_pzm, Ey_pzm, _ = E(pzm)
        
        dEy_dx = (Ey_pxp - Ey_pxm) / (2 * h_m)
        dEz_dx = (Ez_pxp - Ez_pxm) / (2 * h_m)

        dEx_dy = (Ex_pyp - Ex_pym) / (2 * h_m)
        dEz_dy = (Ez_pyp - Ez_pym) / (2 * h_m)

        dEx_dz = (Ex_pzp - Ex_pzm) / (2 * h_m)
        dEy_dz = (Ey_pzp - Ey_pzm) / (2 * h_m)

        curl_x = dEz_dy - dEy_dz
        curl_y = dEx_dz - dEz_dx
        curl_z = dEy_dx - dEx_dy

        return np.array([curl_x, curl_y, curl_z], dtype=complex)*(-1j) / omega /mu0_const
    
    E = electric_field(wl, point)
    H = magnetic_field(electric_field, wl, point)
    
    return np.array([[E[0]], [E[1]],[E[2]]], dtype=complex), np.array([[H[0]], [H[1]], [H[2]]], dtype=complex)
