# Projet-recherche

import numpy as np
import matplotlib.pyplot as plt
def euler_flux(U, n: np.array, gamma=1.4):
    assert (
        U.shape[0] == 5
    ), f"Input P must have shape (5) or (5, N) but has shape: {U.shape}"
    rho, u, v, w, p, E, e = conserved_to_primitive(U, gamma)
    q = u * n[0] + v * n[1] + w * n[2]

    return np.vstack(
        (
            rho * q,
            rho * u * q + p * n[0],
            rho * v * q + p * n[1],
            rho * w * q + p * n[2],
            (E + p) * q,
        )
    )


def primitive_to_conserved(U, gamma=1.4):
    rho, u, v, w, p = U
    e = p / (gamma - 1)
    E = e + .5 * rho * (u**2+v**2+w**2)
    return np.vstack((rho, rho*u, rho*v, rho*w, E))

def conserved_to_primitive(U, gamma=1.4):
    
    assert (
        U.shape[0] == 5
    ), f"Input P must have shape (5) or (5, N) but has shape: {U.shape}"
    rho, _u, _v, _w, E = U
    u, v, w = _u / rho, _v / rho, _w / rho
    e = E - .5 * (u**2 + v**2 + w**2) * rho
    p = (gamma - 1.0) * e

    return np.vstack((rho, u, v, w, p, E, e))


#Rusanov


def rusanov_flux(Ul, Ur, n, gamma=1.4):

    Fl = euler_flux(Ul, n, gamma)
    Fr = euler_flux(Ur, n, gamma)

    rhoL, uL, vL, wL, pL, EL, eL = conserved_to_primitive(Ul, gamma)
    rhoR, uR, vR, wR, pR, ER, eR = conserved_to_primitive(Ur, gamma)

    # Vitesse du son
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    # Vitesse maximale caractéristique
    s_max = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

    # Flux de Rusanov
    F = 0.5 * (Fl + Fr) - 0.5 * s_max * (Ur - Ul)

    return F , s_max, s_max


    def roe_averaged_speeds(Ul, Ur, nx, gamma=1.4):
    rhoL, uL, vL, wL, pL, EL, eL = conserved_to_primitive(Ul, gamma)
    rhoR, uR, vR, wR, pR, ER, eR = conserved_to_primitive(Ur, gamma)

    # enthalpies
    HL = (EL + pL) / rhoL
    HR = (ER + pR) / rhoR

    sqrtRhoL = np.sqrt(rhoL)
    sqrtRhoR = np.sqrt(rhoR)
    denom = sqrtRhoL + sqrtRhoR

    uT = (sqrtRhoL * uL + sqrtRhoR * uR) / denom
    vT = (sqrtRhoL * vL + sqrtRhoR * vR) / denom
    wT = (sqrtRhoL * wL + sqrtRhoR * wR) / denom
    HT = (sqrtRhoL * HL + sqrtRhoR * HR) / denom

    unL = uL * nx[0] + vL * nx[1] + wL * nx[2]
    unR = uR * nx[0] + vR * nx[1] + wR * nx[2]
    unT = uT * nx[0] + vT * nx[1] + wT * nx[2]

    vel2T = uT * uT + vT * vT + wT * wT
    cT = np.sqrt((gamma - 1.0) * (HT - 0.5 * vel2T))

    SL = np.minimum(
        unL - np.sqrt((gamma - 1.0) * (HL - 0.5 * (uL * uL + vL * vL + wL * wL))),
        unT - cT,
    )
    SR = np.maximum(
        unR + np.sqrt((gamma - 1.0) * (HR - 0.5 * (uR * uR + vR * vR + wR * wR))),
        unT + cT,
    )
    return SL, SR


#HLL

def HLL(Ul, Ur, n, gamma=1.4):

    SL, SR = roe_averaged_speeds(Ul, Ur, n, gamma)

    Fl = euler_flux(Ul, n, gamma)
    Fr = euler_flux(Ur, n, gamma)

    F = np.where(SR<0, Fr, np.where(SL>0, Fl, (SR * Fl - SL*Fr + SL*SR*(Ur-Ul))/(SR - SL) ))
    return F, SL, SR



    #HLLC
def HLLC(Ul, Ur, n, gamma=1.4):

#Vitesse des ondes
    SL, SR = roe_averaged_speeds(Ul, Ur, n, gamma)

    rho_L, uL, vL, wL, pL, EL, eL = conserved_to_primitive(Ul, gamma)
    rho_R, uR, vR, wR, pR, ER, eR = conserved_to_primitive(Ur, gamma)

    ql = uL * n[0] + vL * n[1] + wL * n[2]
    qr = uR * n[0] + vR * n[1] + wR * n[2]

    unL = uL * n[0] + vL * n[1] + wL * n[2]  # vitesse normale gauche
    unR = uR * n[0] + vR * n[1] + wR * n[2]  # vitesse normale droite

#Flux
    Fl = euler_flux(Ul, n, gamma)
    Fr = euler_flux(Ur, n, gamma)

# Vitesse intermédiaire SM
    SM = (pR - pL + rho_L * unL * (SL - unL) - rho_R * unR * (SR - unR)) / (
        rho_L * (SL - unL) - rho_R * (SR - unR)
    )

# Pression intermédiaire p*
    p_etoile = pL + rho_L * (unL - SM) * (SL - unL)

# États intermédiaires Ul* et Ur*

    rho_etoile_l= rho_L * (SL - unL) / (SL - SM)
    rho_etoile_r = rho_R * (SR - unR) / (SR - SM)
    

    U_etoile_L = np.vstack(
        (
            rho_etoile_l,
            (rho_etoile_l) * (uL + (SM - unL) * n[0]),
            (rho_etoile_l) * (vL + (SM - unL) * n[1]),
            (rho_etoile_l) * (wL + (SM - unL) * n[2]),
            (rho_etoile_l) * (EL / rho_L + (SM - unL) * (SM + pL / (rho_L * (SL - unL)))),
        )
    )

    U_etoile_R = np.vstack(
        (
            rho_etoile_r,
            (rho_etoile_r) * (uR + (SM - unR) * n[0]),
            (rho_etoile_r) * (vR + (SM - unR) * n[1]),
            (rho_etoile_r) * (wR + (SM - unR) * n[2]),
            (rho_etoile_r) * (ER / rho_R + (SM - unR) * (SM + pR / (rho_R * (SR - unR)))),
        )
    )

    

# Flux intermédiaires FL* et Fr*
    F_etoile_l = Fl + SL * (U_etoile_L - Ul)   #Fl* =Fl + SL(Ul*-Ul)
    F_etoile_r = Fr + SR * (U_etoile_R - Ur)   #Fr* =Fr + SR(Ur*-Ur)

#Flux final
    F = np.where(
        SR < 0,
        Fr,
        np.where(SL > 0, Fl, np.where(SM > 0, F_etoile_l, F_etoile_r)),
    )
    return F, SL, SR


    # Roe

def roe_flux(Ul, Ur, n, gamma=1.4):
    rhoL, uL, vL, wL, pL, EL, eL = conserved_to_primitive(Ul, gamma)
    rhoR, uR, vR, wR, pR, ER, eR = conserved_to_primitive(Ur, gamma)

    #Flux
    Fl = euler_flux(Ul, n, gamma)
    Fr = euler_flux(Ur, n, gamma)

    # enthalpies
    HL = (EL + pL) / rhoL
    HR = (ER + pR) / rhoR

    sqrtRhoL = np.sqrt(rhoL)
    sqrtRhoR = np.sqrt(rhoR)
    denom = sqrtRhoL + sqrtRhoR

    uT = (sqrtRhoL * uL + sqrtRhoR * uR) / denom
    vT = (sqrtRhoL * vL + sqrtRhoR * vR) / denom
    wT = (sqrtRhoL * wL + sqrtRhoR * wR) / denom
    HT = (sqrtRhoL * HL + sqrtRhoR * HR) / denom

    unL = uL * n[0] + vL * n[1] + wL * n[2]
    unR = uR * n[0] + vR * n[1] + wR * n[2]
    unT = uT * n[0] + vT * n[1] + wT * n[2]

    vel2T = uT * uT + vT * vT + wT * wT
    cT = np.sqrt((gamma - 1.0) * (HT - 0.5 * vel2T))

    
    
#calcul valeurs propres moyennés
    lambda1_tilde= unT - cT
    lambda2_tilde= unT
    lambda3_tilde= unT
    lambda4_tilde= unT
    lambda5_tilde= unT + cT

# Vecteurs propres de Roe
    K1_tilde = np.vstack((np.ones_like(uT), uT - cT, vT, wT, HT - uT * cT))
    K2_tilde = np.vstack((np.ones_like(uT), uT, vT, wT, 0.5 * (uT**2 + vT**2 + wT**2)))
    K3_tilde = np.vstack((np.zeros_like(uT), np.zeros_like(uT), np.ones_like(uT), np.zeros_like(uT), vT))
    K4_tilde = np.vstack((np.zeros_like(uT), np.zeros_like(uT), np.zeros_like(uT), np.ones_like(uT), wT))
    K5_tilde = np.vstack((np.ones_like(uT), uT + cT, vT, wT, HT + uT * cT))


    # Différences d’état
    delta_U = Ur - Ul

    # Coefficients de projection sur les vecteurs propres
    alpha2 = (gamma - 1) / (cT**2) * (delta_U[0] * (HT - unT**2) + unT * delta_U[1] - delta_U[4])
    alpha1 = (1 / (2 * cT)) * (delta_U[0] * (unT + cT) - delta_U[1] - cT * alpha2)
    alpha3 = delta_U[2] - vT * delta_U[0]
    alpha4 = delta_U[3] - wT * delta_U[0]
    alpha5 = delta_U[0] - (alpha1 + alpha2)

    # Flux numérique de Roe
    flux_Roe = 0.5 * (Fl + Fr) - 0.5 * (
        np.abs(lambda1_tilde) * alpha1 * K1_tilde +
        np.abs(lambda2_tilde) * alpha2 * K2_tilde +
        np.abs(lambda3_tilde) * alpha3 * K3_tilde +
        np.abs(lambda4_tilde) * alpha4 * K4_tilde +
        np.abs(lambda5_tilde) * alpha5 * K5_tilde
    )

    return flux_Roe,cT,cT

    def evolve1d(dx, U0, T, gamma=1.4, CFL=0.5):

    U = U0.copy()
    t = 0
    while t < T:

        F, SL, SR = roe_flux(U[:, :-1], U[:, 1:], np.array([1, 0, 0]), gamma)  #modifier la methode

        max_speed = np.max(np.abs(np.hstack((SL, SR))))
        dtCFL = CFL * dx / max_speed if max_speed > 1e-14 else 1e-6
        dt = min(T - t, dtCFL)

        U[:, 1:-1] -= (dt / dx) * (F[:, 1:] - F[:, :-1])
        U[:, 0] = U[:,1]
        U[:, -1] = U[:,-2]
        t += dt

    return U

    def initialize_test(test_number, xh):
    test_cases = {
        1: [1.0, 0.75, 1.0, 0.125, 0.0, 0.1],
        2: [1.0, -2.0, 0.4, 1.0, 2.0, 0.4],
        3: [1.0, 0.0, 1000.0, 1.0, 0.0, 0.01],
        4: [5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950],
        5: [1.0, -19.59745, 1000.0, 1.0, -19.59745, 0.01],
    }

    rhoL, uL, pL, rhoR, uR, pR = test_cases[test_number]

    U = np.zeros((5, len(xh)))
    U[0] = np.where(xh < 0, rhoL, rhoR)  # Densité
    U[1] = np.where(xh < 0, uL, uR)  # Vitesse u
    U[4] = np.where(xh < 0, pL, pR)  # Pression

    return primitive_to_conserved(U)

    CFL = 0.5
T = 0.2
#T=0.012

nx = 400
x = np.linspace(-.2, .5, nx)
#x = np.linspace(0, 1, nx)

xh = 0.5 * (x[1:] + x[:-1])
dx = xh[1] - xh[0]

n = np.array([1, 0, 0])

# Condition initiale

test_number = 1 
U = initialize_test(test_number, xh)


U = evolve1d(dx, U, T, gamma=1.4, CFL=CFL)

plt.figure(figsize=(10,10))

ax = plt.subplot(2,2,1)
ax.plot(xh, U[0], "-", label="methode")
ax.set_xlabel("espace")
ax.set_ylabel("densité (rho)")
ax.legend()

ax = plt.subplot(2,2,2)
ax.plot(xh, conserved_to_primitive(U)[1], "-", label="methode")
ax.set_xlabel("espace")
ax.set_ylabel("vitesse (u)")
ax.legend()

ax = plt.subplot(2,2,3)
ax.plot(xh, conserved_to_primitive(U)[4], "-", label="methode")
ax.set_xlabel("espace")
ax.set_ylabel("Pression (p)")
ax.legend()

ax = plt.subplot(2,2,4)
ax.plot(xh, conserved_to_primitive(U)[6]/U[0], "-", label="methode")
ax.set_xlabel("espace")
ax.set_ylabel("Energie totale(E)")
ax.legend()
