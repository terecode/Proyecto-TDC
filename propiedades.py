import numpy as np

# =======================
#   GEOMETRÍA DEL PANEL
# =======================
L_vert = 2.416
L_horiz = 1.09
Ancho_total = L_horiz

# Longitudes características
A = L_vert * L_horiz
P = 2*(L_vert + L_horiz)

Lc_AP = A / P          # Natural
Lc_L  = L_horiz        # Forzada


# =============================================
#   PROPIEDADES AMBIENTE / CONSTANTES
# =============================================
g = 9.81
T_amb = 20.0 + 273.15
T_cielo = 230.0
T_suelo = 20.0 + 273.15
sigma = 5.67e-8
emisividad = 0.85

V_viento = 1.0
theta = np.radians(45)

# Factores de vista
F_cielo = (1 + np.cos(theta)) / 2
F_suelo_sup = 1 - F_cielo
F_suelo_inf = 1.0


# =============================================
#     PROPIEDADES ÓPTICAS (Tabla del informe)
# =============================================
# Vidrio
rho_vidrio = 0.05
alpha_vidrio = 0.01
tau_vidrio = 1 - rho_vidrio - alpha_vidrio

# Encapsulante
rho_enc = 0.01
tau_enc = 0.98
alpha_enc = 1 - rho_enc - tau_enc

# Celda
rho_celda = 0.05
tau_celda = 0.01
alpha_celda = 1 - rho_celda - tau_celda

# Irradiancia
G_solar = 800.0

# Cálculos de absorción por capa
flujo_q_vidrio = G_solar * alpha_vidrio
flujo_q_enc    = G_solar * tau_vidrio * alpha_enc
flujo_q_celda  = G_solar * tau_vidrio * tau_enc * alpha_celda


# =============================================
#     PROPIEDADES TÉRMICAS DE MATERIALES
# =============================================
k_vidrio = 1.0
k_enc = 0.4
k_celda = 148.0


# =============================================
# PROP AIRE — coherente con el FVM FUNCIONAL
# =============================================
def props_aire(T_K):
    T_C = T_K - 273.15
    rho = 352.977 / T_K
    cp  = 1003.7 - 0.032*T_C + 0.00035*T_C**2
    k   = 0.0241 + 0.000076*T_C
    mu  = (1.74 + 0.0049*T_C - 3.5e-6*T_C**2)*1e-5
    nu  = mu / rho
    alpha = k/(rho*cp)
    Pr = nu/alpha
    beta = 1/T_K
    return rho, mu, k, cp, Pr, beta, nu, alpha


def h_panel(Tsup, Tinf):

    # Propiedades del aire para Tsup
    rhoS, muS, kS, cpS, PrS, betaS, nuS, alphaS = props_aire(Tsup)
    # Para Tinf
    rhoI, muI, kI, cpI, PrI, betaI, nuI, alphaI = props_aire(Tinf)

    # ===== SUPERIOR =====
    g_eff = g*np.cos(theta)

    Ra_sup = g_eff * betaS * abs(Tsup - T_amb) * Lc_AP**3 / (nuS*alphaS)

    if 1e4 <= Ra_sup <= 1e7:
        Nu_nat_sup = 0.54 * Ra_sup**0.25
    elif 1e7 < Ra_sup <= 1e11:
        Nu_nat_sup = 0.15 * Ra_sup**(1/3)
    else:
        Nu_nat_sup = 0.0

    h_nat_sup = Nu_nat_sup * kS / Lc_AP

    # Forzada
    Re_sup = rhoS * V_viento * Lc_L / muS
    if 1e3 < Re_sup < 5e5:
        Nu_forz_sup = 0.664 * Re_sup**0.5 * PrS**(1/3)
    elif 5e5 <= Re_sup < 1e8:
        Nu_forz_sup = (0.037 * Re_sup**0.8 - 871)*PrS**(1/3)
    else:
        Nu_forz_sup = 0

    h_forz_sup = Nu_forz_sup * kS / Lc_L

    n = 3 + np.cos(theta)
    hsup = (h_nat_sup**n + h_forz_sup**n)**(1/n)


    # ===== INFERIOR (Churchill–Chu) =====
    Ra_inf = g_eff * betaI * abs(Tinf - T_amb) * Lc_AP**3 / (nuI*alphaI)

    term1 = 0.825
    term2 = 0.387 * Ra_inf**(1/6)
    term3 = (1 + (0.492/PrI)**(9/16))**(8/27)
    Nu_inf = (term1 + term2/term3)**2

    hinf = Nu_inf * kI / Lc_AP

    return hsup, hinf
