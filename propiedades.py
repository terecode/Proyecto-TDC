import numpy as np


# Geometría
Ancho_celda = 0.090
Ancho_borde = 0.001
Ancho_total = Ancho_celda + 2*Ancho_borde
esp_vidrio_sup = 0.002
esp_enc_sup = 0.001
esp_celda = 0.0003
esp_enc_inf    = 0.001
esp_vidrio_inf = 0.002
L_profundidad = 1.0 # m (Estándar 2D)

# Dimensiones reales
L_vertical = 2.416
L_horizontal = 1.09

# Propiedades Ópticas (Tabla 2)
# Vidrio 
rho_vidrio = 0.05                            # Reflectividad
alpha_vidrio = 0.01                          # Absortividad
tau_vidrio = 1.0 - rho_vidrio - alpha_vidrio # Transmisividad (Conservación de energía)

# Encapsulante 
rho_enc = 0.01                               # Reflectividad
tau_enc = 0.98                               # Transmisividad
alpha_enc = 1.0 - rho_enc - tau_enc          # Absortividad (implícita)

# Celda 
rho_celda = 0.05                             # Reflectividad
tau_celda = 0.01                             # Transmisividad
alpha_celda = 1.0 - rho_celda - tau_celda    # Absortividad (implícita)


# Flujos absorbidos (ópticos) reducidos por la sombra
G_solar_b = 800.0 
G_solar_c = 100.0 

q_abs_vidrio_b = G_solar_b * alpha_vidrio
q_abs_vidrio_c = G_solar_c * alpha_vidrio
q_abs_enc    = G_solar_b * tau_vidrio * alpha_enc
q_abs_enc    = G_solar_c * tau_vidrio * alpha_enc

# Materiales y Ambiente
k_vidrio, k_enc, k_celda = 1.0, 0.4, 148.0
T_amb = 20.0 + 273.15
T_cielo = 230.0
T_suelo = 20.0 + 273.15
sigma, emisividad = 5.67e-8, 0.85
Angulo = 45 * np.pi / 180
V_viento = 1.0
F_cielo = (1 + np.cos(Angulo)) / 2
F_suelo_sup = 1 - F_cielo
F_suelo_inf = 1.0


# DATOS DEL INCISO B
I_circuito = 10.324  # A 
R_celda = 0.215      # Ohm 

# Calor por Efecto Joule
# El enunciado indica usar Q = I^2 * R como calor total generado 
Q_joule_total = (I_circuito**2) * R_celda 
print(f"--- INCISO C ---")
print(f"Corriente forzada: {I_circuito} A")
print(f"Potencia Disipada (I^2*R): {Q_joule_total:.4f} W")


def h_panel_solar_inclinado(Ts_sup, Ts_inf, Tinf, L, g, rho, mu, k, cp, U_inf, beta=None, theta=np.radians(45)):

    def propiedades(Ts):
        """ Devuelve propiedades evaluadas a la temperatura de película correspondiente. """
        Tfilm = 0.5 * (Ts + Tinf)
        beta_local = (1/Tfilm) if beta is None else beta
        nu = mu / rho
        alpha = k/(rho*cp)
        Pr = nu/alpha
        return Tfilm, beta_local, nu, alpha, Pr

    # ---------- SUPERFICIE SUPERIOR ----------
    Tfilm_sup, beta_sup, nu_sup, alpha_sup, Pr_sup = propiedades(Ts_sup)

    g_eff = g * np.cos(theta)

    # Rayleigh para superficie superior
    Ra_sup = g_eff * beta_sup * abs(Ts_sup - Tinf) * L**3 / (nu_sup * alpha_sup)

    # NATURAL – placa horizontal caliente arriba
    if 1e4 <= Ra_sup <= 1e7:
        Nu_nat_sup = 0.54 * Ra_sup**0.25
    elif 1e7 < Ra_sup <= 1e11:
        Nu_nat_sup = 0.15 * Ra_sup**(1/3)
    else:
        Nu_nat_sup = np.nan

    h_nat_sup = Nu_nat_sup * k / L

    # FORZADA
    Re_sup = rho * U_inf * L / mu
    if 1e3 < Re_sup < 5e5:
        Nu_forz_sup = 0.664 * Re_sup**0.5 * Pr_sup**(1/3)
    elif 5e5 <= Re_sup < 1e8:
        Nu_forz_sup = (0.037 * Re_sup**0.8 - 871) * Pr_sup**(1/3)
    else:
        Nu_forz_sup = 0

    h_forz_sup = Nu_forz_sup * k / L

    # COMBINACIÓN
    n = 3 + np.cos(theta)
    h_sup = (h_nat_sup**n + h_forz_sup**n)**(1/n)

    # ---------- SUPERFICIE INFERIOR ----------
    Tfilm_inf, beta_inf, nu_inf, alpha_inf, Pr_inf = propiedades(Ts_inf)

    Ra_inf = g_eff * beta_inf * abs(Ts_inf - Tinf) * L**3 / (nu_inf * alpha_inf)

    # Churchill–Chu
    term1 = 0.825
    term2 = 0.387 * Ra_inf**(1/6)
    term3 = (1 + (0.492/Pr_inf)**(9/16))**(8/27)
    Nu_inf = (term1 + term2/term3)**2

    h_inf = Nu_inf * k / L

    # Información útil
    info = {"superior": {"Ra": Ra_sup, "Re": Re_sup, "Pr": Pr_sup, "Nu_nat": Nu_nat_sup, "Nu_forz": Nu_forz_sup,
        }, "inferior": {"Ra": Ra_inf, "Pr": Pr_inf, "Nu": Nu_inf,}}

    return h_sup, h_inf, info
