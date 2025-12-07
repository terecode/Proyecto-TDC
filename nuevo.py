import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS Y CONSTANTES ---

# Radiación SOMBREADA sobre la celda
G_solar = 100.0  # W/m2

# Corriente fijada igual al inciso b
I_b = 10.30      # A  <-- pon aquí tu valor final del inciso b
R_celda = 0.215  # Ohm

# Propiedades ópticas (vidrio y encapsulante)
rho_vidrio = 0.05
alpha_vidrio = 0.01
tau_vidrio = 1.0 - rho_vidrio - alpha_vidrio

rho_enc = 0.01
tau_enc = 0.98
alpha_enc = 1.0 - rho_enc - tau_enc

# Calor óptico absorbido (G = 100 W/m2)
q_abs_vidrio = G_solar * alpha_vidrio
q_abs_enc    = G_solar * tau_vidrio * alpha_enc

# Geometría en Z (mm -> m)
esp_vidrio_sup = 0.0021   # 2.1 mm
esp_enc_sup    = 0.0003   # 0.3 mm
esp_celda      = 0.0003   # 0.3 mm
esp_enc_inf    = 0.0003   # 0.3 mm
esp_vidrio_inf = 0.0021   # 2.1 mm
Espesor_total  = esp_vidrio_sup + esp_enc_sup + esp_celda + esp_enc_inf + esp_vidrio_inf

# Geometría en X
Ancho_celda = 0.090   # 90 mm
Ancho_borde = 0.001   # 1 mm
Ancho_total = Ancho_celda + 2*Ancho_borde

# Materiales y ambiente
k_vidrio = 1.0
k_enc    = 0.4
k_celda  = 148.0

T_amb   = 20.0 + 273.15
T_cielo = 230.0
T_suelo = 20.0 + 273.15
sigma   = 5.67e-8
emisividad = 0.85

L_caracteristico = 2.416
L_horiz = 1.09
Angulo  = 45.0 * np.pi / 180.0
V_viento = 1.0
F_cielo = (1.0 + np.cos(Angulo)) / 2.0
F_suelo_sup = 1.0 - F_cielo
F_suelo_inf = 1.0

# --- 2. MALLA UNIFORME (DIFERENCIAS FINITAS CLÁSICAS) ---

# Nodos en X y Z (puedes cambiar N_x si quieres más resolución)
Nx = 60
Nz = 17   # 5.1 mm / 0.3 mm = 17 nodos en Z

dx = Ancho_total / (Nx - 1)
dz = Espesor_total / (Nz - 1)

x = np.linspace(0.0, Ancho_total, Nx)
z = np.linspace(0.0, Espesor_total, Nz)

# --- 3. MAPAS DE k Y q_vol (óptico) ---

Mapa_K = np.zeros((Nz, Nx))
Mapa_Q_optico = np.zeros((Nz, Nx))

for j in range(Nz):
    zc = z[j]

    # Capa según z
    if zc <= esp_vidrio_sup:
        k_base = k_vidrio
        q_vol_base = q_abs_vidrio / esp_vidrio_sup
    elif zc <= esp_vidrio_sup + esp_enc_sup:
        k_base = k_enc
        q_vol_base = q_abs_enc / esp_enc_sup
    elif zc <= esp_vidrio_sup + esp_enc_sup + esp_celda:
        k_base = 0.0
        q_vol_base = 0.0
    elif zc <= esp_vidrio_sup + esp_enc_sup + esp_celda + esp_enc_inf:
        k_base = k_enc
        q_vol_base = 0.0
    else:
        k_base = k_vidrio
        q_vol_base = 0.0

    for i in range(Nx):
        xc = x[i]

        # Zona de celda activa
        if (esp_vidrio_sup + esp_enc_sup <= zc <= esp_vidrio_sup + esp_enc_sup + esp_celda and
            Ancho_borde <= xc <= Ancho_borde + Ancho_celda):

            Mapa_K[j, i] = k_celda
            Mapa_Q_optico[j, i] = 0.0   # en c no hay fuente óptica en la celda
        else:
            Mapa_K[j, i] = k_base
            Mapa_Q_optico[j, i] = q_vol_base

# --- 4. FUNCIONES AUXILIARES PARA AIRE Y h ---

def props_aire(T_celsius):
    T = T_celsius
    tk = T + 273.15
    rho = 352.977 / tk
    cp  = 1003.7 - 0.032*T + 0.00035*T**2
    k   = 0.0241 + 0.000076*T
    mu  = (1.74 + 0.0049*T - 3.5e-6*T**2)*1e-5
    nu  = mu / rho
    pr  = nu / (k/(rho*cp))
    beta = 1.0 / tk
    return k, nu, pr, beta

def calc_h_conv(T_top, T_bot):
    # Convección superior (forzada + natural)
    T_film_top = 0.5*(T_top + T_amb)
    ka, nu, pr, beta = props_aire(T_film_top - 273.15)

    Re_h = V_viento * L_horiz / nu
    Nu_f = 0.664 * Re_h**0.5 * pr**(1/3)
    h_f = Nu_f * ka / L_horiz

    ka_op, nu_op, pr_op, beta_op = props_aire(T_top - 273.15)
    Ra = 9.81 * beta_op * abs(T_top - T_amb) * L_caracteristico**3 / (
         nu_op * (ka_op / ((352.977/T_top)*1005.0)) )
    h_n_top = 0.14 * (Ra*np.cos(Angulo))**(1/3) * ka_op / L_caracteristico

    n_exp = 3.0 + np.cos(Angulo)
    h_top = (h_n_top**n_exp + h_f**n_exp)**(1.0/n_exp)

    # Convección inferior (natural)
    kb_op, nub_op, prb_op, betab_op = props_aire(T_bot - 273.15)
    Ra_b = 9.81 * betab_op * abs(T_bot - T_amb) * L_caracteristico**3 / (
           nub_op * (kb_op / ((352.977/T_bot)*1005.0)) )
    h_bot = 0.27 * (Ra_b*np.cos(Angulo))**(1/4) * kb_op / L_caracteristico

    return h_top, h_bot

# --- 5. SOLVER DIFERENCIAS FINITAS (5-PUNTOS) ---

T = np.ones((Nz, Nx)) * (40.0 + 273.15)

error = 1.0
tol = 1e-5
max_iter = 50000
omega = 1.2
cnt = 0

# Fuente Joule (I_b^2 R) en volumen de celda
Area_panel = 2.416 * 1.09   # m2 (solo para referencia; no lo uso directo)
Q_joule_total = (I_b**2) * R_celda   # W totales
volumen_celda = Ancho_celda * esp_celda * 1.0
q_vol_celda = Q_joule_total / volumen_celda

print("Resolviendo inciso c) con DIFERENCIAS FINITAS CLÁSICAS...")

dx2 = dx*dx
dz2 = dz*dz

while error > tol and cnt < max_iter:
    cnt += 1
    T_old = T.copy()

    # Temperaturas promedio de superficies para h_conv
    T_sup_avg = np.mean(T[0, :])
    T_inf_avg = np.mean(T[-1, :])
    h_conv_top, h_conv_bot = calc_h_conv(T_sup_avg, T_inf_avg)

    # Radiación linealizada (promedio, para simplificar)
    T_top_ref = T_sup_avg
    T_bot_ref = T_inf_avg

    h_rad_top = emisividad * sigma * (
        (T_top_ref**2 + T_cielo**2)*(T_top_ref + T_cielo)*F_cielo +
        (T_top_ref**2 + T_suelo**2)*(T_top_ref + T_suelo)*F_suelo_sup
    )
    h_rad_bot = emisividad * sigma * (
        (T_bot_ref**2 + T_suelo**2)*(T_bot_ref + T_suelo)*F_suelo_inf
    )

    h_eff_top = h_conv_top + h_rad_top
    h_eff_bot = h_conv_bot + h_rad_bot

    # Temperaturas "efectivas" para BC de Robin
    T_inf_eff_top = (h_conv_top*T_amb +
                     h_rad_top*(F_cielo*T_cielo + F_suelo_sup*T_suelo)) / h_eff_top
    T_inf_eff_bot = (h_conv_bot*T_amb +
                     h_rad_bot*T_suelo) / h_eff_bot

    for j in range(Nz):
        for i in range(Nx):
            kP = Mapa_K[j, i]
            if kP == 0:
                # por seguridad (no debería pasar en zonas físicas)
                T[j, i] = T_old[j, i]
                continue

            # Definir fuente volumétrica del nodo
            if (esp_vidrio_sup + esp_enc_sup <= z[j] <= esp_vidrio_sup + esp_enc_sup + esp_celda and
                Ancho_borde <= x[i] <= Ancho_borde + Ancho_celda):
                # celda activa -> Joule
                q_vol = q_vol_celda
            else:
                q_vol = Mapa_Q_optico[j, i]

            # --- NODOS INTERIORES (clásico 5 puntos) ---
            if 0 < j < Nz-1 and 0 < i < Nx-1:
                TE = T_old[j, i+1]
                TW = T[j, i-1]
                TN = T_old[j+1, i]
                TS = T[j-1, i]

                T_new = (
                    kP*(TE*dz2 + TW*dz2 + TN*dx2 + TS*dx2) - dx2*dz2*q_vol
                ) / (2.0*kP*(dx2 + dz2))

                T[j, i] = T[j, i] + omega*(T_new - T[j, i])
                continue

            # --- BORDES EN X: adiabáticos (dT/dx = 0) ---
            # Se implementa con nodo fantasma tal que T(-1)=T(1) o T(N)=T(N-2)
            if i == 0:
                TW = T_old[j, 1]   # T_{-1} = T_1 -> laplaciano en x usa 2*(T1 - T0)
                TE = T_old[j, 1]
            elif i == Nx-1:
                TE = T_old[j, Nx-2]
                TW = T[j, Nx-2]
            else:
                TE = T_old[j, i+1]
                TW = T[j, i-1]

            # --- Borde superior (j = 0) con Robin ---
            if j == 0:
                T1 = T_old[j+1, i]
                # Fórmula DF con BC Robin (derivada con nodo fantasma eliminada):
                # Resultado simbólico:
                #   -k[ d2x + d2z ] + q = 0
                #   d2z usando nodo fantasma que satisface -k dT/dz = h_eff (T - T_inf_eff)
                TN = T1
                TS = None  # no se usa directamente

                # d2T/dz2 en el nodo superior (ver derivación)
                # y sustitución en ecuación 2D -> T0 explícito:
                T0_new = (
                    2*kP*dx2*TN +
                    2*kP*dz2*(TE + TW) -
                    T_inf_eff_top*dx2*dz*h_eff_top -
                    2*dx2*dz2*q_vol
                ) / (-dx2*dz*h_eff_top + 2*dx2*kP + 4*dz2*kP)

                T[j, i] = T[j, i] + omega*(T0_new - T[j, i])
                continue

            # --- Borde inferior (j = Nz-1) con Robin ---
            if j == Nz-1:
                Tm1 = T[j-1, i]
                # Análogo al caso superior (mismo tipo de fórmula)
                # cambiando h_eff_bot y T_inf_eff_bot
                TN = None
                TS = Tm1

                # derivación análoga da:
                Tn_new = (
                    2*kP*dx2*TS +
                    2*kP*dz2*(TE + TW) -
                    T_inf_eff_bot*dx2*dz*h_eff_bot -
                    2*dx2*dz2*q_vol
                ) / (-dx2*dz*h_eff_bot + 2*dx2*kP + 4*dz2*kP)

                T[j, i] = T[j, i] + omega*(Tn_new - T[j, i])
                continue

    error = np.max(np.abs(T - T_old))
    if cnt % 2000 == 0 or cnt == 1:
        T_celda_avg = np.mean(T[(z >= esp_vidrio_sup + esp_enc_sup) &
                                (z <= esp_vidrio_sup + esp_enc_sup + esp_celda), :])
        print(f"Iter {cnt}: err={error:.2e}, T_celda≈{T_celda_avg-273.15:.2f} °C")

# --- 6. RESULTADOS Y GRÁFICOS ---

T_c = T - 273.15
print("\n--- RESULTADOS INCISO c (DF clásica) ---")
print(f"Iteraciones: {cnt}")
print(f"Temp máxima panel: {np.max(T_c):.2f} °C")
print(f"Temp mínima panel: {np.min(T_c):.2f} °C")
print(f"I fija: {I_b:.3f} A, Q_Joule = {Q_joule_total:.2f} W")

# Mapa de calor
plt.figure(figsize=(10,4))
plt.imshow(
    T_c,
    origin='upper',
    extent=[0, Ancho_total*1000, Espesor_total*1000, 0],
    aspect='auto',
    cmap='inferno'
)
plt.colorbar(label='Temperatura (°C)')
plt.xlabel('Ancho del panel (mm)')
plt.ylabel('Profundidad Z (mm)')
plt.title('Distribución de Temperatura 2D - Inciso c (DF clásica)')
plt.tight_layout()
plt.show()

# Perfil en el centro
i_centro = Nx // 2
plt.figure()
plt.plot(z*1000, T_c[:, i_centro], 'o-')
plt.xlabel('Profundidad Z (mm)')
plt.ylabel('Temperatura (°C)')
plt.title('Perfil de Temperatura en el centro - Inciso c (DF clásica)')
plt.grid(True)
plt.tight_layout()
plt.show()
