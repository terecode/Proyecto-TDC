import numpy as np
import matplotlib.pyplot as plt

G_solar = 800.0  
R_celda = 0.215  

#Vidrio
rho_vidrio = 0.05
alpha_vidrio = 0.01
tau_vidrio = 1.0 - rho_vidrio - alpha_vidrio

#Encapsulante
rho_enc = 0.01
tau_enc = 0.98
alpha_enc = 1.0 - rho_enc - tau_enc

# Calor absorbido
q_abs_vidrio = G_solar * alpha_vidrio                              
q_abs_enc    = G_solar * tau_vidrio * alpha_enc                    

# Dimensiones
esp_vidrio_sup = 0.0021   
esp_enc_sup    = 0.0003   
esp_celda      = 0.0003   
esp_enc_inf    = 0.0003  
esp_vidrio_inf = 0.0021 

Espesor_total = esp_vidrio_sup + esp_enc_sup + esp_celda + esp_enc_inf + esp_vidrio_inf

Ancho_celda = 0.090   
Ancho_borde = 0.001  
Ancho_total = Ancho_celda + 2*Ancho_borde

L_profundidad = 1.0

# Materiales y ambiente
k_vidrio = 1.0
k_enc    = 0.4
k_celda  = 148.0

T_amb   = 20.0 + 273.15
T_cielo = 230.0
T_suelo = 20.0 + 273.15
sigma   = 5.67e-8
emisividad = 0.85

# Parámetros para h
L_caracteristico = 2.416
L_horiz = 1.09
Angulo = 45.0 * np.pi / 180.0
V_viento = 1.0
F_cielo = (1.0 + np.cos(Angulo)) / 2.0
F_suelo_sup = 1.0 - F_cielo
F_suelo_inf = 1.0

# Generación de malla

nx_borde, nx_celda = 3, 30
nx = 2*nx_borde + nx_celda

caras_x = np.concatenate([
    np.linspace(0.0, Ancho_borde, nx_borde+1)[:-1],
    np.linspace(Ancho_borde, Ancho_borde+Ancho_celda, nx_celda+1)[:-1],
    np.linspace(Ancho_borde+Ancho_celda, Ancho_total, nx_borde+1)
])
dx = np.diff(caras_x)
X_nodo = (caras_x[:-1] + caras_x[1:]) / 2.0

dz_const = 300e-6  

n_vidrio = int(round(esp_vidrio_sup / dz_const))  
n_enc    = int(round(esp_enc_sup    / dz_const))  
n_celda  = int(round(esp_celda      / dz_const))  

nz = 2*n_vidrio + 2*n_enc + n_celda 

assert abs(nz*dz_const - Espesor_total) < 1e-9, "Revisa espesores o dz_const, no cierran."

dz = np.full(nz, dz_const)
nodos_z = (np.arange(nz) + 0.5) * dz_const 

idx_vid_sup_end = n_vidrio
idx_enc_sup_end = idx_vid_sup_end + n_enc
idx_celda_end   = idx_enc_sup_end + n_celda
idx_enc_inf_end = idx_celda_end + n_enc

# Mapas [j,i]

Mapa_K = np.zeros((nz, nx))
Mapa_Q_optico = np.zeros((nz, nx)) 

for j in range(nz):

    if j < idx_vid_sup_end:
        k_base = k_vidrio
        q_vol_base = q_abs_vidrio / esp_vidrio_sup

    elif j < idx_enc_sup_end:
        k_base = k_enc
        q_vol_base = q_abs_enc / esp_enc_sup

    elif j < idx_celda_end:
        k_base = 0.0
        q_vol_base = 0.0

    elif j < idx_enc_inf_end:
        k_base = k_enc
        q_vol_base = 0.0

    else:
        k_base = k_vidrio
        q_vol_base = 0.0

    for i in range(nx):
        x_c = X_nodo[i]
        if idx_enc_sup_end <= j < idx_celda_end:
            if (x_c > Ancho_borde) and (x_c < Ancho_borde + Ancho_celda):
                Mapa_K[j, i] = k_celda
                Mapa_Q_optico[j, i] = 0.0
            else:
                Mapa_K[j, i] = k_enc
                Mapa_Q_optico[j, i] = 0.0
        else:
            Mapa_K[j, i] = k_base
            Mapa_Q_optico[j, i] = q_vol_base

# Funciones

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

def calc_h(T_top, T_bot):
    # Superficie superior combinada
    T_film_top = (T_top + T_amb) / 2
    ka, nu, pr, beta = props_aire(T_film_top - 273.15)
    
    # Convección forzada 
    Re_L = V_viento * L_caracteristico / nu
    if Re_L < 5e5:
        Nu_f = 0.664 * (Re_L**0.5) * (pr**(1/3))
    else:
        Nu_f = (0.037 * (Re_L**0.8) - 871) * (pr**(1/3))
        
    h_f = Nu_f * ka / L_caracteristico
    
    # Conveccion natural
    Area_panel = L_caracteristico * L_horiz
    Perimetro_panel = 2 * (L_caracteristico + L_horiz)
    L_char_nat = Area_panel / Perimetro_panel
    
    delta_T = abs(T_top - T_amb)
    Ra_nat = (9.81 * np.cos(Angulo) * beta * delta_T * (L_char_nat**3) * pr) / (nu**2)
    
    if Ra_nat < 1e7:
        Nu_n = 0.54 * (Ra_nat**(1/4))
    else:
        Nu_n = 0.15 * (Ra_nat**(1/3))
        
    h_n_top = Nu_n * ka / L_char_nat
    
    # Convección combinada
    n_exp = 3 + np.cos(Angulo)
    h_top = (h_n_top**n_exp + h_f**n_exp)**(1/n_exp)
    
    # Superficie inferior
    T_film_bot = (T_bot + T_amb) / 2
    kb, nub, prb, betab = props_aire(T_film_bot - 273.15)
    delta_T = abs(T_bot - T_amb)
    Ra_L = (9.81 * np.cos(Angulo) * betab * delta_T * L_caracteristico**3 * prb) / (nub**2)
    numerator = 0.387 * (Ra_L**(1/6))
    denominator = (1 + (0.492 / prb)**(9/16))**(8/27)
    
    Nu_churchill = (0.825 + numerator / denominator)**2
    h_bot = Nu_churchill * kb / L_caracteristico
    
    return h_top, h_bot

def calc_eff_I(T_celda_K, G=800.0):
    Tc = T_celda_K - 273.15
    eta = 0.2183 * (1.0 - (0.34/100.0)*(Tc - 25.0))
    I   = 12.87 * (1.0 + (0.04/100.0)*(Tc - 25.0)) * (G/1000.0)
    return eta, I

# Iteraciones

T = np.ones((nz, nx)) * (40.0 + 273.15)

error = 1.0
tol = 1e-5
max_iter = 50000
cnt = 0

eta_final = 0.0
I_final   = 0.0

print("Iniciando Método Iterativo...")

while error > tol and cnt < max_iter:
    cnt += 1
    T_old = T.copy()

    T_sup_avg = np.mean(T[0, :])
    T_inf_avg = np.mean(T[-1, :])
    h_top, h_bot = calc_h(T_sup_avg, T_inf_avg)

    T_celda_avg = np.mean(T[idx_enc_sup_end:idx_celda_end, nx_borde:-nx_borde])

    eta, I_celda = calc_eff_I(T_celda_avg, G_solar)
    eta_final, I_final = eta, I_celda

    Q_joule_total = (I_celda**2) * R_celda 
    volumen_celda = Ancho_celda * L_profundidad * esp_celda
    q_vol_celda = Q_joule_total / volumen_celda  

    for j in range(nz):
        dzj = dz[j]
        for i in range(nx):
            dxi = dx[i]
            dV  = dxi * dzj
            Ax  = dzj   
            Az  = dxi   
            k_P = Mapa_K[j, i]

            # Fuente del nodo
            if (idx_enc_sup_end <= j < idx_celda_end and
                nx_borde <= i < nx - nx_borde and
                Mapa_K[j, i] == k_celda):
                rhs = q_vol_celda * dV
            else:
                rhs = Mapa_Q_optico[j, i] * dV

            sigma_a_nb = 0.0

            # Oeste
            if i > 0:
                k_W = Mapa_K[j, i-1]
                dist_x = (dxi + dx[i-1]) / 2.0
                k_int_W = 2.0*k_P*k_W/(k_P + k_W) if (k_P + k_W) != 0 else k_P
                a_W = k_int_W * Ax / dist_x
                sigma_a_nb += a_W
                rhs += a_W * T[j, i-1]

            # Este
            if i < nx - 1:
                k_E = Mapa_K[j, i+1]
                dist_x = (dxi + dx[i+1]) / 2.0
                k_int_E = 2.0*k_P*k_E/(k_P + k_E) if (k_P + k_E) != 0 else k_P
                a_E = k_int_E * Ax / dist_x
                sigma_a_nb += a_E
                rhs += a_E * T_old[j, i+1]

            # Norte
            if j > 0:
                k_N = Mapa_K[j-1, i]
                dist_z = (dzj + dz[j-1]) / 2.0
                k_int_N = 2.0*k_P*k_N/(k_P + k_N) if (k_P + k_N) != 0 else k_P
                a_N = k_int_N * Az / dist_z
                sigma_a_nb += a_N
                rhs += a_N * T[j-1, i]
            else:
                # Borde superior: convección + radiación
                T_val = T_old[j, i]
                h_rad = emisividad * sigma * (
                    (T_val**2 + T_cielo**2)*(T_val + T_cielo)*F_cielo +
                    (T_val**2 + T_suelo**2)*(T_val + T_suelo)*F_suelo_sup
                )
                h_comb = h_top + h_rad
                coef_sup = h_comb * Az
                sigma_a_nb += coef_sup
                rhs += h_top*T_amb*Az
                rhs += h_rad*(F_cielo*T_cielo + F_suelo_sup*T_suelo) * Az

            # Sur
            if j < nz - 1:
                k_S = Mapa_K[j+1, i]
                dist_z = (dzj + dz[j+1]) / 2.0
                k_int_S = 2.0*k_P*k_S/(k_P + k_S) if (k_P + k_S) != 0 else k_P
                a_S = k_int_S * Az / dist_z
                sigma_a_nb += a_S
                rhs += a_S * T_old[j+1, i]
            else:
                # Borde inferior: convección + radiación
                T_val = T_old[j, i]
                h_rad = emisividad * sigma * (T_val**2 + T_suelo**2)*(T_val + T_suelo)*F_suelo_inf
                h_comb = h_bot + h_rad
                coef_inf = h_comb * Az
                sigma_a_nb += coef_inf
                rhs += h_bot*T_amb*Az
                rhs += h_rad*T_suelo*Az

            T[j, i]= rhs / sigma_a_nb

    error = np.max(np.abs(T - T_old))
    if cnt % 5000 == 0 or cnt == 1:
        print(f"Iter {cnt}: Err={error:.2e}, T_celda={T_celda_avg-273.15:.2f} °C, I={I_celda:.2f} A")

# Resultados

T_c = T - 273.15

print("\n--- RESULTADOS INCISO b ---")
print(f"Convergencia en {cnt} iteraciones.")
print(f"Temp máxima en todo el panel: {np.max(T_c):.2f} °C")
print(f"Temp máxima en la celda: {np.max(T_c[idx_enc_sup_end:idx_celda_end, nx_borde:-nx_borde]):.2f} °C")
print(f"Eficiencia: {eta_final*100.0:.2f} %")
print(f"Corriente: {I_final:.3f} A")
print(f"Calor Joule total (I^2 R): {I_final**2 * R_celda:.2f} W")

# Mapa de calor
plt.figure(figsize=(10, 4))
plt.imshow(
    T_c,
    aspect='auto',
    cmap='inferno',
    extent=[0, Ancho_total*1000, Espesor_total*1000, 0]
)
plt.colorbar(label='Temperatura (°C)')
plt.title('Distribución de Temperatura 2D')
plt.xlabel('Ancho del panel (mm)')
plt.ylabel('Profundidad Z (mm)')
plt.tight_layout()
plt.show()

#Grafico
indice_centro = nx // 2
profundidades_mm = nodos_z * 1000.0
temps_centro = T_c[:, indice_centro]

plt.figure()
plt.plot(profundidades_mm, temps_centro, 'o-', label='Centro celda')
plt.grid(True)
plt.xlabel('Profundidad (mm)')
plt.ylabel('Temperatura (°C)')
plt.title('Perfil de Temperatura a través del espesor')
plt.legend()
plt.tight_layout()
plt.show()