import numpy as np
import matplotlib.pyplot as plt


G_solar = 800.0

# Vidrio 
rho_vidrio = 0.05
alpha_vidrio = 0.01
tau_vidrio = 1.0 - rho_vidrio - alpha_vidrio

# Encapsulante 
rho_enc = 0.01
tau_enc = 0.98
alpha_enc = 1.0 - rho_enc - tau_enc

# Celda 
rho_celda = 0.05
tau_celda = 0.01
alpha_celda = 1.0 - rho_celda - tau_celda

# Cálculo de flujos de calor absorbido (q'')
q_abs_vidrio = G_solar * alpha_vidrio
G_transmitida_vidrio = G_solar * tau_vidrio

q_abs_enc = G_transmitida_vidrio * alpha_enc
G_transmitida_enc = G_transmitida_vidrio * tau_enc

q_abs_celda = G_transmitida_enc * alpha_celda

# Dimensiones
esp_vidrio_sup = 0.0021  
esp_enc_sup    = 0.0003  
esp_celda      = 0.0003  
esp_enc_inf    = 0.0003  
esp_vidrio_inf = 0.0021  

Espesor_total = esp_vidrio_sup + esp_enc_sup + esp_celda + esp_enc_inf + esp_vidrio_inf

# Flujos de calor volumétrico
flujo_q_vidrio_dot = q_abs_vidrio / esp_vidrio_sup
flujo_q_enc_dot    = q_abs_enc    / esp_enc_sup
flujo_q_celda_dot  = q_abs_celda  / esp_celda

print(f"--- Balance calculado ---")
print(f"Calor abs. Vidrio volumétrico: {flujo_q_vidrio_dot:.2f} W/m3")
print(f"Calor abs. Encap volumétrico:  {flujo_q_enc_dot:.2f} W/m3")
print(f"Calor abs. Celda volumétrico:  {flujo_q_celda_dot:.2f} W/m3")
print(f"--------------------------------")

Ancho_celda = 0.090  
Ancho_borde = 0.001  
Ancho_total = Ancho_celda + 2*Ancho_borde 

# Materiales y ambiente
k_vidrio = 1.0 
k_enc    = 0.4
k_celda  = 148.0
T_amb    = 20.0 + 273.15
T_cielo  = 230.0
T_suelo  = 20.0 + 273.15
sigma    = 5.67e-8
emisividad = 0.85

# Parámetros para h
L_caracteristico = 2.416 
L_horiz = 1.09
Angulo = 45 * np.pi / 180
V_viento = 1.0 
F_cielo = (1 + np.cos(Angulo)) / 2
F_suelo_sup = 1 - F_cielo
F_suelo_inf = 1.0

# Generacion de malla

nx_borde, nx_celda = 3, 30
nx = 2*nx_borde + nx_celda 

caras_x = np.concatenate([
    np.linspace(0, Ancho_borde, nx_borde+1)[:-1],
    np.linspace(Ancho_borde, Ancho_borde+Ancho_celda, nx_celda+1)[:-1],
    np.linspace(Ancho_borde+Ancho_celda, Ancho_total, nx_borde+1)
])
dx = np.diff(caras_x)
X_nodo = (caras_x[:-1] + caras_x[1:]) / 2 

dz_const = 300e-6

n_vidrio = int(round(esp_vidrio_sup / dz_const))
n_enc    = int(round(esp_enc_sup    / dz_const))  
n_celda  = int(round(esp_celda      / dz_const))  

nz = 2*n_vidrio + 2*n_enc + n_celda 

assert abs(nz*dz_const - Espesor_total) < 1e-9, "Chequea espesores y dz_const"

nodos_z = (np.arange(nz) + 0.5) * dz_const
dz = np.full(nz, dz_const)

idx_vid_sup_end = n_vidrio                   
idx_enc_sup_end = idx_vid_sup_end + n_enc     
idx_celda_end   = idx_enc_sup_end + n_celda   
idx_enc_inf_end = idx_celda_end + n_enc       

# Mapas [j,i]
Mapa_K = np.zeros((nz, nx))
Mapa_Q_dot = np.zeros((nz, nx))

for j in range(nz):

    if j < idx_vid_sup_end:          
        k_val_base = k_vidrio
        q_dot_base = flujo_q_vidrio_dot

    elif j < idx_enc_sup_end:        
        k_val_base = k_enc
        q_dot_base = flujo_q_enc_dot

    elif j < idx_celda_end:         

        k_val_base = 0.0  
        q_dot_base = 0.0

    elif j < idx_enc_inf_end:       
        k_val_base = k_enc
        q_dot_base = 0.0

    else:                           
        k_val_base = k_vidrio
        q_dot_base = 0.0

    for i in range(nx):
        x_c = X_nodo[i]

        if (j >= idx_enc_sup_end) and (j < idx_celda_end):
            if (x_c > Ancho_borde) and (x_c < Ancho_borde + Ancho_celda):
                Mapa_K[j, i] = k_celda
                Mapa_Q_dot[j, i] = flujo_q_celda_dot
            else:
                Mapa_K[j, i] = k_enc
                Mapa_Q_dot[j, i] = 0.0
        else:
            Mapa_K[j, i] = k_val_base
            Mapa_Q_dot[j, i] = q_dot_base

# Funciones
def props_aire(T_celsius):
    T = T_celsius
    tk = T + 273.15
    rho = 352.977/tk
    cp = 1003.7 - 0.032*T + 0.00035*T**2
    k = 0.0241 + 0.000076*T
    mu = (1.74 + 0.0049*T - 3.5e-6*T**2)*1e-5
    nu = mu/rho
    pr = nu/(k/(rho*cp))
    beta = 1/tk
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

# Iteraciones

T = np.ones((nz, nx)) * (40.0 + 273.15) 
error = 1.0
tol = 1e-5
max_iter = 20000
cnt = 0

print("Iniciando Método Iterativo...")

while error > tol and cnt < max_iter:
    cnt += 1
    T_old_global = T.copy()
    
    T_sup_avg = np.mean(T[0, :])
    T_inf_avg = np.mean(T[-1, :])
    h_top, h_bot = calc_h(T_sup_avg, T_inf_avg)
    
    for j in range(nz):
        dzj = dz[j]
        for i in range(nx):
            dxi = dx[i]
            k_P = Mapa_K[j, i]
            
            sigma_a_nb = 0.0
            rhs = Mapa_Q_dot[j, i] * dxi * dzj

            # Oeste
            if i > 0:
                k_W = Mapa_K[j, i-1]
                dist_x = (dxi + dx[i-1]) / 2.0
                k_interfaz_W = 2.0 * k_P * k_W / (k_P + k_W) if (k_P + k_W) != 0 else k_P
                a_W = k_interfaz_W * dzj / dist_x
                sigma_a_nb += a_W
                rhs += a_W * T[j, i-1]
            
            # Este
            if i < nx - 1:
                k_E = Mapa_K[j, i+1]
                dist_x = (dxi + dx[i+1]) / 2.0
                k_interfaz_E = 2.0 * k_P * k_E / (k_P + k_E) if (k_P + k_E) != 0 else k_P
                a_E = k_interfaz_E * dzj / dist_x
                sigma_a_nb += a_E
                rhs += a_E * T[j, i+1] 

            # Norte (j-1)
            if j > 0 and j < nz - 1:
                k_N = Mapa_K[j-1, i]
                dist_z = (dzj + dz[j-1]) / 2.0
                k_interfaz_N = 2.0 * k_P * k_N / (k_P + k_N) if (k_P + k_N) != 0 else k_P
                a_N = k_interfaz_N * dxi / dist_z
                sigma_a_nb += a_N
                rhs += a_N * T[j-1, i]
            
            # Borde superior (j = 0)
            if j == 0:
                T_val = T[j, i]
                dz_sup = dzj / 2.0
                
                # Conducción al nodo inferior (j=1)
                k_S = Mapa_K[j+1, i]
                dist_z_S = (dzj + dz[j+1]) / 2.0
                k_interfaz_S = 2.0 * k_P * k_S / (k_P + k_S) if (k_P + k_S) != 0 else k_P
                a_S = k_interfaz_S * dxi / dist_z_S
                sigma_a_nb += a_S
                rhs += a_S * T[j+1, i]
                
                # Conducción hasta la superficie
                a_cond_surf = k_P * dxi / dz_sup
                sigma_a_nb += a_cond_surf
                
                # Convección + radiación
                h_rad = emisividad * sigma * (
                    (T_val**2 + T_cielo**2) * (T_val + T_cielo) * F_cielo +
                    (T_val**2 + T_suelo**2) * (T_val + T_suelo) * F_suelo_sup
                )
                h_eff = h_top + h_rad
                a_conv_rad = h_eff * dxi
                
                rhs += a_cond_surf * T[j, i] 
                rhs += a_conv_rad * T_amb * (h_top / h_eff)
                rhs += a_conv_rad * (h_rad / h_eff) * (F_cielo*T_cielo + F_suelo_sup*T_suelo)

                sigma_a_nb += a_conv_rad

            # Nodos internos hacia abajo 
            if j < nz - 1 and j > 0:
                k_S = Mapa_K[j+1, i]
                dist_z = (dzj + dz[j+1]) / 2.0
                k_interfaz_S = 2.0 * k_P * k_S / (k_P + k_S) if (k_P + k_S) != 0 else k_P
                a_S = k_interfaz_S * dxi / dist_z
                sigma_a_nb += a_S
                rhs += a_S * T[j+1, i]

            # Borde inferior (j = nz-1)
            if j == nz - 1:
                T_val = T[j, i]
                dz_inf = dzj / 2.0
                
                # Conducción al nodo superior (j = nz-2)
                k_N = Mapa_K[j-1, i]
                dist_z_N = (dzj + dz[j-1]) / 2.0
                k_interfaz_N = 2.0 * k_P * k_N / (k_P + k_N) if (k_P + k_N) != 0 else k_P
                a_N = k_interfaz_N * dxi / dist_z_N
                sigma_a_nb += a_N
                rhs += a_N * T[j-1, i]
                
                # Conducción a la superficie inferior
                a_cond_surf = k_P * dxi / dz_inf
                sigma_a_nb += a_cond_surf
                
                # Convección + radiación
                h_rad = emisividad * sigma * (T_val**2 + T_suelo**2) * (T_val + T_suelo) * F_suelo_inf
                h_eff = h_bot + h_rad
                a_conv_rad = h_eff * dxi
                
                rhs += a_cond_surf * T[j, i]
                rhs += a_conv_rad * T_amb * (h_bot / h_eff)
                rhs += a_conv_rad * (h_rad / h_eff) * T_suelo
                
                sigma_a_nb += a_conv_rad

            T[j, i] = rhs / sigma_a_nb
    
    error = np.max(np.abs(T - T_old_global))
    if cnt % 100 == 0 or cnt == 1:
        print(f"Iter {cnt}: Error {error:.2e}, T_max {np.max(T)-273.15:.2f} °C")
    
    if np.max(T) > 500:
        print("¡ADVERTENCIA! Divergencia detectada (T > 500 K).")
        break

# Resultados
T_celsius = T - 273.15
print(f"\nConvergencia alcanzada en {cnt} iteraciones.")
print(f"Temp Máxima: {np.max(T_celsius):.2f} °C")
print(f"Temp Promedio Superficie Sup: {np.mean(T_celsius[0,:]):.2f} °C")

# Mapa de calor 
plt.figure(figsize=(10, 4))
plt.imshow(
    T_celsius, aspect='auto', cmap='inferno',
    extent=[0, Ancho_total*1000, Espesor_total*1000, 0]
)
plt.colorbar(label='Temperatura (°C)')
plt.title('Distribución de Temperatura 2D (Gauss-Seidel)')
plt.xlabel('Ancho (mm)')
plt.ylabel('Profundidad Z (mm)')
plt.show()

#Grafico
indice_centro = nx // 2
profundidades = nodos_z * 1000 
temps_centro = T_celsius[:, indice_centro]

plt.figure()
plt.plot(profundidades, temps_centro, 'o-', label='Centro Celda')
plt.xlabel('Profundidad (mm)')
plt.ylabel('Temperatura (°C)')
plt.title('Perfil de Temperatura a través del espesor')
plt.grid(True)
plt.legend()
plt.show()