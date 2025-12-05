import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS Y CONSTANTES ---
# Condición: Sombreado (Baja Radiación)
G_solar = 100.0 

# DATOS DEL INCISO B (Tus resultados)
I_circuito = 10.324  # A 
R_celda = 0.215      # Ohm 

# Calor por Efecto Joule
# El enunciado indica usar Q = I^2 * R como calor total generado 
Q_joule_total = (I_circuito**2) * R_celda 
print(f"--- INCISO C ---")
print(f"Corriente forzada: {I_circuito} A")
print(f"Potencia Disipada (I^2*R): {Q_joule_total:.4f} W")

# Dimensiones características
L_vertical = 2.416
L_horizontal = 1.09

# Propiedades Ópticas (Para capas transparentes)
rho_vidrio, alpha_vidrio = 0.05, 0.01
tau_vidrio = 1.0 - rho_vidrio - alpha_vidrio
rho_enc, tau_enc = 0.01, 0.98
alpha_enc = 1.0 - rho_enc - tau_enc

# Flujos absorbidos (ópticos) reducidos por la sombra
q_abs_vidrio = G_solar * alpha_vidrio
q_abs_enc    = G_solar * tau_vidrio * alpha_enc

# Geometría
Ancho_celda = 0.090
Ancho_borde = 0.001
Ancho_total = Ancho_celda + 2*Ancho_borde
esp_vidrio_sup, esp_enc_sup = 0.002, 0.001
esp_celda = 0.0003
esp_enc_inf, esp_vidrio_inf = 0.001, 0.002
L_profundidad = 1.0 # m (Estándar 2D)

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

# --- 2. MALLADO (Ajustado a instrucción c) ---
nx_borde, nx_celda = 3, 30
nx = 2*nx_borde + nx_celda

# "considere un único elemento diferencial de espesor" en la celda
nz_vidrio, nz_enc = 4, 2
nz_celda = 1  # <--- CAMBIO IMPORTANTE
nz = 2*nz_vidrio + 2*nz_enc + nz_celda

caras_x = np.concatenate([
    np.linspace(0, Ancho_borde, nx_borde+1)[:-1],
    np.linspace(Ancho_borde, Ancho_borde+Ancho_celda, nx_celda+1)[:-1],
    np.linspace(Ancho_borde+Ancho_celda, Ancho_total, nx_borde+1)
])
z_lista = [0]
for espesor, n_nodos in zip([esp_vidrio_sup, esp_enc_sup, esp_celda, esp_enc_inf, esp_vidrio_inf], 
                            [nz_vidrio, nz_enc, nz_celda, nz_enc, nz_vidrio]):
    z_lista.extend(np.linspace(z_lista[-1], z_lista[-1]+espesor, n_nodos+1)[1:])
caras_z = np.array(z_lista)
dx, dz = np.diff(caras_x), np.diff(caras_z)
X_nodo, Z_nodo = (caras_x[:-1] + caras_x[1:]) / 2, (caras_z[:-1] + caras_z[1:]) / 2

# Mapas de Propiedades
Mapa_K = np.zeros((nz, nx))
Mapa_Q = np.zeros((nz, nx))

idx_z_celda_start = nz_vidrio + nz_enc
idx_z_celda_end = idx_z_celda_start + nz_celda

# Densidad de potencia Joule en el volumen de la celda
volumen_celda = Ancho_celda * L_profundidad * esp_celda
q_vol_joule = Q_joule_total / volumen_celda

for j in range(nz):
    # K y Q base (óptico) para capas transparentes
    if j < nz_vidrio: k_v, q_v = k_vidrio, q_abs_vidrio/esp_vidrio_sup
    elif j < nz_vidrio+nz_enc: k_v, q_v = k_enc, q_abs_enc/esp_enc_sup
    elif j < idx_z_celda_end: k_v, q_v = 0, 0
    elif j < idx_z_celda_end+nz_enc: k_v, q_v = k_enc, 0
    else: k_v, q_v = k_vidrio, 0
    
    for i in range(nx):
        if (j >= idx_z_celda_start) and (j < idx_z_celda_end):
            # ZONA CELDA
            if (X_nodo[i] > Ancho_borde) and (X_nodo[i] < Ancho_borde+Ancho_celda):
                Mapa_K[j,i] = k_celda
                Mapa_Q[j,i] = q_vol_joule # FUENTE DE CALOR PRINCIPAL
            else: 
                Mapa_K[j,i] = k_enc
                Mapa_Q[j,i] = 0
        else:
            Mapa_K[j,i] = k_v
            Mapa_Q[j,i] = q_v

# --- 3. FUNCIONES AUXILIARES ---
def props_aire(T_celsius):
    T = T_celsius; tk = T + 273.15
    rho = 352.977/tk
    cp = 1003.7 - 0.032*T + 0.00035*T**2
    k = 0.0241 + 0.000076*T
    mu = (1.74 + 0.0049*T - 3.5e-6*T**2)*1e-5
    nu = mu/rho; pr = nu/(k/(rho*cp)); beta = 1/tk
    return k, nu, pr, beta

def calc_h(T_top, T_bot):
    ka, nu, pr, beta = props_aire(T_top - 273.15)
    Re = V_viento * L_horizontal / nu 
    Nu_f = 0.664*Re**0.5*pr**(1/3) if Re<5e5 else (0.037*Re**0.8-871)*pr**(1/3)
    h_f = Nu_f * ka / L_horizontal
    Ra = 9.81*beta*abs(T_top-T_amb)*L_vertical**3/(nu*(ka/((352.977/T_top)*1005)))
    Nu_n = 0.14*(Ra*np.cos(Angulo))**(1/3) #
    h_n_top = Nu_n * ka / L_vertical
    h_top = (h_n_top**3.7 + h_f**3.7)**(1/3.7) #
    
    kb, nub, prb, betab = props_aire(T_bot - 273.15)
    Rab = 9.81*betab*abs(T_bot-T_amb)*L_vertical**3/(nub*(kb/((352.977/T_bot)*1005)))
    Nu_bot = 0.27*(Rab*np.cos(Angulo))**(1/4)
    h_bot = Nu_bot * kb / L_vertical
    return h_top, h_bot

# --- 4. SOLVER ITERATIVO (INCISO C) ---
# Empezamos desde 30°C para eficiencia numérica
T = np.ones((nz, nx)) * (40.0 + 273.15)
error = 1.0; tol = 1e-5; max_iter = 50000; cnt = 0

print("Calculando Inciso c) (Hot Spot)...")

while error > tol and cnt < max_iter:
    cnt += 1
    T_old = T.copy()
    h_top, h_bot = calc_h(np.mean(T[0,:]), np.mean(T[-1,:]))
    
    for j in range(nz):
        for i in range(nx):
            dV = dx[i]*dz[j]; Ax = dz[j]; Az = dx[i]
            k_P = Mapa_K[j,i]
            rhs = Mapa_Q[j,i] * dV
            sigma_a = 0
            
            # Vecinos
            if i>0: 
                dist=(dx[i]+dx[i-1])/2; a=(2*k_P*Mapa_K[j,i-1]/(k_P+Mapa_K[j,i-1]))*Ax/dist
                sigma_a+=a; rhs+=a*T[j,i-1]
            if i<nx-1:
                dist=(dx[i]+dx[i+1])/2; a=(2*k_P*Mapa_K[j,i+1]/(k_P+Mapa_K[j,i+1]))*Ax/dist
                sigma_a+=a; rhs+=a*T_old[j,i+1]
            if j>0:
                dist=(dz[j]+dz[j-1])/2; a=(2*k_P*Mapa_K[j-1,i]/(k_P+Mapa_K[j-1,i]))*Az/dist
                sigma_a+=a; rhs+=a*T[j-1,i]
            else: 
                T_v = T_old[j,i]
                h_r = emisividad*sigma*((T_v**2+T_cielo**2)*(T_v+T_cielo)*F_cielo + 
                                        (T_v**2+T_suelo**2)*(T_v+T_suelo)*F_suelo_sup)
                sigma_a += (h_top+h_r)*Az
                rhs += (h_top*T_amb + h_r*(F_cielo*T_cielo+F_suelo_sup*T_suelo))*Az
            
            if j<nz-1:
                dist=(dz[j]+dz[j+1])/2; a=(2*k_P*Mapa_K[j+1,i]/(k_P+Mapa_K[j+1,i]))*Az/dist
                sigma_a+=a; rhs+=a*T_old[j+1,i]
            else:
                T_v = T_old[j,i]
                h_r = emisividad*sigma*(T_v**2+T_suelo**2)*(T_v+T_suelo)*F_suelo_inf
                sigma_a += (h_bot+h_r)*Az
                rhs += (h_bot*T_amb + h_r*T_suelo)*Az
            
            T[j,i] = rhs / sigma_a

    error = np.max(np.abs(T - T_old))
    if cnt % 5000 == 0:
        print(f"Iter {cnt}: Err {error:.2e}, T_max {np.max(T)-273.15:.2f}C")

# --- RESULTADOS ---
T_c = T - 273.15
print(f"Converged in {cnt} iterations.")
print(f"--- RESULTADOS INCISO c) ---")
print(f"Temp Máxima: {np.max(T_c):.2f} °C")
print(f"Temp Superficie Sup: {np.mean(T_c[0,:]):.2f} °C")

# Mapa de Calor
plt.figure(figsize=(10, 4))
plt.imshow(T_c, aspect='auto', cmap='inferno', extent=[0, Ancho_total*1000, 0, 1]) 
plt.colorbar(label='Temperatura (°C)')
plt.title('Distribución de Temperatura (Inciso c - Hot Spot)')
plt.xlabel('Ancho (mm)'); plt.ylabel('Espesor (Normalizado)')
plt.show()

# Perfil
plt.figure()
mid = nx//2
depths = np.cumsum(dz) - dz/2
depths = np.insert(depths, 0, 0)
temps = np.insert(T_c[:, mid], 0, T_c[0, mid])
plt.plot(depths*1000, temps, 'o-', color='red', label='Inciso c (Hot Spot)')
plt.grid(True); plt.xlabel('Profundidad (mm)'); plt.ylabel('Temperatura (°C)')
plt.title('Perfil de Temperatura: Inciso c)')
plt.legend(); plt.show()