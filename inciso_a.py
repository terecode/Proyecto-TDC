import numpy as np
import matplotlib.pyplot as plt

#HOLA
# --- 1. PARÁMETROS Y CONSTANTES ---

# Balance Óptico Automático
G_solar = 800.0  # W/m2

# Propiedades Ópticas (Tabla 2)
# Vidrio 
rho_vidrio = 0.05   # Reflectividad
alpha_vidrio = 0.01 # Absortividad
tau_vidrio = 1.0 - rho_vidrio - alpha_vidrio # Transmisividad (Conservación de energía)

# Encapsulante 
rho_enc = 0.01      # Reflectividad
tau_enc = 0.98      # Transmisividad
alpha_enc = 1.0 - rho_enc - tau_enc # Absortividad (implícita)

# Celda 
rho_celda = 0.05    # Reflectividad
tau_celda = 0.01    # Transmisividad
alpha_celda = 1.0 - rho_celda - tau_celda # Absortividad (implícita)

# 1. Capa Vidrio Superior
q_abs_vidrio = G_solar * alpha_vidrio
G_transmitida_vidrio = G_solar * tau_vidrio

# 2. Capa Encapsulante Superior
# La luz que llega es la que pasó el vidrio
q_abs_enc = G_transmitida_vidrio * alpha_enc
G_transmitida_enc = G_transmitida_vidrio * tau_enc

# 3. Capa Celda
# La luz que llega es la que pasó el encapsulante
q_abs_celda = G_transmitida_enc * alpha_celda

# Asignación a las variables del modelo
flujo_q_vidrio = q_abs_vidrio
flujo_q_enc    = q_abs_enc
flujo_q_celda  = q_abs_celda

# (Opcional) Imprimir para verificar
print(f"--- Balance Óptico Calculado ---")
print(f"Irradiancia Inicial: {G_solar} W/m2")
print(f"Calor abs. Vidrio:   {flujo_q_vidrio:.2f} W/m2")
print(f"Calor abs. Encap:    {flujo_q_enc:.2f} W/m2")
print(f"Calor abs. Celda:    {flujo_q_celda:.2f} W/m2")
print(f"--------------------------------")

# Geometría
Ancho_celda = 0.090
Ancho_borde = 0.001
Ancho_total = Ancho_celda + 2*Ancho_borde
esp_vidrio_sup = 0.002
esp_enc_sup    = 0.001
esp_celda      = 0.0003
esp_enc_inf    = 0.001
esp_vidrio_inf = 0.002

# Materiales y Ambiente
k_vidrio = 1.0
k_enc    = 0.4
k_celda  = 148.0
T_amb    = 20.0 + 273.15
T_cielo  = 230.0
T_suelo  = 20.0 + 273.15
sigma    = 5.67e-8
emisividad = 0.85

L_caracteristico = 2.416
Angulo = 45 * np.pi / 180
V_viento = 1.0
F_cielo = (1 + np.cos(Angulo)) / 2
F_suelo_sup = 1 - F_cielo
F_suelo_inf = 1.0

# --- 2. GENERACIÓN DE MALLA (GRID) ---
nx_borde, nx_celda = 3, 30
nx = 2*nx_borde + nx_celda
nz_vidrio, nz_enc, nz_celda = 4, 2, 2
nz = 2*nz_vidrio + 2*nz_enc + nz_celda

# Coordenadas X
caras_x = np.concatenate([
    np.linspace(0, Ancho_borde, nx_borde+1)[:-1],
    np.linspace(Ancho_borde, Ancho_borde+Ancho_celda, nx_celda+1)[:-1],
    np.linspace(Ancho_borde+Ancho_celda, Ancho_total, nx_borde+1)
])

# Coordenadas Z
z_lista = [0]
for espesor, n_nodos in zip([esp_vidrio_sup, esp_enc_sup, esp_celda, esp_enc_inf, esp_vidrio_inf], 
                            [nz_vidrio, nz_enc, nz_celda, nz_enc, nz_vidrio]):
    z_lista.extend(np.linspace(z_lista[-1], z_lista[-1]+espesor, n_nodos+1)[1:])
caras_z = np.array(z_lista)

dx = np.diff(caras_x)
dz = np.diff(caras_z)
X_nodo = (caras_x[:-1] + caras_x[1:]) / 2
Z_nodo = (caras_z[:-1] + caras_z[1:]) / 2

# Mapas de Propiedades
Mapa_K = np.zeros((nz, nx))
Mapa_Q = np.zeros((nz, nx))

# Llenar Mapas
idx_z_celda_start = nz_vidrio + nz_enc
idx_z_celda_end = idx_z_celda_start + nz_celda

for j in range(nz):
    z_c = Z_nodo[j]
    # Determinar capa base
    if j < nz_vidrio: k_val, q_val = k_vidrio, flujo_q_vidrio/esp_vidrio_sup
    elif j < nz_vidrio+nz_enc: k_val, q_val = k_enc, flujo_q_enc/esp_enc_sup
    elif j < idx_z_celda_end: k_val, q_val = 0, 0 # Se define en bucle x
    elif j < idx_z_celda_end+nz_enc: k_val, q_val = k_enc, 0
    else: k_val, q_val = k_vidrio, 0
    
    for i in range(nx):
        x_c = X_nodo[i]
        if (j >= idx_z_celda_start) and (j < idx_z_celda_end):
            if (x_c > Ancho_borde) and (x_c < Ancho_borde+Ancho_celda):
                Mapa_K[j,i] = k_celda
                Mapa_Q[j,i] = flujo_q_celda/esp_celda
            else:
                Mapa_K[j,i] = k_enc
                Mapa_Q[j,i] = 0
        else:
            Mapa_K[j,i] = k_val
            Mapa_Q[j,i] = q_val

# --- 3. FUNCIONES AUXILIARES ---
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
    # Top
    ka, nu, pr, beta = props_aire(T_top - 273.15)
    Re = V_viento*L_caracteristico/nu
    Nu_f = 0.664*Re**0.5*pr**(1/3) if Re<5e5 else (0.037*Re**0.8-871)*pr**(1/3)
    h_f = Nu_f*ka/L_caracteristico
    Ra = 9.81*beta*abs(T_top-T_amb)*L_caracteristico**3/(nu*(ka/((352.977/T_top)*1005)))
    h_n_top = 0.14*(Ra*np.cos(Angulo))**(1/3)*ka/L_caracteristico
    h_top = (h_n_top**3.7 + h_f**3.7)**(1/3.7)
    
    # Bot
    kb, nub, prb, betab = props_aire(T_bot - 273.15)
    Rab = 9.81*betab*abs(T_bot-T_amb)*L_caracteristico**3/(nub*(kb/((352.977/T_bot)*1005)))
    h_bot = 0.27*(Rab*np.cos(Angulo))**(1/4)*kb/L_caracteristico
    return h_top, h_bot

# --- 4. SOLVER ITERATIVO ---
T = np.ones((nz, nx)) * (40.0 + 273.15) 

error = 1.0
tol = 1e-5
max_iter = 50000
cnt = 0

print("Iniciando Método Iterativo...")

while error > tol and cnt < max_iter:
    cnt += 1
    T_old_global = T.copy()
    
    # Calcular h con temperaturas promedio actuales
    T_sup_avg = np.mean(T[0, :])
    T_inf_avg = np.mean(T[-1, :])
    h_top, h_bot = calc_h(T_sup_avg, T_inf_avg)
    
    # Bucle espacial (Gauss-Seidel)
    for j in range(nz):
        for i in range(nx):
            dV = dx[i] * dz[j]
            Ax = dz[j] # Área cara lateral
            Az = dx[i] # Área cara superior/inferior
            k_P = Mapa_K[j,i]
            
            # Acumuladores: a_P * T_P = Sum(a_nb * T_nb) + RHS
            sigma_a_nb = 0 # Suma de coeficientes a_nb (que formarán parte de a_P)
            rhs = Mapa_Q[j,i] * dV # Término fuente inicial
            
            # --- VECINOS ---
            # Oeste
            if i > 0:
                k_W = Mapa_K[j, i-1]
                dist = (dx[i] + dx[i-1])/2
                a_W = (2*k_P*k_W/(k_P+k_W)) * Ax / dist
                sigma_a_nb += a_W
                rhs += a_W * T[j, i-1] # Usamos valor actualizado (Gauss-Seidel)

            # Este
            if i < nx - 1:
                k_E = Mapa_K[j, i+1]
                dist = (dx[i] + dx[i+1])/2
                a_E = (2*k_P*k_E/(k_P+k_E)) * Ax / dist
                sigma_a_nb += a_E
                rhs += a_E * T_old_global[j, i+1] # Valor iteración anterior

            # Norte (Arriba)
            if j > 0:
                k_N = Mapa_K[j-1, i]
                dist = (dz[j] + dz[j-1])/2
                a_N = (2*k_P*k_N/(k_P+k_N)) * Az / dist
                sigma_a_nb += a_N
                rhs += a_N * T[j-1, i]
            else:
                # Borde Superior (Convección + Rad)
                T_val = T_old_global[j, i]
                # Linearización de radiación
                h_rad = emisividad * sigma * ((T_val**2 + T_cielo**2)*(T_val + T_cielo)*F_cielo + 
                                              (T_val**2 + T_suelo**2)*(T_val + T_suelo)*F_suelo_sup)
                
                coef_total_sup = (h_top + h_rad) * Az
                sigma_a_nb += coef_total_sup
                
                T_alr_equiv = (h_top*T_amb + h_rad*(F_cielo*T_cielo + F_suelo_sup*T_suelo)) / (h_top + h_rad)
                # O mejor, sumar directamente al RHS
                rhs += (h_top*T_amb + h_rad*(F_cielo*T_cielo + F_suelo_sup*T_suelo)) * Az

            # Sur (Abajo)
            if j < nz - 1:
                k_S = Mapa_K[j+1, i]
                dist = (dz[j] + dz[j+1])/2
                a_S = (2*k_P*k_S/(k_P+k_S)) * Az / dist
                sigma_a_nb += a_S
                rhs += a_S * T_old_global[j+1, i]
            else:
                # Borde Inferior
                T_val = T_old_global[j, i]
                h_rad = emisividad * sigma * (T_val**2 + T_suelo**2)*(T_val + T_suelo) * F_suelo_inf
                coef_total_inf = (h_bot + h_rad) * Az
                sigma_a_nb += coef_total_inf
                rhs += (h_bot*T_amb + h_rad*T_suelo) * Az

            # Actualizar temperatura del nodo
            # T_P = RHS / a_P (donde a_P = sum(a_nb))
            T[j,i] = rhs / sigma_a_nb
            
    error = np.max(np.abs(T - T_old_global))
    if cnt % 100 == 0:
        print(f"Iter {cnt}: Error {error:.2e}, T_max {np.max(T)-273.15:.2f} C")

# --- 5. RESULTADOS Y GRÁFICOS ---
T_celsius = T - 273.15
print(f"Convergencia alcanzada en {cnt} iteraciones.")
print(f"Temp Máxima: {np.max(T_celsius):.2f} °C")
print(f"Temp Promedio Superficie Sup: {np.mean(T_celsius[0,:]):.2f} °C")

# Mapa de Calor
plt.figure(figsize=(10, 4))
# Escalar Z para visualización
plt.imshow(T_celsius, aspect='auto', cmap='inferno', 
           extent=[0, Ancho_total*1000, 0, 1]) 
plt.colorbar(label='Temperatura (°C)')
plt.title('Distribución de Temperatura 2D en el Panel (Inciso a9')
plt.xlabel('Ancho (mm)')
plt.ylabel('Espesor (Normalizado)')
plt.show()

# Perfil de Temperatura en el centro
indice_centro = nx // 2
profundidades = np.cumsum(dz) - dz/2
profundidades = np.insert(profundidades, 0, 0) # Agregar superficie
temps_centro = T_celsius[:, indice_centro]
temps_centro = np.insert(temps_centro, 0, T_celsius[0, indice_centro]) # Aprox superficie

plt.figure()
plt.plot(profundidades*1000, temps_centro, 'o-', label='Centro Celda')
plt.xlabel('Profundidad (mm)')
plt.ylabel('Temperatura (°C)')
plt.title('Perfil de Temperatura a través del espesor (Inciso a)')
plt.grid(True)
plt.legend()
plt.show()