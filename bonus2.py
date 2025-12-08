import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS FIJOS ---
T_amb_C = 20.0
T_amb = T_amb_C + 273.15
T_cielo = 230.0
T_suelo = T_amb_C + 273.15

sigma = 5.67e-8
G_solar = 800.0 

# Dimensiones y Geometría
L_vidrio = 0.0021   
L_encap  = 0.0003  
L_celda  = 0.0003   
L_caracteristico = 2.416 
L_horiz = 1.09            

# Propiedades Materiales
k_vidrio = 1.0
k_encap = 0.4
k_celda = 148.0

eps_vidrio = 0.85

# Propiedades Ópticas (Absorción)
alpha_vidrio = 0.01
tau_vidrio = 1.0 - 0.05 - 0.01
alpha_enc = 1.0 - 0.01 - 0.98
alpha_celda = 1.0 - 0.05 - 0.01 

# Flujos de Absorción Óptica (Base)
q_abs_vidrio = G_solar * alpha_vidrio
G_trans_v = G_solar * tau_vidrio
q_abs_enc = G_trans_v * alpha_enc
G_trans_e = G_trans_v * 0.98
q_abs_celda_optico = G_trans_e * alpha_celda

# Propiedades Eléctricas (Tabla 2 y Enunciado)
R_elec = 0.215e-3 # 215 mOhm a Ohm -> REVISAR UNIDADES. Dice 215 mOhm? 
# En tabla 2: "Resistencia eléctrica 215 mΩ". -> 0.215 Ohm.
R_elec = 0.215 

# Ángulo
theta_rad = np.deg2rad(45)

# Factores de Vista
F_cielo = (1 + np.cos(theta_rad))/2
F_suelo_sup = 1 - F_cielo
F_suelo_inf = 1.0

# --- 2. MALLADO (Igual al anterior) ---
nx_borde, nx_celda = 3, 30
nx = 2*nx_borde + nx_celda 
Ancho_celda = 0.090
Ancho_borde = 0.001
Ancho_total = Ancho_celda + 2*Ancho_borde

caras_x = np.concatenate([
    np.linspace(0, Ancho_borde, nx_borde+1)[:-1],
    np.linspace(Ancho_borde, Ancho_borde+Ancho_celda, nx_celda+1)[:-1],
    np.linspace(Ancho_borde+Ancho_celda, Ancho_total, nx_borde+1)
])
dx = np.diff(caras_x)
X_nodo = (caras_x[:-1] + caras_x[1:]) / 2 

dz_const = 300e-6
n_vidrio = int(round(L_vidrio / dz_const))
n_enc    = 1
n_celda  = 1
nz = 2*n_vidrio + 2*n_enc + n_celda
dz = np.full(nz, dz_const)

# Índices Z
idx_v1_end = n_vidrio
idx_e1_end = idx_v1_end + n_enc
idx_c_end  = idx_e1_end + n_celda
idx_e2_end = idx_c_end + n_enc

# Mapa K base
Mapa_K = np.zeros((nz, nx))
for j in range(nz):
    if j < idx_v1_end: k_val = k_vidrio
    elif j < idx_e1_end: k_val = k_encap
    elif j < idx_c_end: k_val = 0.0 # Se asigna luego
    elif j < idx_e2_end: k_val = k_encap
    else: k_val = k_vidrio
    
    for i in range(nx):
        if (j >= idx_e1_end) and (j < idx_c_end):
            x_c = X_nodo[i]
            if (x_c > Ancho_borde) and (x_c < Ancho_borde+Ancho_celda):
                Mapa_K[j, i] = k_celda
            else:
                Mapa_K[j, i] = k_encap
        else:
            Mapa_K[j, i] = k_val

# --- 3. FUNCIONES AUXILIARES ---
def props_aire(T_c):
    Tk = T_c + 273.15
    rho = 352.977/Tk
    cp = 1003.7 - 0.032*T_c + 0.00035*T_c**2
    k = 0.0241 + 0.000076*T_c
    mu = (1.74 + 0.0049*T_c - 3.5e-6*T_c**2)*1e-5
    nu = mu/rho
    pr = nu/(k/(rho*cp))
    beta = 1/Tk
    return k, nu, pr, beta

def calc_h_coeffs(T_top, T_bot, V_wind):
    # Top Combinada
    T_film = (T_top + T_amb_C)/2
    ka, nu, pr, beta = props_aire(T_film)
    
    # Forzada
    Re = V_wind * L_caracteristico / nu
    if Re < 5e5: Nu_f = 0.664 * Re**0.5 * pr**(1/3)
    else:        Nu_f = (0.037 * Re**0.8 - 871) * pr**(1/3)
    h_f = Nu_f * ka / L_caracteristico
    
    # Natural Top
    L_nat = (L_caracteristico*L_horiz)/(2*(L_caracteristico+L_horiz))
    g_eff = 9.81 * np.cos(theta_rad)
    Ra = g_eff * beta * abs(T_top - T_amb_C) * L_nat**3 * pr / nu**2
    if Ra < 1e7: Nu_n = 0.54 * Ra**0.25
    else:        Nu_n = 0.15 * Ra**(1/3)
    h_n = Nu_n * ka / L_nat
    
    n_exp = 3 + np.cos(theta_rad)
    h_top = (h_n**n_exp + h_f**n_exp)**(1/n_exp)
    
    # Bot Churchill-Chu
    T_film_b = (T_bot + T_amb_C)/2
    kb, nub, prb, betab = props_aire(T_film_b)
    Ra_b = g_eff * betab * abs(T_bot - T_amb_C) * L_caracteristico**3 * prb / nub**2
    num = 0.387 * Ra_b**(1/6)
    den = (1 + (0.492/prb)**(9/16))**(8/27)
    Nu_cc = (0.825 + num/den)**2
    h_bot = Nu_cc * kb / L_caracteristico
    
    return h_top, h_bot

def get_electrical_source(T_celda_C):
    # Inciso b: Generación Joule I^2 * R
    # I depende de T
    # I(Tc) = 12.87 * (1 + 0.0004*(Tc - 25)) * (G_rad / 1000) -> G_rad es la incidente (800)
    # Ojo: enunciado dice "G_rad corresponde a la radiación recibida por el conjunto... 800"
    
    I_ref = 12.87
    G_ratio = G_solar / 1000.0
    I_val = I_ref * (1 + 0.0004 * (T_celda_C - 25.0)) * G_ratio
    
    Q_joule = I_val**2 * R_elec # Watts totales generados en la celda
    
    # Necesitamos Q volumetrico (W/m3) para la celda
    # Volumen celda = espesor * Ancho_celda * 1m (profundidad unitaria?)
    # Ojo: R_elec suele ser para el panel completo. 
    # El modelo 2D asume ancho unitario en Y. 
    # Asumiremos que Q_joule se distribuye en el volumen de la celda modelada.
    # Volumen_celda_modelado = Area_transversal * 1m
    # Area_transversal = Ancho_celda * espesor_celda
    # PERO cuidado: R_elec 215 mOhm es del panel completo (1.09m ancho x 2.4m alto).
    # Si metemos todo el calor de un panel de 2m2 en una franja 2D, explotará.
    # Debemos escalar Q_joule por densidad de área o volumen.
    
    # Area total panel = 2.416 * 1.09 = 2.63 m2
    q_joule_m2 = Q_joule / (L_caracteristico * L_horiz) # W/m2 de panel
    
    # Ahora pasamos a W/m3 dividiendo por espesor celda
    q_joule_vol = q_joule_m2 / L_celda
    
    return q_joule_vol

# --- 4. BUCLE PRINCIPAL (WIND SWEEP) ---

winds = [0, 1, 2, 3]
U_values = []

print(f"{'V_viento':<10} | {'T_celda_avg':<15} | {'U_global':<10}")
print("-" * 45)

for v in winds:
    # Reiniciar Temperatura para cada simulación
    T = np.ones((nz, nx)) * (40.0 + 273.15)
    
    # Solver Gauss-Seidel
    tol = 1e-4
    max_iter = 5000
    err = 1.0
    cnt = 0
    
    while err > tol and cnt < max_iter:
        cnt += 1
        T_old = T.copy()
        
        # 1. Temperaturas representativas
        T_sup_avg_C = np.mean(T[0, :]) - 273.15
        T_inf_avg_C = np.mean(T[-1, :]) - 273.15
        
        # Calcular T promedio de la celda (capa celda)
        # Nodos de celda están entre idx_e1_end y idx_c_end
        T_cell_nodes = T[idx_e1_end:idx_c_end, nx_borde:nx-nx_borde] # Solo parte activa
        T_cell_avg_C = np.mean(T_cell_nodes) - 273.15
        
        # 2. Actualizar h
        h_t, h_b = calc_h_coeffs(T_sup_avg_C, T_inf_avg_C, v)
        
        # 3. Actualizar Generación (Inciso b: Óptico + Joule)
        q_joule_dot = get_electrical_source(T_cell_avg_C)
        
        # 4. Barrido Espacial
        for j in range(nz):
            dzj = dz[j]
            for i in range(nx):
                dxi = dx[i]
                
                # Definir Q_dot local
                q_dot_local = 0.0
                # Vidrio Sup
                if j < idx_v1_end: q_dot_local = q_abs_vidrio / L_vidrio
                # Encap Sup
                elif j < idx_e1_end: q_dot_local = q_abs_enc / L_encap
                # Celda (Solo en zona activa)
                elif j < idx_c_end:
                    x_c = X_nodo[i]
                    if (x_c > Ancho_borde) and (x_c < Ancho_borde+Ancho_celda):
                        # Aquí sumamos Óptico + Joule
                        q_dot_local = (q_abs_celda_optico / L_celda) + q_joule_dot
                    else:
                        q_dot_local = 0.0 # Borde encapsulante
                # Resto 0
                
                # --- FVM (Gauss-Seidel) ---
                k_P = Mapa_K[j, i]
                sigma_anb = 0.0
                rhs = q_dot_local * dxi * dzj
                
                # Vecinos (simplificado para brevedad, misma lógica previa)
                # W
                if i>0:
                    dist = (dxi+dx[i-1])/2
                    k_int = 2*k_P*Mapa_K[j,i-1]/(k_P+Mapa_K[j,i-1]) if (k_P+Mapa_K[j,i-1])!=0 else k_P
                    aw = k_int*dzj/dist
                    sigma_anb += aw
                    rhs += aw*T[j,i-1]
                # E
                if i<nx-1:
                    dist = (dxi+dx[i+1])/2
                    k_int = 2*k_P*Mapa_K[j,i+1]/(k_P+Mapa_K[j,i+1]) if (k_P+Mapa_K[j,i+1])!=0 else k_P
                    ae = k_int*dzj/dist
                    sigma_anb += ae
                    rhs += ae*T[j,i+1] # old
                # N
                if j>0:
                    dist = (dzj+dz[j-1])/2
                    k_int = 2*k_P*Mapa_K[j-1,i]/(k_P+Mapa_K[j-1,i]) if (k_P+Mapa_K[j-1,i])!=0 else k_P
                    an = k_int*dxi/dist
                    sigma_anb += an
                    rhs += an*T[j-1,i]
                # S
                if j<nz-1:
                    dist = (dzj+dz[j+1])/2
                    k_int = 2*k_P*Mapa_K[j+1,i]/(k_P+Mapa_K[j+1,i]) if (k_P+Mapa_K[j+1,i])!=0 else k_P
                    aS = k_int*dxi/dist
                    sigma_anb += aS
                    rhs += aS*T[j+1,i] # old
                
                # Bordes Superficie
                if j==0: # Top
                    T_val = T[j,i]
                    # Rad
                    h_rad = eps_vidrio*sigma*((T_val**2+T_cielo**2)*(T_val+T_cielo)*F_cielo + \
                                              (T_val**2+T_suelo**2)*(T_val+T_suelo)*F_suelo_sup)
                    h_eff = h_t + h_rad
                    cond_int = k_P*dxi/(dzj/2)
                    sigma_anb += cond_int
                    rhs += cond_int * T[j,i] # Inercia nodo
                    # Fuente externa
                    rhs += h_eff*dxi * ((h_t*T_amb + h_rad*(F_cielo*T_cielo+F_suelo_sup*T_suelo))/h_eff) 
                    sigma_anb += h_eff*dxi
                    
                if j==nz-1: # Bot
                    T_val = T[j,i]
                    h_rad = eps_vidrio*sigma*(T_val**2+T_suelo**2)*(T_val+T_suelo)*F_suelo_inf
                    h_eff = h_b + h_rad
                    cond_int = k_P*dxi/(dzj/2)
                    sigma_anb += cond_int
                    rhs += cond_int * T[j,i]
                    rhs += h_eff*dxi * ((h_b*T_amb + h_rad*T_suelo)/h_eff)
                    sigma_anb += h_eff*dxi
                
                # Update
                T[j,i] = rhs / sigma_anb
        
        err = np.max(np.abs(T - T_old))
    
    # --- CÁLCULO DE U PARA ESTE VIENTO ---
    # U = Q_loss_total / (Area * (T_cell - T_amb))
    # Q_loss_total es la suma de calor que sale por arriba y por abajo (W)
    
    # Recuperamos T finales
    T_top_s = T[0, :]
    T_bot_s = T[-1, :]
    T_cell_avg = np.mean(T[idx_e1_end:idx_c_end, nx_borde:nx-nx_borde])
    
    # Calcular flujos locales y sumar
    Q_loss_top = 0.0
    Q_loss_bot = 0.0
    Area_simulada = Ancho_total * 1.0 # m2
    
    # Recalculamos h finales
    h_t_final, h_b_final = calc_h_coeffs(np.mean(T_top_s)-273.15, np.mean(T_bot_s)-273.15, v)
    
    for i in range(nx):
        dxi = dx[i]
        
        # Top
        T_val = T_top_s[i]
        q_conv = h_t_final * (T_val - T_amb)
        q_rad = eps_vidrio*sigma*(F_cielo*(T_val**4 - T_cielo**4) + F_suelo_sup*(T_val**4 - T_suelo**4))
        Q_loss_top += (q_conv + q_rad) * dxi
        
        # Bot
        T_val = T_bot_s[i]
        q_conv = h_b_final * (T_val - T_amb)
        q_rad = eps_vidrio*sigma*F_suelo_inf*(T_val**4 - T_suelo**4)
        Q_loss_bot += (q_conv + q_rad) * dxi

    Q_total = Q_loss_top + Q_loss_bot
    delta_T = T_cell_avg - T_amb
    
    U_global = (Q_total / Area_simulada) / delta_T
    U_values.append(U_global)
    
    print(f"{v:<10} | {T_cell_avg-273.15:<15.2f} | {U_global:<10.2f}")

# --- GRÁFICO ---
plt.figure(figsize=(7, 5))
plt.plot(winds, U_values, 'o-', linewidth=2, color='teal')
plt.xlabel('Velocidad del viento (m/s)')
plt.ylabel('Coeficiente Global U ($W/m^2 K$)')
plt.title('Coeficiente Global de Transferencia de Calor vs Viento')
plt.grid(True)
plt.xticks(winds)
plt.show()