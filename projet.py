import numpy as np
import matplotlib.pyplot as plt

def pression_compartiment(t, P0, P_alv, demi_vie):
    """
    Pression partielle d'azote dans un compartiment tissulaire au temps t (minutes),
    pour une pression alvéolaire P_alv constante (modèle de Haldane).
    P0 : pression initiale dans le compartiment (bar)
    demi_vie : demi-vie du compartiment (minutes)
    """
    k = np.log(2) / demi_vie
    return P_alv + (P0 - P_alv) * np.exp(-k * t)

# Plongée à 30 m
profondeur = 30                     # mètres
P_amb = 1 + profondeur / 10         # pression ambiante (bar), approximation classique
P_N2_air = 0.79                     # fraction d'azote dans l'air
P_alv = P_amb * P_N2_air            # pression partielle d'azote inspirée à 30m

P0 = 0.79 * 1.0   # équilibre en surface, 1 bar

temps = np.linspace(0, 120, 300)
demi_vie_test = 5   # minutes, compartiment "rapide"
P = pression_compartiment(temps, P0, P_alv, demi_vie_test)

plt.figure()
plt.plot(temps, P)
plt.axhline(P_alv, color='gray', linestyle='--', label="équilibre (P_alv)")
plt.xlabel("Temps (minutes)")
plt.ylabel("Pression partielle d'azote dans le tissu (bar)")
plt.title(f"Chargement d'un compartiment, demi-vie = {demi_vie_test} min")
plt.legend()
plt.show()

print(P[:5])

# Table ZHL-16C, azote (source : Bühlmann decompression algorithm, Wikipedia)
compartiments = [
    {"T": 5.0,   "a": 1.1696, "b": 0.5578},
    {"T": 8.0,   "a": 1.0,    "b": 0.6514},
    {"T": 12.5,  "a": 0.8618, "b": 0.7222},
    {"T": 18.5,  "a": 0.7562, "b": 0.7825},
    {"T": 27.0,  "a": 0.6200, "b": 0.8126},
    {"T": 38.3,  "a": 0.5043, "b": 0.8434},
    {"T": 54.3,  "a": 0.4410, "b": 0.8693},
    {"T": 77.0,  "a": 0.4000, "b": 0.8910},
    {"T": 109.0, "a": 0.3750, "b": 0.9092},
    {"T": 146.0, "a": 0.3500, "b": 0.9222},
    {"T": 187.0, "a": 0.3295, "b": 0.9319},
    {"T": 239.0, "a": 0.3065, "b": 0.9403},
    {"T": 305.0, "a": 0.2835, "b": 0.9477},
    {"T": 390.0, "a": 0.2610, "b": 0.9544},
    {"T": 498.0, "a": 0.2480, "b": 0.9602},
    {"T": 635.0, "a": 0.2327, "b": 0.9653},
]

temps = np.linspace(0, 120, 300)

plt.figure(figsize=(10, 6))
for i, comp in enumerate(compartiments):
    P = pression_compartiment(temps, P0, P_alv, comp["T"])
    plt.plot(temps, P, label=f"C{i+1} (T={comp['T']} min)")

plt.axhline(P_alv, color='black', linestyle='--', linewidth=1, label="équilibre (P_alv)")
plt.xlabel("Temps (minutes)")
plt.ylabel("Pression partielle d'azote (bar)")
plt.title("Chargement des 16 compartiments à 30 m")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()

def plafond(P_comp, a, b):
    """
    Pression ambiante minimale tolérée (plafond de décompression) pour un compartiment
    chargé à P_comp, selon les coefficients de Bühlmann a et b.
    """
    P_tol = b * (P_comp - a)
    return max(P_tol, 1.0)   # jamais en dessous de la surface (1 bar)

print("Plafond de décompression par compartiment, après 120 min à 30 m :")
for i, comp in enumerate(compartiments):
    P_comp = pression_compartiment(120, P0, P_alv, comp["T"])
    P_ceil = plafond(P_comp, comp["a"], comp["b"])
    profondeur_ceil = (P_ceil - 1) * 10   # conversion bar -> mètres
    print(f"C{i+1:>2} (T={comp['T']:>5} min) : P_comp={P_comp:.3f} bar, "
          f"plafond={P_ceil:.3f} bar ({profondeur_ceil:.1f} m)")

plafond_global = max(plafond(pression_compartiment(120, P0, P_alv, c["T"]), c["a"], c["b"])
                      for c in compartiments)
profondeur_globale = (plafond_global - 1) * 10
print(f"\nPlafond global (compartiment directeur) : {plafond_global:.3f} bar "
      f"({profondeur_globale:.1f} m)")

def plafond_global_a_t(t, P_alv, P0=0.79):
    """Plafond global (bar) après un temps t (minutes) à la pression alvéolaire P_alv."""
    plafonds = [plafond(pression_compartiment(t, P0, P_alv, c["T"]), c["a"], c["b"])
                for c in compartiments]
    return max(plafonds)

def limite_non_deco(profondeur, P0=0.79, t_max=300, tol=0.01):
    """
    Cherche par dichotomie le temps maximal (minutes) passé à `profondeur` (m)
    avant que le plafond global dépasse la surface (1.0 bar) — NDL.
    plafond_global_a_t est croissante en t, la dichotomie s'applique directement.
    """
    P_amb = 1 + profondeur / 10
    P_alv_loc = P_amb * P_N2_air

    if plafond_global_a_t(t_max, P_alv_loc, P0) <= 1.0:
        return t_max  # jamais de palier nécessaire dans la limite testée

    lo, hi = 0.0, t_max
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if plafond_global_a_t(mid, P_alv_loc, P0) > 1.0:
            hi = mid
        else:
            lo = mid
    return lo

for prof in [12, 18, 20, 25, 30, 35, 40]:
    ndl = limite_non_deco(prof)
    print(f"Profondeur {prof:>2} m : NDL ≈ {ndl:.1f} min")
    
    
def profil_libre(t, t_descente, prof_max, t_fond, vitesse_remontee=10):
    """Profondeur voulue (m) si on descend, reste au fond, puis remonte librement sans palier."""
    if t <= t_descente:
        return prof_max * (t / t_descente)
    elif t <= t_descente + t_fond:
        return prof_max
    else:
        t_remontee = t - (t_descente + t_fond)
        return max(prof_max - vitesse_remontee * t_remontee, 0)

def simuler_plongee(prof_max, t_fond, t_descente=2, vitesse_remontee=10,
                     dt=0.1, duree_max=400):
    """
    Simule une plongée complète avec paliers automatiques : à chaque instant,
    on essaie de remonter au rythme standard, mais on ne remonte jamais
    au-dessus du plafond de décompression courant (on s'y arrête si besoin).
    """
    P = np.full(16, P0)   # les 16 compartiments démarrent en équilibre surface

    temps = [0.0]
    profondeurs = [0.0]

    N = int(duree_max / dt)
    for i in range(1, N):
        t = i * dt
        prof_voulue = profil_libre(t, t_descente, prof_max, t_fond, vitesse_remontee)

        # plafond actuel, calculé sur l'état des compartiments à l'instant précédent
        plaf = max(plafond(P[c], compartiments[c]["a"], compartiments[c]["b"])
                   for c in range(16))
        profondeur_plafond = (plaf - 1) * 10

        # on ne remonte jamais plus haut que le plafond
        profondeur_actuelle = max(prof_voulue, profondeur_plafond)

        # on fait évoluer les 16 compartiments pendant ce pas de temps dt
        P_amb = 1 + profondeur_actuelle / 10
        P_alv_i = P_amb * P_N2_air
        for c in range(16):
            P[c] = pression_compartiment(dt, P[c], P_alv_i, compartiments[c]["T"])

        temps.append(t)
        profondeurs.append(profondeur_actuelle)

        if profondeur_actuelle <= 0 and prof_voulue <= 0:
            break

    return np.array(temps), np.array(profondeurs)

# Exemple : 30 m, 25 min de fond (bien au-delà du NDL de 15.4 min trouvé à l'étape 4
# → on s'attend à voir apparaître un ou plusieurs paliers)
temps, profondeurs = simuler_plongee(prof_max=30, t_fond=25)

plt.figure(figsize=(10, 5))
plt.plot(temps, -profondeurs)
plt.xlabel("Temps (minutes)")
plt.ylabel("Profondeur (m)")
plt.title("Profil de plongée avec paliers automatiques : 30 m, 25 min de fond")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Durée totale de la plongée (paliers inclus) : {temps[-1]:.1f} min")
