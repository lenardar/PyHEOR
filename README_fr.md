# PyHEOR — Python Health Economics and Outcome Research

[English](README.md) | [中文](README_zh.md) | **Français**

> **Modélisation en économie de la santé avec Python — aussi professionnel que hesim / DARTH en R, mais plus concis.**

PyHEOR est un framework Python dédié à la recherche en économie de la santé, prenant en charge :

| Fonctionnalité                    | Description                                                                                  |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| **Modèle de Markov par cohorte**  | Modèle de transition d'états en temps discret (cDTSTM), matrices de transition homogènes / dépendantes du temps |
| **Modèle de survie partitionnée (PSM)** | Répartition des probabilités d'états basée sur des courbes de survie paramétriques       |
| **Microsimulation**               | Modèle de transition d'états au niveau individuel avec hétérogénéité des patients, gestionnaires d'événements, ASP à deux niveaux |
| **Simulation à événements discrets (SED)** | Simulation individuelle en temps continu, risques concurrents, pilotée par des distributions temps-événement, intégration HR/AFT |
| **Distributions de survie paramétriques** | Exponentielle, Weibull, Log-logistique, Log-normale, Gompertz, Gamma généralisée, et 10 autres |
| **Définitions flexibles des coûts** | Coûts du premier cycle, fonctions dépendantes du temps, coûts ponctuels, méthode WLOS, calendriers de coûts de transition, fonctions de coûts personnalisées |
| **Cas de base / ASUD / ASP**     | Analyse déterministe, diagrammes en tornade (BNMI/RCEI), Monte Carlo + nuage de points CE + CACE |
| **Comparaison multi-stratégies & BNM** | Frontière d'efficience, détection de dominance/dominance étendue, courbes BNM, FACE, VEIP |
| **Ajustement de courbes de survie IPD** | Ajustement par MLE avec 6 distributions paramétriques, comparaison AIC/BIC, sélection automatique du meilleur modèle |
| **Numérisation et reconstruction de courbes KM** | Méthode de Guyot pour reconstruire les IPD à partir de courbes KM publiées, avec prétraitement du bruit de numérisation |
| **Intégration NMA**               | Importation d'échantillons postérieurs R, préservation des corrélations, génération automatique de courbes PH/AFT |
| **Analyse d'impact budgétaire (AIB)** | Modèles de taille de population, évolution des parts de marché, courbes d'adoption, analyse de scénarios/sensibilité univariée |
| **Calibration de modèle**         | Estimation de paramètres inconnus à partir de données observées : optimisation multi-départ Nelder-Mead, recherche aléatoire LHS, SSE/WSSE/vraisemblance GoF |
| **Visualisation**                 | 28 graphiques professionnels : diagrammes de transition d'états, graphiques de frontière, courbes BNM, FACE, VEIP, CACE, courbes KM + ajustées, graphiques d'impact AIB, etc. |
| **Exportation**                   | Exportation Excel multi-feuilles, modèle de vérification par formules Excel, rapports Markdown en un clic |

---

## Table des matières

- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Guide de l'utilisateur](#guide-de-lutilisateur)
  - [Types de modèles](#types-de-modèles) · [Système de paramètres](#système-de-paramètres) · [Matrice de transition](#matrice-de-transition) · [Coûts et utilités](#coûts-et-utilités) · [Analyse de survie](#analyse-de-survie) · [Analyse de sensibilité et rapports](#analyse-de-sensibilité-et-rapports) · [Fonctionnalités avancées](#fonctionnalités-avancées) · [Exportation](#exportation)
- [Galerie de visualisations](#galerie-de-visualisations)
- [Structure du projet](#structure-du-projet) · [Philosophie de conception](#philosophie-de-conception) · [Feuille de route](#feuille-de-route)

---

## Installation

```bash
# Installation depuis les sources
git clone <repo-url>
cd pyheor
pip install -e .
```

Dépendances : `numpy`, `pandas`, `matplotlib`, `scipy` (optionnel : `openpyxl` pour l'exportation Excel, `tabulate` pour les rapports Markdown)

---

## Démarrage rapide

```python
import pyheor as ph

# ── Définir le modèle ──
model = ph.MarkovModel(
    states=["Healthy", "Sick", "Dead"],
    strategies=["SOC", "Treatment"],
    n_cycles=40,
    cycle_length=1,
    dr_cost=ph.Param(0.03, low=0.0, high=0.08, label="Taux d'actualisation des coûts"),
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05, label="Taux d'actualisation des utilités"),
    half_cycle_correction=True,
)

# ── Paramètres ──
model.add_param("p_HS", base=0.15, low=0.10, high=0.20,
    dist=ph.Beta(mean=0.15, sd=0.03))
model.add_param("c_drug", base=2000, low=1500, high=2500,
    dist=ph.Gamma(mean=2000, sd=400))

# ── Matrice de transition (ph.C = complément) ──
model.set_transitions("SOC", lambda p, t: [
    [ph.C,  p["p_HS"], 0.02],
    [0,     ph.C,      0.10],
    [0,     0,         1   ],
])
model.set_transitions("Treatment", lambda p, t: [
    [ph.C,  p["p_HS"] * 0.7, 0.02],
    [0,     ph.C,             0.08],
    [0,     0,                1   ],
])

# ── Coûts et utilités ──
model.set_state_cost("medical", {"Healthy": 500, "Sick": 3000, "Dead": 0})
model.set_state_cost("drug", {
    "SOC": {"Healthy": 0, "Sick": 0, "Dead": 0},
    "Treatment": {
        "Healthy": lambda p, t: p["c_drug"],
        "Sick": lambda p, t: p["c_drug"],
        "Dead": 0,
    },
})
model.set_utility({"Healthy": 0.95, "Sick": 0.60, "Dead": 0.0})

# ── Exécuter les analyses ──
result = model.run_base_case()
print(result.summary())
print(result.icer())

owsa = model.run_owsa()       # Taux d'actualisation inclus automatiquement via Param
owsa.plot_tornado()

psa = model.run_psa(n_sim=1000)
psa.plot_ceac()

# ── Rapport Markdown en un clic ──
ph.generate_report(model, "report.md")
```

---

## Guide de l'utilisateur

### Types de modèles

#### Modèle de Markov par cohorte

Modèle de cohorte en temps discret (cDTSTM), adapté aux modèles simples avec des probabilités de transition d'états connues. Voir l'exemple complet dans [Démarrage rapide](#démarrage-rapide).

#### Modèle de survie partitionnée (PSM)

Dérive les proportions d'états à partir de courbes de survie paramétriques, adapté au cadre d'analyse PFS/OS couramment utilisé en économie de la santé en oncologie.

```python
import pyheor as ph

psm = ph.PSMModel(
    states=["PFS", "Progressed", "Dead"],
    survival_endpoints=["PFS", "OS"],
    strategies=["SOC", "New Drug"],
    n_cycles=120,
    cycle_length=1/12,
    dr_cost=0.03,
    dr_qaly=0.03,
)

# Courbes de survie de référence
baseline_pfs = ph.LogLogistic(shape=1.5, scale=18)
baseline_os = ph.Weibull(shape=1.2, scale=36)

# SOC : utiliser directement la référence
psm.set_survival("SOC", "PFS", baseline_pfs)
psm.set_survival("SOC", "OS", baseline_os)

# Nouveau médicament : modification HR / AFT
psm.set_survival("New Drug", "PFS",
    lambda p: ph.AcceleratedFailureTime(baseline_pfs, af=1.3))
psm.set_survival("New Drug", "OS",
    lambda p: ph.ProportionalHazards(baseline_os, hr=0.7))

# Coûts et utilités
psm.set_state_cost("treatment", {
    "SOC": {"PFS": 1000, "Progressed": 2500, "Dead": 0},
    "New Drug": {"PFS": 6000, "Progressed": 2500, "Dead": 0},
})
psm.set_utility({"PFS": 0.80, "Progressed": 0.55, "Dead": 0.0})

result = psm.run_base_case()
print(result.summary())
result.plot_survival()
result.plot_state_area()
```

#### Microsimulation

Modèle de transition d'états au niveau individuel partageant la même API que MarkovModel (`add_param`, `set_transitions`, `set_state_cost`, `set_utility`), mais chaque patient est échantillonné indépendamment, produisant des résultats individuels hétérogènes.

```python
import pyheor as ph

model = ph.MicroSimModel(
    states=["Healthy", "Sick", "Sicker", "Dead"],
    strategies=["SOC", "Treatment"],
    n_cycles=30,
    n_patients=5000,
    cycle_length=1.0,
    dr_cost=0.03,
    dr_qaly=0.03,
    seed=42,
)

model.add_param("p_HS", base=0.15, dist=ph.Beta(mean=0.15, sd=0.03))
model.add_param("hr_trt", base=0.70, dist=ph.LogNormal(mean=0.70, sd=0.10))

model.set_transitions("SOC", lambda p, t: [
    [ph.C,  p["p_HS"],                0,     0.005],
    [0,     ph.C,                     0.10,  0.05 ],
    [0,     0,                        ph.C,  0.10 ],
    [0,     0,                        0,     1    ],
])
model.set_transitions("Treatment", lambda p, t: [
    [ph.C,  p["p_HS"] * p["hr_trt"], 0,     0.005],
    [0,     ph.C,                     0.10 * p["hr_trt"], 0.05],
    [0,     0,                        ph.C,  0.10 ],
    [0,     0,                        0,     1    ],
])

model.set_state_cost("medical", {"Healthy": 500, "Sick": 3000, "Sicker": 8000, "Dead": 0})
model.set_state_cost("drug", {
    "SOC": {"Healthy": 0, "Sick": 0, "Sicker": 0, "Dead": 0},
    "Treatment": {"Healthy": 5000, "Sick": 5000, "Sicker": 5000, "Dead": 0},
})
model.set_utility({"Healthy": 0.95, "Sick": 0.75, "Sicker": 0.50, "Dead": 0.0})

# Gestionnaire d'événements : coût d'hospitalisation ponctuel à l'entrée dans Sicker
model.on_state_enter("Sicker", lambda idx, t, attrs: {"cost": 15000})

result = model.run_base_case(verbose=True)
print(result.summary())   # Inclut l'écart-type et les percentiles à 95%

# ASP : incertitude paramétrique externe × stochasticité individuelle interne
psa = model.run_psa(n_outer=500, n_inner=2000, seed=42)
psa.plot_ceac(wtp_range=(0, 150000))
```

**Hétérogénéité des patients** : Les probabilités de transition supportent un lambda à 3 arguments `(params, cycle, attrs)`, permettant des ajustements basés sur les attributs individuels (âge, sexe, etc.) :

```python
import numpy as np

pop = ph.PatientProfile(
    n_patients=5000,
    attributes={
        "age": np.random.normal(55, 12, 5000).clip(20, 90),
        "female": np.random.binomial(1, 0.52, 5000),
    }
)
model.set_population(pop)

model.set_transitions("SOC", lambda p, t, attrs: [
    [ph.C,  p["p_HS"] * (1 + (attrs["age"] - 55) * 0.02), 0.005],
    [0,     ph.C,  0.05],
    [0,     0,     1],
])
```

**Optimisation des performances** : Lorsque la matrice de transition ne dépend pas des attributs individuels (lambda à 2 arguments), le moteur utilise automatiquement un échantillonnage vectorisé par lots, atteignant des vitesses comparables au modèle de cohorte.

#### Simulation à événements discrets (SED)

La SED simule des patients individuels en **temps continu**, les temps d'événements étant échantillonnés directement à partir de distributions de survie, éliminant le besoin de durées de cycle fixes.

```python
import pyheor as ph

model = ph.DESModel(
    states=["PFS", "Progressed", "Dead"],
    strategies={"SOC": "Standard of Care", "TRT": "New Treatment"},
    time_horizon=40,
    dr_cost=0.03,
    dr_qaly=0.03,
)

model.add_param("hr_pfs", base=0.70,
    dist=ph.LogNormal(mean=-0.36, sd=0.15))

baseline_pfs2prog = ph.Weibull(shape=1.2, scale=5.0)
baseline_pfs2dead = ph.Weibull(shape=1.0, scale=20.0)
baseline_prog2dead = ph.Weibull(shape=1.5, scale=3.0)

# SOC : utiliser directement la référence
model.set_event("SOC", "PFS", "Progressed", baseline_pfs2prog)
model.set_event("SOC", "PFS", "Dead",       baseline_pfs2dead)
model.set_event("SOC", "Progressed", "Dead", baseline_prog2dead)

# TRT : HR appliqué à PFS→Progressed
model.set_event("TRT", "PFS", "Progressed",
    lambda p: ph.ProportionalHazards(baseline_pfs2prog, p["hr_pfs"]))
model.set_event("TRT", "PFS", "Dead",       baseline_pfs2dead)
model.set_event("TRT", "Progressed", "Dead", baseline_prog2dead)

# Coûts (taux en temps continu : $/an)
model.set_state_cost("drug", {
    "SOC": {"PFS": 500, "Progressed": 200, "Dead": 0},
    "TRT": {"PFS": 3000, "Progressed": 200, "Dead": 0},
})
model.set_state_cost("medical", {"PFS": 1000, "Progressed": 5000, "Dead": 0})
model.set_entry_cost("surgery", "Progressed", 50000)

model.set_utility({"PFS": 0.85, "Progressed": 0.50, "Dead": 0})

# Exécution
result = model.run(n_patients=3000, seed=42)
result.summary()
result.icer()

# ASP
psa = model.run_psa(n_sim=200, n_patients=1000, seed=123)
psa.summary()
```

**SED vs autres types de modèles** :

| Caractéristique | MarkovModel | MicroSimModel | DESModel |
|-----------------|-------------|---------------|----------|
| Axe temporel | Cycles discrets | Cycles discrets | Temps continu |
| Niveau d'analyse | Cohorte | Individuel | Individuel |
| Mécanisme de transition | Matrice de transition | Probabilités de transition | Distributions temps-événement |
| Risques concurrents | Traitement manuel requis | Traitement manuel requis | Support natif |
| Artefacts de cycle | Présents (correction demi-cycle requise) | Présents | Aucun |
| Vitesse | La plus rapide | Modérée | Plus lente |
| Cas d'utilisation | Modèles simples | Hétérogénéité complexe | Modèles complexes pilotés par événements |

---

### Système de paramètres

Chaque paramètre est défini via `add_param()`, contenant :

| Attribut           | Description                                                                            |
| ------------------ | -------------------------------------------------------------------------------------- |
| `base`           | Valeur de référence (analyse déterministe)                                             |
| `low` / `high` | Plage ASUD                                                                             |
| `dist`           | Distribution ASP (Beta, Gamma, Normal, LogNormal, Uniform, Triangular, Dirichlet, Fixed) |

```python
model.add_param("p_progression",
    base=0.15,           # Pour l'analyse du cas de base
    low=0.10, high=0.20, # Plage ASUD
    dist=ph.Beta(mean=0.15, sd=0.03),  # Pour l'ASP
    label="Probabilité de progression de la maladie",  # Pour l'affichage des graphiques
)
```

#### Taux d'actualisation

Tous les modèles définissent les taux d'actualisation via deux paramètres indépendants, `dr_cost` et `dr_qaly`. **La valeur par défaut est 0 (pas d'actualisation)** ; celui qui n'est pas défini ne sera pas actualisé.

```python
# Taux d'actualisation fixes
model = ph.MarkovModel(..., dr_cost=0.03, dr_qaly=0.03)

# Actualiser uniquement les coûts
model = ph.MarkovModel(..., dr_cost=0.06)  # dr_qaly par défaut à 0
```

Passez un objet `Param` pour inclure les taux d'actualisation dans l'ASUD / ASP sans avoir besoin d'un appel supplémentaire à `add_param()` :

```python
model = ph.MarkovModel(
    ...,
    dr_cost=ph.Param(0.03, low=0.0, high=0.08, label="Taux d'actualisation des coûts"),
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05, label="Taux d'actualisation des utilités"),
)

owsa = model.run_owsa()
owsa.plot_tornado()  # Le diagramme en tornade inclut les taux d'actualisation

# Vous pouvez aussi appliquer l'analyse de sensibilité à un seul
model = ph.MarkovModel(
    ...,
    dr_cost=0.03,                                        # Fixe
    dr_qaly=ph.Param(0.03, low=0.0, high=0.05),          # Variable
)
```

> **Principe de conception** : La valeur de référence et la plage d'analyse de sensibilité des taux d'actualisation sont définies au même endroit, évitant toute spécification redondante. `float` = valeur fixe, `Param` = valeur variable.

#### Correction de demi-cycle

| Valeur                     | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| `True` / `"trapezoidal"` | Méthode trapézoïdale : poids des premier et dernier cycles ×0.5 (par défaut) |
| `"life-table"`            | Méthode table de mortalité : moyenne des lignes adjacentes du trace (cohérent avec R heemod) |
| `False` / `None`          | Pas de correction                                         |

```python
model.half_cycle_correction = "life-table"
model.half_cycle_correction = "trapezoidal"
model.half_cycle_correction = False
```

---

### Matrice de transition

Utilisez `ph.C` (sentinelle de complément) pour calculer automatiquement les éléments diagonaux :

```python
# Matrice homogène dans le temps
model.set_transitions("Strategy", lambda p, t: [
    [ph.C,  p["p_AB"], p["p_AD"]],
    [0,     ph.C,      p["p_BD"]],
    [0,     0,         1        ],
])

# Matrice dépendante du temps (t est le numéro de cycle)
model.set_transitions("Strategy", lambda p, t: [
    [ph.C,  p["p_AB"] * (1 + 0.01 * t), p["p_AD"]],
    [0,     ph.C,                        p["p_BD"] + 0.001 * t],
    [0,     0,                           1],
])
```

---

### Coûts et utilités

#### Coûts d'état

```python
# Coût d'état de base
model.set_state_cost("Sick", "Treatment", lambda p, t: 3000)

# Coût dépendant du temps
model.set_state_cost("Sick", "Treatment", lambda p, t: 3000 if t < 5 else 2000)

# Coût ponctuel du premier cycle
model.set_state_cost("Sick", "Treatment", lambda p, t: 50000,
                     first_cycle_only=True)

# Restreint à des cycles spécifiques
model.set_state_cost("Sick", "Treatment", lambda p, t: p["c_drug"],
                     apply_cycles=(0, 24))  # Uniquement les 24 premiers cycles

# Méthode WLOS (Weighted Length of Stay)
model.set_state_cost("Sick", "Treatment", lambda p, t: 5000,
                     method="wlos")
```

#### Coûts de transition

Coûts déclenchés lors des transitions d'états (ex. : coûts chirurgicaux lors de la progression, coûts d'hospitalisation lors du transfert en soins intensifs). Calculés automatiquement à partir des **flux de transition** par cycle : `trace[t-1, de] × P[de→vers] × coût unitaire`.

```python
# Coût chirurgical lors de la transition de Healthy à Sick
model.set_transition_cost("surgery", "Healthy", "Sick", 50000)

# Référence de paramètre
model.set_transition_cost("surgery", "Healthy", "Sick", "c_surgery")

# Spécifique à la stratégie
model.set_transition_cost("icu", "Sick", "Dead", {
    "SOC": 20000,
    "Treatment": 15000,
})
```

**Calendriers de coûts** : Lorsqu'une transition déclenche des coûts s'étalant sur plusieurs cycles (ex. : chirurgie + suivi), passez une liste. Le moteur gère automatiquement l'empilement des coûts de plusieurs cohortes de patients en transition par convolution :

```python
# Progression : chirurgie 50000, suivi cycle suivant 10000 → s'étale sur 2 cycles
model.set_transition_cost("surgery", "PFS", "Progressed", [50000, 10000])

# Les références de paramètres peuvent aussi être utilisées dans les listes
model.set_transition_cost("chemo", "PFS", "Progressed",
    ["c_chemo_init", "c_chemo_maint", "c_chemo_maint"])

# Spécifique à la stratégie + utilisation mixte de calendriers
model.set_transition_cost("rescue", "PFS", "Progressed", {
    "SOC": [30000, 5000],       # Calendrier
    "New Drug": 15000,           # Scalaire
})
```

> **Différence avec `first_cycle_only`** : `first_cycle_only` ne s'applique qu'au cycle 0 (une seule fois) ; les coûts de transition sont encourus à **chaque cycle** dès que des patients transitent. Les coûts de transition ne sont pas affectés par la correction de demi-cycle (coûts de type événementiel).

#### Coûts personnalisés

Lorsque `set_transition_cost` avec des définitions par paire d'états n'est pas assez flexible, utilisez `set_custom_cost` pour passer une fonction personnalisée qui calcule les coûts directement à partir de la matrice de transition et de la distribution d'états. Pris en charge par MarkovModel et PSMModel.

```python
# Signature de la fonction
# func(strategy, params, t, state_prev, state_curr, P, states) -> float

# MarkovModel : calculer le coût chirurgical à partir des flux de transition
def surgery_cost(strategy, params, t, state_prev, state_curr, P, states):
    i_from = states.index("PFS")
    i_to = states.index("Progressed")
    flow = state_prev[i_from] * P[i_from, i_to]
    return flow * params["c_surgery"]

model.set_custom_cost("surgery", surgery_cost)

# PSMModel : calculer le coût de progression à partir des changements d'état (pas de matrice de transition, P=None)
def progression_cost(strategy, params, t, state_prev, state_curr, P, states):
    i_prog = states.index("Progressed")
    new_prog = max(0, state_curr[i_prog] - state_prev[i_prog])
    return new_prog * params["c_progression"]

psm.set_custom_cost("progression", progression_cost)
```

> Les coûts personnalisés ne sont pas affectés par la correction de demi-cycle (cohérent avec les coûts de transition). La fonction reçoit les valeurs des paramètres via `params`, et les variations et échantillonnages de paramètres ASUD/ASP se propagent naturellement.

---

### Analyse de survie

#### Distributions de survie paramétriques

10 distributions de survie intégrées :

| Distribution                       | Paramètres | Caractéristiques de la forme du risque      |
| ---------------------------------- | ---------- | ------------------------------------------- |
| `Exponential(rate)`              | λ         | Risque constant                              |
| `Weibull(shape, scale)`          | α, λ     | shape>1 croissant, <1 décroissant            |
| `LogLogistic(shape, scale)`      | α, λ     | shape>1 monte puis descend                   |
| `SurvLogNormal(meanlog, sdlog)`  | μ, σ     | Monte puis descend                           |
| `Gompertz(shape, rate)`          | a, b       | shape>0 croissant, <0 décroissant            |
| `GeneralizedGamma(mu, sigma, Q)` | μ, σ, Q  | Flexible (inclut Weibull, LogNormal comme cas particuliers) |

Distributions auxiliaires :

| Distribution                                 | Description                               |
| -------------------------------------------- | ----------------------------------------- |
| `ProportionalHazards(baseline, hr)`        | Risques proportionnels : h(t) = h₀(t) × HR |
| `AcceleratedFailureTime(baseline, af)`     | Temps de défaillance accéléré : S(t) = S₀(t/AF) |
| `KaplanMeier(times, probs)`                | Distribution empirique + extrapolation    |
| `PiecewiseExponential(breakpoints, rates)` | Risque constant par morceaux              |

Chaque distribution fournit les méthodes `survival(t)`, `hazard(t)`, `pdf(t)`, `quantile(p)`, `cumulative_hazard(t)`, `restricted_mean(t_max)`.

#### Ajustement de courbes de survie IPD

```python
import pyheor as ph
import pandas as pd

df = pd.read_csv("patient_data.csv")
fitter = ph.SurvivalFitter(
    time=df["time"],
    event=df["event"],
    label="Overall Survival",
)
fitter.fit()

# Tableau de comparaison AIC/BIC
print(fitter.summary())

# Sélection automatique du meilleur modèle
best = fitter.best_model()           # Par défaut : AIC
dist = best.distribution             # Utilisable directement dans PSM
print(fitter.selection_report())     # Rapport détaillé de sélection de modèle

# Graphiques de diagnostic
fitter.plot_fits()                   # KM + toutes les courbes ajustées
fitter.plot_hazard()                 # Fonctions de risque
fitter.plot_cumhazard_diagnostic()   # log(H) vs log(t)
fitter.plot_qq()                     # Graphique Q-Q

# Exportation
fitter.to_excel("fitting_results.xlsx")
```

**Critères de sélection de modèle** :

| Métrique       | Formule               | Description                                          |
| -------------- | --------------------- | ---------------------------------------------------- |
| AIC            | 2k - 2ln(L)           | Favorise bon ajustement + parcimonie ; adapté à la prédiction |
| BIC            | k·ln(n) - 2ln(L)     | Pénalise davantage la complexité que l'AIC ; adapté aux grands échantillons |
| ΔAIC          | AIC - AIC_min          | <2 non significatif, >10 différence décisive         |
| Poids AIC      | exp(-0.5·ΔAIC) / Σ  | Poids de vraisemblance relative du modèle            |

**Flux intégré IPD vers PSM** :

```python
fitter_os = ph.SurvivalFitter(time=df_os["time"], event=df_os["event"], label="OS")
fitter_pfs = ph.SurvivalFitter(time=df_pfs["time"], event=df_pfs["event"], label="PFS")
fitter_os.fit()
fitter_pfs.fit()

psm = ph.PSMModel(...)
psm.set_survival("SOC", "OS", fitter_os.best_model().distribution)
psm.set_survival("SOC", "PFS", fitter_pfs.best_model().distribution)
```

#### Numérisation et reconstruction de courbes KM

Reconstruire les IPD à partir de courbes KM publiées, permettant le flux complet : « Courbe KM publiée → IPD → Ajustement paramétrique → Modélisation ». Basé sur l'algorithme de Guyot et al. (2012).

```python
# 1. Obtenir les coordonnées KM à partir d'un outil de numérisation (ex. WebPlotDigitizer)
t_digitized = [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24]
s_digitized = [1.0, 0.92, 0.83, 0.74, 0.66, 0.58, 0.50, 0.40, 0.32, 0.25, 0.20]

# 2. Lire le tableau des effectifs à risque depuis la publication
t_risk = [0, 6, 12, 18, 24]
n_risk = [120, 88, 60, 38, 22]

# 3. Reconstruire les IPD
ipd_time, ipd_event = ph.guyot_reconstruct(
    t_digitized, s_digitized, t_risk, n_risk, tot_events=96,
)

# 4. Alimenter directement le SurvivalFitter
fitter = ph.SurvivalFitter(ipd_time, ipd_event, label="OS")
fitter.fit()
```

**Prétraitement des coordonnées numérisées** : `clean_digitized_km` fournit un nettoyage automatique (tri, suppression hors limites, détection des valeurs aberrantes, monotonie forcée, etc.). `guyot_reconstruct` l'appelle aussi en interne.

Références :

- Guyot P, Ades AE, Ouwens MJ, Welton NJ (2012). Enhanced secondary analysis of survival data. *BMC Med Res Methodol*, 12:9.
- Liu N, Zhou Y, Lee JJ (2021). IPDfromKM. *BMC Med Res Methodol*.

---

### Analyse de sensibilité et rapports

#### ASUD et ASP

```python
# ASUD (taux d'actualisation enregistrés automatiquement via Param)
owsa = model.run_owsa(wtp=50000)
print(owsa.summary(outcome="icer"))   # Trié par amplitude d'impact sur le RCEI
owsa.plot_tornado(outcome="nmb", max_params=10)

# ASP (Monte Carlo)
psa = model.run_psa(n_sim=1000, seed=42)
print(psa.summary())
print(psa.icer())
psa.plot_scatter(wtp=50000)
psa.plot_ceac()
psa.plot_convergence()
```

#### Rapport en un clic (`generate_report`)

Une fois les paramètres du modèle configurés, exécutez toutes les analyses et générez un rapport Markdown + figures associées en un seul appel :

```python
ph.generate_report(
    model,
    "report.md",       # Chemin de sortie ; figures enregistrées dans report_files/
    wtp=50000,          # Seuil de DAP
    n_sim=1000,         # Nombre de simulations ASP
    max_params=10,      # Nombre max de paramètres affichés dans le diagramme en tornade
    run_psa=None,       # None = détection auto (exécute si dist est défini)
)
```

Le rapport comprend : aperçu du modèle, tableau des paramètres, résultats du cas de base, RCEI, diagramme en tornade ASUD et tableau de classement, statistiques résumées ASP et analyse incrémentale, nuage de points du plan CE, et courbe CACE. Tous les types de modèles (Markov / PSM / MicroSim / SED) sont pris en charge.

---

### Fonctionnalités avancées

#### Comparaison multi-stratégies et analyse BNM

```python
# Créer une CEAnalysis à partir de résultats déterministes
result = model.run_base_case()
cea = ph.CEAnalysis.from_result(result)

# Frontière d'efficience : RCEI séquentiel + détection de dominance/dominance étendue
print(cea.frontier())

# Classement BNM
print(cea.nmb(wtp=100000))
print(f"Stratégie optimale : {cea.optimal_strategy(wtp=100000)}")

# Visualisation
cea.plot_frontier(wtp=100000)
cea.plot_nmb_curve(wtp_range=(0, 200000))
```

**ASP → FACE et VEIP** :

```python
psa_result = model.run_psa(n_sim=2000)
cea_psa = ph.CEAnalysis.from_psa(psa_result)

cea_psa.plot_ceaf(wtp_range=(0, 200000))
print(f"VEIP à DAP=100K$ : ${cea_psa.evpi_single(100000):,.0f}")
cea_psa.plot_evpi(wtp_range=(0, 200000), population=100000)
```

#### Intégration NMA

Le module NMA de PyHEOR est responsable de **l'importation et l'utilisation** des échantillons postérieurs produits par les packages R (gemtc / multinma / bnma).

```python
# Charger les échantillons postérieurs (supporte les formats CSV large/long)
nma = ph.load_nma_samples("nma_hr_samples.csv", log_scale=True)
print(nma.summary())

# Injection en lot dans les paramètres du modèle
nma.add_params_to_model(model, param_prefix="hr",
                        treatments=["Drug_A", "Drug_B"])

# Construire rapidement des courbes de survie
baseline = ph.Weibull(shape=1.2, scale=8.0)
curves = ph.make_ph_curves(baseline, nma)      # PH
curves_aft = ph.make_aft_curves(baseline, nma)  # AFT
```

| Classe / Fonction | Description |
|---|---|
| `load_nma_samples()` | Charger les postérieurs depuis CSV/Excel/Feather (format large/long, supporte la transformation log) |
| `NMAPosterior` | Conteneur postérieur fournissant `dist()` / `correlated()` / `summary()` / `add_params_to_model()` |
| `PosteriorDist` | Sous-classe de `Distribution`, échantillonne avec remplacement depuis la colonne postérieure |
| `CorrelatedPosterior` | Postérieur joint, échantillonnage sur la même ligne pour préserver les corrélations |
| `make_ph_curves()` / `make_aft_curves()` | Postérieur NMA + courbe de référence → dictionnaire de courbes PH/AFT |

#### Analyse d'impact budgétaire (AIB)

L'analyse d'impact budgétaire estime l'impact financier de l'introduction d'une nouvelle technologie sur le budget sur un horizon temporel court (généralement 1 à 5 ans). Conforme aux recommandations de bonnes pratiques AIB de l'ISPOR.

```python
bia = ph.BudgetImpactAnalysis(
    strategies=["Drug A", "Drug B", "Drug C"],
    per_patient_costs={"Drug A": 5000, "Drug B": 12000, "Drug C": 8000},
    population=10000,
    market_share_current={"Drug A": 0.6, "Drug B": 0.1, "Drug C": 0.3},
    market_share_new={"Drug A": 0.4, "Drug B": 0.3, "Drug C": 0.3},
    time_horizon=5,
)

bia.summary()
bia.cost_by_strategy()
```

**Modèles de population** :

```python
population=10000                                    # Population fixe
population=[10000, 10500, 11000, 11500, 12000]      # Spécifiée par année
population={"base": 10000, "growth_rate": 0.05}      # Croissance composée
population={"base": 10000, "annual_increase": 500}   # Croissance linéaire
```

**Courbes d'adoption des parts de marché** :

```python
ph.BudgetImpactAnalysis.linear_uptake(0.0, 0.4, 5)           # Linéaire
ph.BudgetImpactAnalysis.sigmoid_uptake(0.0, 0.4, 5, steepness=1.5)  # Sigmoïde
```

**Création à partir de résultats de modèle / Analyse de scénarios / Analyse de sensibilité** :

```python
# À partir de résultats de modèle
bia = ph.BudgetImpactAnalysis.from_result(result, population=10000, ...)

# Analyse de scénarios
bia.scenario_analysis({
    "Base Case": {},
    "High Population": {"population": 15000},
    "Fast Uptake": {"market_share_new": {"SoC": 0.3, "New": 0.7}},
})

# Sensibilité univariée
bia.one_way_sensitivity("population", values=[8000, 9000, 10000, 11000, 12000])

# Diagramme en tornade
bia.tornado({"population": (8000, 12000), "Drug B": (10000, 15000)})
```

#### Calibration de modèle

La calibration de modèle utilise des données observées pour estimer les paramètres du modèle qui ne peuvent pas être directement observés. Basée sur Vanni et al. (2011) et Alarid-Escudero et al. (2018).

```python
# Définir les cibles de calibration
targets = [
    ph.CalibrationTarget(
        name="10yr_healthy",
        observed=0.42, se=0.05,
        extract_fn=lambda sim: sim["SOC"]["trace"][10, 0],
    ),
]

# Définir les paramètres à calibrer
calib_params = [
    ph.CalibrationParam("p_HS", lower=0.01, upper=0.30),
    ph.CalibrationParam("p_SD", lower=0.01, upper=0.20),
]

# Exécuter la calibration
result = ph.calibrate(
    model, targets, calib_params,
    gof="wsse",
    method="nelder_mead",
    n_restarts=10,
    seed=42,
)

print(result.summary())
print(result.target_comparison())
result.apply_to_model(model)
```

| Méthode de recherche | Paramètres | Caractéristiques |
|----------------------|------------|------------------|
| `nelder_mead` | `n_restarts=10` | Optimisation sans dérivées multi-départ, précise mais plus lente |
| `random_search` | `n_samples=1000` | Échantillonnage LHS avec évaluation individuelle, simple et intuitif |

| Métrique GoF | Formule | Cas d'utilisation |
|--------------|---------|-------------------|
| `sse` | Σ(obs - préd)² | Par défaut, simple et rapide |
| `wsse` | Σ(obs - préd)²/SE² | Lorsque les cibles multiples ont des échelles différentes |
| `loglik_normal` | -Σ log N(obs \| préd, SE²) | Statistiquement fondé |

---

### Exportation

#### Exportation Excel

```python
# Exportation des données de résultats (multi-feuilles)
ph.export_to_excel(result, "base_case.xlsx")
ph.export_to_excel(owsa, "owsa.xlsx")
ph.export_to_excel(psa, "psa.xlsx")

# Comparaison multi-stratégies
ph.export_comparison_excel({"Strategy A": result_a, "Strategy B": result_b}, "comparison.xlsx")

# Résultats d'ajustement IPD
fitter.to_excel("fitting_results.xlsx")
```

#### Modèle de vérification par formules Excel

Exportez un fichier modèle complet qui **calcule indépendamment à l'aide de formules Excel**, pour la validation croisée des résultats Python :

```python
result = model.run_base_case()
ph.export_excel_model(result, "verification.xlsx")

# Ou exporter directement depuis le modèle
ph.export_excel_model(model, "verification.xlsx")
```

| Section | Contenu |
|---------|---------|
| **Zone d'entrée** (fond jaune) | Matrice de probabilités de transition, vecteur de coûts d'état, poids d'utilité, taux d'actualisation |
| **Zone de calcul** (formules) | `SUMPRODUCT` calcule Trace, Coûts, QALYs ; `SUM` calcule les totaux actualisés |
| **Feuille résumé** | Résultats formules Excel vs résultats Python vs différence (devrait être ~0) |

**Types de modèles pris en charge** :

| Modèle | Trace | Coûts/QALYs/Actualisation | RCEI |
|--------|-------|---------------------------|------|
| Markov (homogène dans le temps) | Formules Excel | Formules Excel | Formules Excel |
| Markov (dépendant du temps) | Valeurs Python | Formules Excel | Formules Excel |
| PSM | Valeurs de survie Python → formules de probabilité d'état | Formules Excel | Formules Excel |

#### Contenu des feuilles Excel

| Type d'analyse     | Contenu des feuilles                                                  |
| ------------------ | --------------------------------------------------------------------- |
| Cas de base        | Résumé, Trace d'état, Coût/QALY par cycle, RCEI                      |
| ASUD               | Données tornade, Résultats par paramètre                              |
| ASP                | Statistiques résumées, Toutes les simulations, Données CACE           |
| PSM Cas de base    | Résumé, Probabilités d'état, Données de survie                        |
| Ajustement IPD     | Comparaison de modèles, Données KM, Détails par distribution, Rapport de sélection |
| Modèle de vérification | Résumé (avec différences), Feuille de calcul par stratégie (formules + entrées) |

---

## Galerie de visualisations

PyHEOR fournit **28** graphiques professionnels, couvrant tous les types de modèles et flux d'analyse :

### Modèle de Markov (8 types)

| Fonction                        | Description                                  |
| ------------------------------- | -------------------------------------------- |
| `plot_transition_diagram()`   | Diagramme de transition d'états              |
| `plot_model_diagram()`        | Diagramme de modèle style TreeAge            |
| `plot_trace()`                | Trace de Markov (trajectoire de cohorte)     |
| `plot_tornado()`              | Diagramme en tornade ASUD                    |
| `plot_owsa_param()`           | Graphique linéaire ASUD à paramètre unique   |
| `plot_scatter()`              | Nuage de points CE (coût incrémental vs effet) |
| `plot_ceac()`                 | Courbe d'acceptabilité coût-efficacité       |
| `plot_convergence()`          | Graphique de diagnostic de convergence ASP   |

### Modèle PSM (4 types)

| Fonction                     | Description                       |
| ---------------------------- | --------------------------------- |
| `plot_survival_curves()`   | Courbes de survie paramétriques   |
| `plot_state_area()`        | Graphique en aires (proportions d'états) |
| `plot_psm_trace()`         | Trajectoire d'états PSM           |
| `plot_psm_comparison()`    | Comparaison de courbes de survie multi-stratégies |

### Microsimulation (3 types)

| Fonction                       | Description                                         |
| ------------------------------ | --------------------------------------------------- |
| `plot_microsim_trace()`      | Trajectoire des proportions d'états par simulation individuelle |
| `plot_microsim_survival()`   | Courbe de survie empirique (à partir de données simulées) |
| `plot_microsim_outcomes()`   | Distributions des résultats patients (histogrammes QALYs / Coûts / AV) |

### Ajustement IPD (4 types)

| Méthode                                | Description                        |
| -------------------------------------- | ---------------------------------- |
| `fitter.plot_fits()`                 | KM + toutes les courbes d'ajustement paramétrique |
| `fitter.plot_hazard()`               | Fonctions de risque par distribution |
| `fitter.plot_cumhazard_diagnostic()` | Graphique de diagnostic log(H) vs log(t) |
| `fitter.plot_qq()`                   | Graphique de quantiles Q-Q         |

### ACE / Comparaison multi-stratégies (4 types)

| Fonction               | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `plot_ce_frontier()`   | Frontière d'efficience + ligne DAP + étiquettes RCEI |
| `plot_nmb_curve()`     | Courbe BNM (stratégies multiples selon la DAP) |
| `plot_ceaf()`          | Frontière d'acceptabilité coût-efficacité (FACE) |
| `plot_evpi()`          | Courbe de la valeur espérée de l'information parfaite (VEIP) |

### Analyse d'impact budgétaire (5 types)

| Fonction                       | Description                                    |
| ------------------------------ | ---------------------------------------------- |
| `plot_budget_impact()`       | Diagramme à barres d'impact budgétaire annuel + courbe cumulative |
| `plot_budget_comparison()`   | Comparaison du coût total scénario actuel vs nouveau |
| `plot_market_share()`        | Graphique en aires empilées à double panneau des parts de marché |
| `plot_detail()`              | Ventilation des coûts empilés par stratégie     |
| `plot_tornado()`             | Diagramme en tornade de sensibilité AIB        |

---

## Structure du projet

```text
pyheor/
├── pyproject.toml
├── README.md
├── src/pyheor/              # Source du package (disposition src)
│   ├── __init__.py          # Entrée du package, exports unifiés
│   ├── utils.py             # Fonctions utilitaires (complément C, actualisation, validation)
│   ├── distributions.py     # Distributions de probabilité ASP (Beta, Gamma, ...)
│   ├── survival.py          # 10 distributions de survie paramétriques
│   ├── plotting.py          # Visualisation (28 types de graphiques)
│   │
│   ├── models/              # ── Moteur de modélisation ──
│   │   ├── markov.py        #  Modèle de Markov par cohorte (MarkovModel)
│   │   ├── psm.py           #  Modèle de survie partitionnée (PSMModel)
│   │   ├── microsim.py      #  Microsimulation (MicroSimModel)
│   │   └── des.py           #  Simulation à événements discrets (DESModel)
│   │
│   ├── analysis/            # ── Analyse et décision ──
│   │   ├── results.py       #  Classes de résultats (BaseResult, OWSAResult, PSAResult, ...)
│   │   ├── comparison.py    #  Comparaison multi-stratégies / ACE (CEAnalysis)
│   │   ├── bia.py           #  Analyse d'impact budgétaire (BudgetImpactAnalysis)
│   │   └── calibration.py   #  Calibration de modèle (Nelder-Mead, recherche aléatoire)
│   │
│   ├── evidence/            # ── Données et synthèse des preuves ──
│   │   ├── fitting.py       #  Ajustement de courbes de survie IPD (SurvivalFitter)
│   │   ├── digitize.py      #  Numérisation et reconstruction de courbes KM (méthode Guyot)
│   │   └── nma.py           #  Intégration d'échantillons postérieurs NMA (NMAPosterior)
│   │
│   └── export/              # ── Exportation ──
│       ├── excel.py         #  Exportation de données de résultats Excel
│       ├── excel_model.py   #  Exportation du modèle de vérification par formules Excel
│       └── report.py        #  Rapport Markdown en un clic
│
├── tests/                   # Suite de tests pytest (243 tests)
└── examples/
    ├── demo_hiv_model.py    #  Exemple de modèle de Markov (VIH)
    ├── demo_psm_model.py    #  Exemple de modèle PSM (oncologie)
    ├── demo_ipd_fitting.py  #  Exemple d'ajustement IPD
    ├── demo_microsim.py     #  Exemple de microsimulation
    └── demo_comparison.py   #  Exemple de comparaison multi-stratégies
```

---

## Philosophie de conception

- **API concise** : Un seul objet modèle gère le cas de base / ASUD / ASP sans appels séparés
- **Système de paramètres flexible** : `ph.C` complément automatique, fonctions lambda pour définir les probabilités/coûts dépendants du temps
- **Aligné avec l'écosystème R** : Paramétrisation des distributions et nommage des méthodes en référence à hesim / flexsurv / DARTH
- **Visualisation de qualité production** : Tous les graphiques fonctionnent immédiatement, palette de couleurs cohérente, personnalisable
- **Vérifiabilité** : Exportation Excel des données de trace pour une validation croisée facile avec les modèles TreeAge / Excel

---

## Feuille de route

- [X] Modèle de Markov par cohorte (cDTSTM)
- [X] Analyse de sensibilité univariée déterministe (ASUD) + diagramme en tornade
- [X] Analyse de sensibilité probabiliste (ASP) + CACE + nuage de points CE
- [X] Système de coûts flexible (premier cycle, dépendant du temps, WLOS, fonctions de coûts personnalisées)
- [X] Correction de demi-cycle multi-méthodes (trapézoïdale / table de mortalité / aucune) et taux d'actualisation configurables
- [X] Classement RCEI en tornade ASUD et taux d'actualisation directement inclus dans l'analyse de sensibilité via `Param`
- [X] Modèle de survie partitionnée (PSM)
- [X] 10 distributions de survie paramétriques
- [X] Exportation Excel multi-feuilles + modèle de vérification par formules Excel
- [X] Ajustement de courbes de survie IPD + comparaison de modèles AIC/BIC
- [X] Visualisation KM + courbes ajustées + graphiques de diagnostic
- [X] Microsimulation (simulation au niveau individuel)
- [X] Comparaison multi-cohortes + analyse BNM + FACE + VEIP
- [X] Intégration de méta-analyse en réseau (NMA)
- [X] Simulation à événements discrets (SED) — temps continu, risques concurrents, intégration HR/AFT
- [X] Analyse d'impact budgétaire (AIB) — modèles de population, évolution des parts de marché, courbes d'adoption, analyse de scénarios/sensibilité
- [X] Reconstruction de courbes KM numérisées (méthode Guyot)
- [X] Calibration de modèle (optimisation multi-départ Nelder-Mead, recherche aléatoire LHS, SSE/WSSE/vraisemblance GoF)
- [X] Rapport Markdown en un clic (`generate_report`)
- [X] Suite de tests formelle (pytest, 243 tests couvrant tous les modules)

---

## Licence

MIT License
