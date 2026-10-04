# PyHEOR

Bibliothèque Python de modélisation médico-économique et d'analyse coût-efficacité.

[English](README.md) · [中文](README_zh.md)

## Exemples de résultats

Figures produites par les exemples synthétiques du dépôt. Cliquez sur une image pour l'afficher en taille originale.

| Modélisation de la survie | Incertitude des paramètres |
|:---:|:---:|
| [<img src="examples/psm_oncology/figures/survival_curves.png" width="440" alt="Courbes PFS et OS de deux stratégies">](examples/psm_oncology/figures/survival_curves.png) | [<img src="examples/psm_oncology/figures/ce_scatter.png" width="440" alt="Nuage PSA des coûts et QALY différentiels">](examples/psm_oncology/figures/ce_scatter.png) |
| Comparer les courbes PFS et OS ajustées. | Explorer les coûts et QALY différentiels des tirages PSA. |
| **Acceptabilité coût-efficacité (CEAC)** | **Décision entre plusieurs stratégies (CEAF)** |
| [<img src="examples/psm_oncology/figures/ceac.png" width="440" alt="Courbes d'acceptabilité coût-efficacité de deux stratégies">](examples/psm_oncology/figures/ceac.png) | [<img src="examples/multi_strategy_comparison/figures/ceaf.png" width="440" alt="Frontière d'acceptabilité avec changements de stratégie">](examples/multi_strategy_comparison/figures/ceaf.png) |
| Suivre les probabilités selon le seuil WTP. | Identifier la stratégie recommandée et sa probabilité d'être coût-efficace. |

Reproduisez les figures avec les exemples [PSM oncologie](examples/psm_oncology/example.py) et [comparaison multistratégie](examples/multi_strategy_comparison/example.py). Le lissage visuel CEAC/CEAF est indiqué ; analyses et exports conservent les probabilités brutes.

## Installation

```bash
pip install -e .
```

Python 3.9+ ; NumPy, SciPy, pandas, matplotlib et openpyxl.

## Modèles et unités

MarkovModel, PSMModel et MicroSimModel exigent un `Cycle(length, unit)` explicite. DESModel utilise le temps continu avec `time_unit`. Les coûts et QALY des modèles à cycles sont des quantités **par cycle** ; les taux d'actualisation sont également définis par cycle. Le premier cycle n'est pas actualisé. `qaly()` convertit explicitement une utilité et une durée en QALY.

```python
import pyheor as ph

cycle = ph.Cycle(1, "month")
model = ph.MarkovModel(
    states=["Alive", "Dead"], strategies=["SOC", "TRT"],
    n_cycles=120, cycle=cycle, method="life-table",
    dr_cost=ph.rescale_discount_rate(.03, ph.Cycle(1, "year"), cycle),
    dr_qaly=ph.rescale_discount_rate(.03, ph.Cycle(1, "year"), cycle),
)
model.add_param("u_alive", base=.8, dist=ph.Beta(mean=.8, sd=.05))
model.set_transitions("SOC", [[.98, .02], [0, 1]])
model.set_transitions("TRT", [[.985, .015], [0, 1]])
model.set_state_cost("care", {
    "SOC": {"Alive": 1000}, "TRT": {"Alive": 1500},
})
model.set_state_qaly("health", {
    "Alive": lambda p, k: ph.qaly(p["u_alive"], cycle),
})
model.set_starting_cost("test", 200)
model.set_state_cost("loading", {"Alive": 500}, cycles=1)
model.set_transition_cost("terminal", "Alive", "Dead", 10000)

base = model.run_base_case()
print(base.summary())
print(base.icer())
psa = model.run_psa(n_sim=100, seed=42)
```

## Récompenses et survie

Interfaces parallèles pour coûts et QALY : état, début du modèle, entrée dans un état, transition et fonction personnalisée. `cycles=` limite une récompense d'état à certains cycles. Les callbacks reçoivent les paramètres et éventuellement le numéro du cycle (1…N) et les attributs du patient.

`method` accepte `beginning`, `end` et `life-table` (par défaut), avec les conventions de comptage et de correction des flux de heemod. PSM n'identifie pas les flux de transition individuels ; un état Terminal optionnel permet la comptabilisation documentée des coûts de fin de vie.

## Outils pratiques

Tous les outils sont accessibles avec `import pyheor as ph`. Les unités sont explicites ; les modèles ne les déduisent pas automatiquement.

| Outil | Utilisation |
|---|---|
| `Cycle(length, unit)` | Durée du cycle ; `.years`, `.in_unit(unit)` et `.time(k, unit=...)` convertissent durées et temps aux limites des cycles |
| `qaly(utility, duration, unit=None)` | Utilité × durée en années ; durée `Cycle` ou numérique avec unité explicite, tableaux et diminutions d'utilité acceptés |
| `rescale_discount_rate(rate, from_period, to_period)` | Taux effectif : `(1 + rate) ** (durée cible / durée source) - 1` ; deux objets `Cycle` ou deux durées numériques de même unité |
| `rescale_survival(curve, from_unit=..., to_period=...)` | Conversion du temps de survie vers les périodes du modèle, risques et quantiles inclus |
| `from_flexsurv(distribution, **parameters)` | Paramètres naturels R/flexsurv ; ni coefficients d'optimisation ni conversion automatique du temps |
| `ScaledSurvival(curve, factor)` | `S_new(t) = S_old(t * factor)` ; préférer `rescale_survival()` si les unités sont connues |
| `ProportionalHazards(curve, hr)` | `S_new(t) = S_old(t) ** hr` ; unité temporelle conservée |
| `AcceleratedFailureTime(curve, af)` | `S_new(t) = S_old(t / af)` ; `af=1.2` prolonge les temps de survie de 20% |
| `Beta`, `Gamma`, `LogNormal` avec `mean`/`sd` | Paramètres d'échantillonnage calculés à partir des moments naturels ; `LogNormal` accepte aussi `meanlog`/`sdlog` |
| `C` | Probabilité complémentaire d'une ligne de transition, au plus une par ligne ; `[ph.C, .02]` donne `C=.98` |

```python
import pyheor as ph

month = ph.Cycle(1, "month")
year = ph.Cycle(1, "year")
monthly_qaly = ph.qaly(.8, month)                    # 0.066667 QALY
monthly_cost = 12000 * month.years                  # Coût annuel → mensuel : 1000
monthly_dr = ph.rescale_discount_rate(.03, year, month)
three_month_qaly = ph.qaly(.8, 3, unit="month")      # 0.2 QALY
month_boundaries = month.time([0, 1, 12], unit="year")

# Courbe ajustée en années ; après conversion t=1 représente un mois.
curve = ph.from_flexsurv("weibull", shape=1.3, scale=1.5)
monthly_curve = ph.rescale_survival(
    curve, from_unit="year", to_period=month,
)
treated_curve = ph.ProportionalHazards(monthly_curve, hr=.75)
```

Paramètres acceptés par `from_flexsurv()` :

| Distribution | Paramètres |
|---|---|
| `exp` | `rate` |
| `weibull`, `weibullPH`, `llogis` | `shape`, `scale` |
| `lnorm` | `meanlog`, `sdlog` |
| `gompertz` | `shape`, `rate` |
| `gengamma` | `mu`, `sigma`, `Q` |
| `gengamma.orig` | `shape`, `scale`, `k` |

Les paramètres suivent la définition de chaque distribution R et ne sont pas interchangeables. Pour `weibullPH`, `scale` désigne le coefficient PH, converti en échelle Weibull PyHEOR par l'adaptateur.

Une année correspond à 12 mois, 52 semaines ou 365 jours ; il ne s'agit pas de dates calendaires. Multiplier les coûts annuels d'état par la durée du cycle en années ; saisir les coûts ponctuels à leur montant de survenue. Pour PSA/OWSA, effectuer les conversions dépendant des paramètres dans les callbacks afin de les recalculer à chaque tirage.

## Figures

| Figure | Utilisation | Méthode |
|---|---|---|
| Trajectoires des états | Proportions au fil du temps | Markov, PSM, MicroSim : `base.plot_trace()` |
| Courbes de survie | Comparaison des stratégies | PSM, MicroSim, DES : `base.plot_survival()` |
| Aire de survie partitionnée | États PFS, progression et décès | PSM : `base.plot_state_area()` |
| Histogrammes individuels | Distributions des coûts, QALY ou années de vie | MicroSim, DES : `base.plot_outcomes_histogram()` |
| Diagrammes de modèle / transition | Structure des états et transitions | Markov : `base.plot_model_diagram()`, `base.plot_transition_diagram()` |
| Tornade | Impact des paramètres | `owsa.plot_tornado()` |
| Nuage coût-efficacité PSA | Coûts et QALY différentiels | `psa.plot_scatter()` |
| CEAC | Probabilités coût-efficacité par stratégie | `psa.plot_ceac()` |
| Convergence PSA | Stabilité des simulations | Markov, PSM : `psa.plot_convergence()` |
| Frontière d'efficience / NMB | Comparaison des stratégies et seuils WTP | `cea.plot_frontier()`, `cea.plot_nmb_curve()` |
| CEAF / EVPI | Incertitude et valeur de l'information parfaite | Avec PSA : `cea.plot_ceaf()`, `cea.plot_evpi()` |

Les figures renvoient un objet Matplotlib `Figure`, personnalisable et exportable :

```python
fig = psa.plot_ceac(wtp_range=(0, 100000))
fig.savefig("ceac.png", dpi=150, bbox_inches="tight")

cea = ph.CEAnalysis.from_psa(psa)
cea.plot_ceaf(wtp_range=(0, 100000))
```

Le lissage visuel ne modifie ni l'analyse ni les valeurs exportées ; les courbes de survie empiriques conservent leurs escaliers.

## Résultats et exports

`base.summary()` et `base.icer()` présentent les totaux et l'analyse différentielle. `reward_components` détaille les coûts/QALY ; `cycle_rewards` et `state_occupancy` présentent les cycles. DES fournit aussi les événements et durées par état ; `metadata` décrit les unités et conventions.

Utiliser `ph.export_to_excel(base, "results.xlsx")` pour les tableaux, `ph.export_excel_model(base, "model.xlsx")` pour les classeurs à formules Markov/PSM et `ph.generate_report(model, "report.md")` pour un rapport d'analyse.

## Exemples

Consulter les [exemples](examples) et le [README anglais](README.md). L'historique des versions figure dans [CHANGELOG](CHANGELOG.md).

Tests : `pytest`. Licence : [AGPL-3.0-or-later](LICENSE).

Voir les [règles de développement et de version](CONTRIBUTING.md).
