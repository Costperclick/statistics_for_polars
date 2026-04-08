# Statistiques appliquées — Guide pratique

> Ce guide accompagne la lib `polars_stats`. Il couvre les fondamentaux dans l'ordre
> où tu en as besoin quand tu explores un jeu de données.

---

## Table des matières

1. [Le workflow statistique](#1-le-workflow-statistique)
2. [Étape 1 — Décrire les données](#2-étape-1--décrire-les-données)
3. [Étape 2 — Comprendre la forme (distribution)](#3-étape-2--comprendre-la-forme-distribution)
4. [Étape 3 — Tester la normalité](#4-étape-3--tester-la-normalité)
5. [Étape 4 — Détecter les outliers](#5-étape-4--détecter-les-outliers)
6. [Étape 5 — Tester une hypothèse](#6-étape-5--tester-une-hypothèse)
7. [Étape 6 — Mesurer la taille d'effet](#7-étape-6--mesurer-la-taille-deffet)
8. [Étape 7 — Estimer avec des intervalles de confiance](#8-étape-7--estimer-avec-des-intervalles-de-confiance)
9. [Étape 8 — Ajuster une distribution](#9-étape-8--ajuster-une-distribution)
10. [Arbre de décision — Quel test utiliser ?](#10-arbre-de-décision--quel-test-utiliser-)
11. [Glossaire](#11-glossaire)

---

## 1. Le workflow statistique

Quand tu reçois un jeu de données, l'analyse suit toujours le même ordre :

```
Données brutes
    │
    ▼
┌──────────────────────┐
│  1. DÉCRIRE           │  Moyenne, médiane, écart-type, forme
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  2. VISUALISER        │  Histogramme, QQ-plot, KDE
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  3. VÉRIFIER          │  Normalité, outliers
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  4. TESTER            │  Tests d'hypothèse
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  5. QUANTIFIER        │  Taille d'effet, intervalles de confiance
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  6. MODÉLISER         │  Ajustement de distribution, régression
└──────────────────────┘
```

**Règle d'or** : ne jamais sauter à l'étape 4 sans avoir fait les étapes 1 à 3.
Un test d'hypothèse sur des données qu'on n'a pas explorées, c'est du pilotage à l'aveugle.

---

## 2. Étape 1 — Décrire les données

### Les mesures de tendance centrale

Ce sont les réponses à la question : **"Quelle est la valeur typique ?"**

#### Moyenne arithmétique (`mean`)

La somme divisée par le nombre de valeurs.

```
Données : [10, 20, 30, 40, 50]
Moyenne : (10 + 20 + 30 + 40 + 50) / 5 = 30
```

**Avantage** : utilise toute l'information.
**Défaut** : un seul outlier peut la tirer violemment.

```
Données : [10, 20, 30, 40, 500]
Moyenne : 120   ← ne représente personne
Médiane : 30    ← plus fidèle
```

#### Médiane (`median`)

La valeur du milieu quand on trie les données.

```
Données triées : [10, 20, ►30◄, 40, 50]    → médiane = 30
Données triées : [10, 20, ►30, 40◄, 50, 60] → médiane = (30+40)/2 = 35
```

**Avantage** : insensible aux outliers.
**Défaut** : ignore les valeurs extrêmes, même quand elles sont informatives.

**Règle pratique** : si `mean ≈ median`, la distribution est symétrique. Si elles divergent, il y a de l'asymétrie.

#### Mode (`mode`)

La valeur la plus fréquente.

```
Données : [1, 2, 2, 3, 3, 3, 4]
Mode    : 3 (apparaît 3 fois)
```

Seule mesure de tendance centrale qui fonctionne sur des données catégorielles ("la couleur la plus vendue est le bleu").

#### Moyenne géométrique (`geometric_mean`)

La moyenne adaptée aux **données multiplicatives** (taux de croissance, rendements).

```
Rendements annuels : +10%, -20%, +30%
Multiplicateurs    : [1.10, 0.80, 1.30]

Moyenne arithmétique : (1.10 + 0.80 + 1.30) / 3 = 1.067  → +6.7% ← FAUX
Moyenne géométrique  : (1.10 × 0.80 × 1.30)^(1/3) = 1.046 → +4.6% ← CORRECT
```

**Utilise-la quand** : les données se multiplient entre elles (rendements, taux, ratios).

#### Moyenne harmonique (`harmonic_mean`)

La moyenne adaptée aux **taux avec numérateur fixe** (vitesses, prix unitaires).

```
Aller  : 60 km/h sur 120 km → 2h
Retour : 40 km/h sur 120 km → 3h
Total  : 240 km en 5h = 48 km/h

Moyenne arithmétique : (60 + 40) / 2 = 50 km/h  ← FAUX
Moyenne harmonique   : 2 / (1/60 + 1/40) = 48 km/h ← CORRECT
```

**Utilise-la quand** : le numérateur du ratio est fixe (même distance, même budget, même volume).

#### Moyenne pondérée (`weighted_mean`)

Chaque valeur a un poids différent.

```
Notes   : [12, 15, 8]
Coeffs  : [1,  3,  2]

Moyenne pondérée : (12×1 + 15×3 + 8×2) / (1+3+2) = 73/6 = 12.17
```

---

### Quand utiliser quelle moyenne ?

```
┌──────────────────────────────────────────────────────────┐
│ Les données s'additionnent ?     → Moyenne arithmétique  │
│ Les données se multiplient ?     → Moyenne géométrique   │
│ Les données sont des taux (÷) ?  → Moyenne harmonique    │
│ Certaines valeurs pèsent plus ?  → Moyenne pondérée      │
│ Il y a des outliers ?            → Médiane                │
│ Les données sont catégorielles ? → Mode                   │
└──────────────────────────────────────────────────────────┘
```

Et dans tous les cas : **moyenne arithmétique ≥ moyenne géométrique ≥ moyenne harmonique**
(pour des valeurs positives). C'est une propriété mathématique, pas un hasard.

---

### Les mesures de dispersion

Répondent à la question : **"Les données sont-elles resserrées ou étalées ?"**

#### Variance (`variance`) et écart-type (`standard_deviation`)

La variance mesure la distance moyenne au carré par rapport à la moyenne.
L'écart-type est sa racine carrée — il a la même unité que les données.

```
Données : [2, 4, 6, 8, 10]
Moyenne : 6

Écarts      : [-4, -2, 0, 2, 4]
Écarts²     : [16,  4, 0, 4, 16]
Variance    : (16+4+0+4+16) / (5-1) = 10
Écart-type  : √10 ≈ 3.16
```

**La règle des 68-95-99.7** (pour les données normales) :

```
         ◄── 68% ──►
      ◄──── 95% ────►
   ◄────── 99.7% ──────►

   ╔════════════════════════════════╗
   ║      ·  ·  █  ·  ·            ║
   ║    ·  ████████  ·             ║
   ║  · █████████████ ·            ║
   ║ ████████████████████          ║
   ╚════════════════════════════════╝
  -3σ  -2σ  -1σ   μ   +1σ  +2σ  +3σ
```

- ~68% des données entre μ ± 1σ
- ~95% entre μ ± 2σ
- ~99.7% entre μ ± 3σ

#### Pourquoi (n-1) et pas (n) ?

On divise par `n-1` (correction de Bessel) parce qu'on estime la variance d'une **population**
à partir d'un **échantillon**. Diviser par `n` sous-estime systématiquement. Le `-1` corrige ce biais.

Si tu as la population entière (rare), utilise `n`. Sinon, `n-1`.

#### MAD — Median Absolute Deviation (`median_absolute_deviation`)

Version robuste de l'écart-type. Utilise la médiane au lieu de la moyenne.

```
Données : [2, 4, 5, 7, 200]

Écart-type : 87.7  ← explosé par le 200
MAD        : 2.0   ← stable
```

**Utilise MAD quand** : tu as des outliers, ou tu n'es pas sûr d'en avoir.

#### IQR — Interquartile Range (`iqr`)

La plage du milieu 50% des données.

```
Données triées : [1, 3, │5, 7, 9│, 11, 13]
                        Q1=5    Q3=11
IQR = Q3 - Q1 = 11 - 5 = 6
```

C'est ce qui définit la "boîte" dans un boxplot.

#### Coefficient de variation (`variance_coefficient`)

L'écart-type divisé par la moyenne. Permet de comparer la dispersion entre séries
d'échelles différentes.

```
Salaires    : mean=50 000€, std=10 000€ → CV = 0.20
Nb commandes: mean=500,     std=100     → CV = 0.20
→ Même variabilité relative malgré des échelles très différentes
```

#### Range (`data_range`)

Le plus simple : max - min. Très sensible aux outliers. Utile uniquement pour un coup d'œil rapide.

---

### Les mesures de forme

Répondent à la question : **"À quoi ressemble la distribution ?"**

#### Skewness — Asymétrie (`skewness`)

```
Skewness négatif          Skewness = 0            Skewness positif
(queue à gauche)          (symétrique)            (queue à droite)

        ██                    ██                  ██
       ████                  ████                ████
      ██████                ██████              ██████
    ████████              ████████            ████████
  ████████████          ████████████        ████████████
◄─────────────        ─────────────        ─────────────►
```

- **= 0** : symétrique
- **> 0** : queue à droite (salaires, prix immobiliers)
- **< 0** : queue à gauche (notes d'examen avec plafond)

**Repères** : entre -0.5 et +0.5 → à peu près symétrique. Au-delà de ±1 → asymétrie marquée.

#### Kurtosis — Épaisseur des queues (`kurtosis`)

```
Kurtosis < 0              Kurtosis = 0            Kurtosis > 0
(queues légères)          (normal)                (queues lourdes)
Platykurtique             Mésokurtique            Leptokurtique

  ████████████              ·····                    ██
  ████████████             ·█████·                  ████
  ████████████            ·███████·                ██████
  ████████████           ·█████████·            ████████████
  ████████████          ·███████████·        ██████████████████
```

- **= 0** : queues normales
- **> 0** : queues lourdes, les événements extrêmes arrivent plus souvent (risque !)
- **< 0** : queues légères, données très concentrées

**Cas d'usage clé** : en finance, un kurtosis élevé = risque de "cygne noir" sous-estimé.

---

### Diversité et concentration

#### Entropie de Shannon (`s_entropy`)

Mesure le niveau de "surprise" dans tes données.

```
Boutique A : [chaussures 98%, pulls 1%, sacs 1%]   → Entropie basse (prévisible)
Boutique B : [chaussures 33%, pulls 33%, sacs 34%]  → Entropie haute (diverse)
```

- **0** = une seule catégorie, aucune surprise
- **ln(k)** = maximum, toutes les k catégories sont équiprobables

#### Coefficient de Gini (`gini`)

Mesure l'inégalité de répartition.

```
                     Ligne d'égalité parfaite (Gini = 0)
                   /
  100% ┌────────/──────────────┐
       │      /  ·····         │
   %   │    /  ··              │  ← Courbe de Lorenz (données réelles)
  cum.  │  / ··                 │
       │/··                    │
   0%  └───────────────────────┘
       0%                    100%
              % population

  Gini = Aire entre la diagonale et la courbe
         ────────────────────────────────────
           Aire totale sous la diagonale
```

- **0** = égalité parfaite (tout le monde a la même valeur)
- **1** = inégalité maximale (une personne a tout)

**Exemples** :
- Revenus par client → Gini élevé = quelques clients font tout le CA
- Charge par serveur → Gini élevé = mauvais load balancing

---

## 3. Étape 2 — Comprendre la forme (distribution)

### Qu'est-ce qu'une distribution ?

Si tu mesures la taille de 10 000 personnes et que tu traces un histogramme,
tu verras une forme de cloche apparaître. Cette forme, c'est la **distribution**.

Une distribution c'est un modèle mathématique qui décrit :
- Quelles valeurs sont possibles
- Avec quelle probabilité chacune apparaît

### Les distributions à connaître

#### Loi normale (gaussienne)

La plus importante. La courbe en cloche.

```
               ████
             ████████
           ████████████
         ████████████████
       ████████████████████
    █████████████████████████████
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                 μ
```

**Quand** : tailles, poids, erreurs de mesure, moyennes d'échantillons.
**Paramètres** : μ (moyenne), σ (écart-type).
**Propriété clé** : symétrique, 68-95-99.7.

#### Loi exponentielle

Le temps entre deux événements.

```
  ██
  ████
  ██████
  ████████
  ████████████
  ██████████████████
  ████████████████████████████████
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  0
```

**Quand** : temps entre deux achats, temps entre deux pannes, durée d'un appel.
**Paramètre** : λ (taux, inverse de la moyenne).
**Propriété clé** : "sans mémoire" — le temps déjà écoulé n'influence pas le temps restant.

#### Loi log-normale

Des données dont le logarithme suit une loi normale.

```
     ████
    ██████
   ████████
  ██████████
  ████████████
  ██████████████████
  ██████████████████████████████████
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  0
```

**Quand** : revenus, prix immobiliers, tailles de fichiers, durées de session.
**Propriété clé** : ressemble à l'exponentielle mais avec une bosse à gauche.
Beaucoup de petites valeurs, quelques très grandes.

#### Loi de Poisson

Le nombre d'événements dans un intervalle fixe.

```
           ██
         ██████
        ████████
       ██████████
      ████████████
    ████████████████
  ████████████████████████
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  0  1  2  3  4  5  6  7  8
```

**Quand** : nombre de bugs par jour, nombre d'emails par heure, nombre d'accidents par mois.
**Paramètre** : λ (moyenne = variance).
**Propriété clé** : données entières uniquement. Si moyenne ≈ variance, probablement Poisson.

#### Loi uniforme

Toutes les valeurs ont la même probabilité.

```
  ████████████████████████████████
  ████████████████████████████████
  ████████████████████████████████
  ████████████████████████████████
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  a                              b
```

**Quand** : dé, générateur aléatoire, tirage au sort.
**Paramètres** : a (min), b (max).
**Rare dans les vraies données** — si tes données semblent uniformes, c'est souvent un artefact.

### Comment reconnaître une distribution ?

```
┌─────────────────────────────────────────────────────────────────────┐
│  Indice                          │  Distribution probable           │
├─────────────────────────────────────────────────────────────────────┤
│  Symétrique, en cloche           │  Normale                        │
│  Skew > 0, part de 0             │  Exponentielle ou Log-normale   │
│  Données entières, rares         │  Poisson                        │
│  Plate, bornée                   │  Uniforme                       │
│  Moyenne ≈ Variance (comptages)  │  Poisson                        │
│  log(données) semble normal      │  Log-normale                    │
│  Skew > 0, bosse puis queue      │  Log-normale ou Gamma           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Étape 3 — Tester la normalité

### Pourquoi tester la normalité ?

Beaucoup de tests statistiques (t-test, ANOVA, régression linéaire) **supposent** que les données
sont normales. Si elles ne le sont pas, les résultats de ces tests peuvent être faux.

### Quel test utiliser ?

```
                         Tes données
                             │
                     Combien de valeurs ?
                        /          \
                    < 5000        ≥ 5000
                      │              │
                Shapiro-Wilk    D'Agostino-Pearson
                      │              │
                  p-value          p-value
                 /       \        /       \
             ≥ 0.05    < 0.05  ≥ 0.05   < 0.05
               │         │      │         │
          "Probablement  "On    "Probablement  "On
           normal"     rejette   normal"      rejette
                       la                     la
                       normalité"             normalité"
```

### Comment interpréter la p-value ?

La p-value répond à cette question : **"Si mes données étaient vraiment normales,
quelle serait la probabilité d'observer un résultat aussi extrême que celui-ci ?"**

- **p ≥ 0.05** → "Les données sont compatibles avec une loi normale." On ne rejette pas.
- **p < 0.05** → "Il serait très surprenant d'observer ça si les données étaient normales." On rejette.

**Attention** : p ≥ 0.05 ne **prouve** pas la normalité. Ça dit juste qu'on n'a pas assez de preuves pour la rejeter. Nuance importante.

### Le QQ-plot — le diagnostic visuel

Plus intuitif qu'un test formel. On compare les quantiles de tes données (axe Y)
aux quantiles théoriques d'une loi normale (axe X).

```
  Données normales           Queues lourdes            Asymétrie droite

  observé                    observé                   observé
  │      ··/                 │      ·  /               │     ·    /
  │    ·/                    │    ·  /                  │    ·   /
  │  ·/                      │   · /                    │   ·  /
  │ /·                       │  / ·                     │  / ·
  │/·                        │ / ·                      │ /·
  │·                         │/·                        │/·
  └────── théorique          └────── théorique          └────── théorique

  Points sur la diagonale    Points qui s'écartent      Points qui courbent
  → normal ✓                 aux extrémités → non       d'un côté → non
```

Si les points suivent la diagonale → normal. Si ça dévie → pas normal.

---

## 5. Étape 4 — Détecter les outliers

### Qu'est-ce qu'un outlier ?

Une valeur anormalement éloignée des autres. Ça peut être :
- Une **erreur** (saisie, capteur, bug) → à corriger
- Un **signal** (fraude, anomalie, événement rare) → à investiguer
- Une **vraie valeur extrême** → à garder mais à comprendre

**Ne jamais supprimer un outlier sans comprendre pourquoi il est là.**

### Les trois méthodes

#### Z-score (`outliers_zscore`)

"À combien d'écarts-types de la moyenne ?"

```
  │                     │  seuil = 3σ
  │      ████████       │
  │    ████████████     │
  │  ████████████████   │
  │████████████████████ │  ·  ← outlier
  └─────────────────────┴──────
  -3σ   -1σ   μ   +1σ  3σ
```

**Défaut** : les outliers eux-mêmes gonflent la moyenne et l'écart-type, ce qui peut masquer d'autres outliers.

#### MAD (`outliers_mad`)

Même principe mais avec la médiane et la MAD au lieu de la moyenne et l'écart-type.
Les outliers n'influencent pas le seuil de détection. C'est la méthode la plus robuste.

#### IQR / Tukey (`outliers_iqr`)

C'est la méthode des boxplots.

```
   outlier     Q1    médiane   Q3      outlier
     ·    ┣━━━━┫───────┼───────┫━━━━┫     ·
           │                         │
     Q1 - 1.5×IQR             Q3 + 1.5×IQR
```

Tout ce qui dépasse les "moustaches" est un outlier.

### Quelle méthode choisir ?

```
┌──────────────────────────────────────────────────────────────────┐
│  Situation                          │  Méthode recommandée       │
├──────────────────────────────────────────────────────────────────┤
│  Données normales, pas d'outliers   │  outliers_zscore           │
│  Données avec outliers existants    │  outliers_mad (robuste)    │
│  Données non-normales / asymétriques│  outliers_iqr              │
│  Besoin d'une preuve statistique    │  outliers_grubbs           │
│  Pas sûr                            │  outliers_mad              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Étape 5 — Tester une hypothèse

### Le principe

Un test d'hypothèse répond à une question oui/non avec un niveau de confiance.

**Exemple** : "Le temps de réponse moyen de notre API est-il différent de 200ms ?"

On formule deux hypothèses :
- **H0** (hypothèse nulle) : "Non, la moyenne est 200ms." (le statu quo)
- **H1** (hypothèse alternative) : "Si, la moyenne est différente de 200ms."

Le test calcule une p-value. Si elle est inférieure au seuil alpha (souvent 0.05),
on rejette H0. Sinon, on ne rejette pas.

### Analogie avec un procès

```
  H0 = "L'accusé est innocent"     (présomption d'innocence)
  H1 = "L'accusé est coupable"

  Le test statistique = le procès
  La p-value = la force des preuves

  p < 0.05  → preuves suffisantes → on rejette H0 → "coupable"
  p ≥ 0.05  → preuves insuffisantes → on ne rejette pas H0 → "non coupable"
                                        (pas "innocent" !)
```

### Les erreurs possibles

```
                          Réalité
                    H0 vraie    H0 fausse
                  ┌───────────┬───────────┐
  Décision  On ne │     ✓     │ Erreur    │
            rejette│  Correct  │ Type II   │
            pas   │           │ (β)       │
                  ├───────────┼───────────┤
            On    │ Erreur    │     ✓     │
            rejette│ Type I    │  Correct  │
                  │ (α)       │           │
                  └───────────┴───────────┘

  Erreur Type I  (faux positif) : on croit voir un effet qui n'existe pas
  Erreur Type II (faux négatif) : on rate un effet qui existe
```

Alpha (souvent 0.05) contrôle le risque d'erreur Type I. C'est toi qui le choisis.

### Quel test one-sample utiliser ?

```
                    Tes données
                        │
                 Sont-elles normales ?
                   /            \
                 Oui             Non
                  │               │
            ttest_1samp     La distribution
                  │         est-elle symétrique ?
                  │            /          \
                  │          Oui          Non
                  │           │            │
                  │    wilcoxon_1samp   sign_test
                  │
            (plus puissant → détecte des effets plus petits)
```

### Interpréter les résultats

```python
p = ttest_1samp(response_times, mu=200)

if p < 0.05:
    # "Le temps moyen est significativement différent de 200ms"
    # MAIS ça ne dit pas si la différence est GRANDE ou PETITE
    # → regarde le Cohen's d
else:
    # "On n'a pas assez de preuves pour dire que c'est différent de 200ms"
    # MAIS ça ne prouve pas que c'est égal
    # → peut-être juste pas assez de données
```

**P-value basse ≠ effet important.** Avec assez de données, même une différence de 0.001ms
peut être "significative". C'est pour ça qu'on a l'étape suivante.

---

## 7. Étape 6 — Mesurer la taille d'effet

### Pourquoi la p-value ne suffit pas

La p-value dit : "est-ce que l'effet existe ?"
La taille d'effet dit : "est-ce que l'effet est grand ?"

```
  Exemple 1 : n = 1 000 000, différence = 0.01ms, p = 0.001
  → Statistiquement significatif, mais on s'en fiche (effet minuscule)

  Exemple 2 : n = 10, différence = 50ms, p = 0.08
  → Non significatif, mais l'effet est potentiellement gros (pas assez de données)
```

### Cohen's d (`cohens_d`)

Distance entre la moyenne observée et la valeur de référence, en nombre d'écarts-types.

```
  d = (moyenne observée - μ₀) / écart-type

  Interprétation :
  ┌────────────────────────────────────┐
  │  |d| ≈ 0.2  →  petit effet        │
  │  |d| ≈ 0.5  →  effet moyen        │
  │  |d| ≈ 0.8  →  grand effet        │
  └────────────────────────────────────┘
```

Visualisation :

```
  Petit (d=0.2)                Moyen (d=0.5)               Grand (d=0.8)

    ███   ███                  ███     ███                 ███       ███
   █████ █████                █████   █████               █████     █████
  ███████████████            ██████████████              █████████████████
  Beaucoup de               Chevauchement              Groupes plus
  chevauchement             modéré                     distincts
```

**Bonne pratique** : toujours rapporter la taille d'effet à côté de la p-value.

---

## 8. Étape 7 — Estimer avec des intervalles de confiance

### Qu'est-ce qu'un intervalle de confiance ?

Au lieu de dire "la moyenne est 42", on dit "la moyenne est entre 39 et 45
avec 95% de confiance".

C'est plus honnête — ça reflète notre incertitude.

```
  Estimation ponctuelle :     ●               "La moyenne est 42"
  Intervalle de confiance :   [━━━━━●━━━━━]   "Entre 39 et 45 (95%)"
                              39    42   45
```

### Comment l'interpréter ?

"Si on répétait l'échantillonnage 100 fois, environ 95 des intervalles calculés
contiendraient la vraie valeur."

Ce n'est **PAS** "il y a 95% de chances que la vraie valeur soit dans cet intervalle".
(La vraie valeur est fixe, c'est l'intervalle qui varie d'un échantillon à l'autre.)

### Largeur de l'intervalle

```
  Petit échantillon :  [━━━━━━━━━━━━━━●━━━━━━━━━━━━━━]   Large → beaucoup d'incertitude
  Grand échantillon :       [━━━━━●━━━━━]                  Étroit → plus de précision

  Confiance 90% :           [━━━━━●━━━━━]                  Plus étroit
  Confiance 95% :         [━━━━━━━●━━━━━━━]                Plus large
  Confiance 99% :      [━━━━━━━━━━●━━━━━━━━━━]             Encore plus large
```

Plus tu veux être sûr, plus l'intervalle est large. C'est le compromis précision/confiance.

### Méthodes

- **`ci_mean`** — IC paramétrique pour la moyenne (via distribution t de Student). Suppose la normalité.
- **`ci_mean_bootstrap`** — IC par bootstrap (rééchantillonnage). Aucune hypothèse sur la distribution.
- **`ci_median_bootstrap`** — IC par bootstrap pour la médiane.
- **`ci_proportion`** — IC pour une proportion (taux de conversion, taux d'erreur).
- **`ci_variance`** — IC pour la variance (via distribution chi²).

### Bootstrap en un mot

Tu n'as pas de formule ? Pas grave. Le bootstrap simule des milliers d'échantillons
en tirant au hasard (avec remise) dans tes données, calcule la statistique à chaque fois,
et regarde la distribution des résultats.

```
  Tes données : [3, 7, 2, 9, 5]

  Échantillon bootstrap 1 : [7, 3, 3, 9, 2] → moyenne = 4.8
  Échantillon bootstrap 2 : [5, 9, 7, 7, 3] → moyenne = 6.2
  Échantillon bootstrap 3 : [2, 2, 9, 5, 3] → moyenne = 4.2
  ... ×10 000

  IC 95% = [percentile 2.5%, percentile 97.5%] des 10 000 moyennes
```

---

## 9. Étape 8 — Ajuster une distribution

### Pourquoi ajuster une distribution ?

Tu as exploré tes données (étapes 1-7). Maintenant tu veux un **modèle** :
- Pour prédire : "Quelle probabilité qu'un client dépense plus de 500€ ?"
- Pour simuler : "Si j'ajoute 1000 utilisateurs, quel sera le temps de réponse ?"
- Pour caractériser : "Mes données suivent une exponentielle de paramètre λ = 0.3"

### Le workflow

```
  1. Observer la forme (histogramme, skewness, kurtosis)
      │
      ▼
  2. Choisir une distribution candidate
      │
      ▼
  3. Estimer les paramètres (distribution_fit)
      │
      ▼
  4. Vérifier le fit (QQ-plot, KS test)
      │
      ▼
  5. Si le fit est mauvais → retour à l'étape 2
```

### KDE — Kernel Density Estimation

Avant d'ajuster un modèle paramétrique, le KDE te montre la forme réelle de tes données
en lissant l'histogramme.

```
  Histogramme            KDE (lissé)

  █                        ·
  ██   █                  ···  ·
  ███  ██                ····· ··
  ████ ███              ·········
  ████ █████          ·············
  ████████████      ····················
```

C'est comme un histogramme avec une résolution infinie. Utile pour repérer
des formes que l'histogramme cache (bimodalité par exemple).

---

## 10. Arbre de décision — Quel test utiliser ?

### "Je veux décrire mes données"

```
  Valeur typique ?
  ├── Données additives            → mean()
  ├── Données multiplicatives      → geometric_mean()
  ├── Taux / ratios                → harmonic_mean()
  ├── Avec outliers                → median()
  └── Catégorielles                → mode()

  Dispersion ?
  ├── Standard                     → standard_deviation()
  ├── Robuste aux outliers         → median_absolute_deviation()
  └── Comparer des échelles        → variance_coefficient()

  Forme ?
  ├── Asymétrie                    → skewness()
  └── Queues lourdes               → kurtosis()
```

### "Je veux vérifier si mes données sont normales"

```
  n < 5000   → shapiro_wilk()
  n ≥ 5000   → dagostino_pearson()
  Pas sûr    → normality()        (lance les deux)
  Visuel     → qqplot_data()
```

### "Je veux tester si la moyenne vaut X"

```
  Données normales ?
  ├── Oui     → ttest_1samp()
  ├── Non mais symétriques → wilcoxon_1samp()
  └── Non et asymétriques  → sign_test()

  Puis toujours : cohens_d() pour la taille d'effet
```

### "Je veux trouver les outliers"

```
  Données normales sans outliers évidents   → outliers_zscore()
  Outliers existants / données contaminées  → outliers_mad()
  Données non-normales                      → outliers_iqr()
  Preuve formelle nécessaire                → outliers_grubbs()
```

### "Je ne sais pas quel test utiliser"

```python
from polars_stats.univariate.tests import which_test

which_test(ma_serie, "normality")  # → te dit lequel utiliser et pourquoi
which_test(ma_serie, "location")   # → te guide vers le bon test
which_test(ma_serie, "outliers")   # → recommande la bonne méthode
```

---

## 11. Glossaire

| Terme | Définition |
|---|---|
| **α (alpha)** | Seuil de significativité. Souvent 0.05. Le risque qu'on accepte de se tromper en rejetant H0. |
| **Bootstrap** | Technique de rééchantillonnage : on tire des échantillons avec remise pour estimer une statistique. |
| **ddof** | Delta Degrees of Freedom. `ddof=1` → variance d'échantillon (n-1). `ddof=0` → variance de population (n). |
| **Distribution** | Modèle qui décrit les valeurs possibles et leur probabilité. |
| **Écart-type (σ, s)** | Racine carrée de la variance. Mesure la dispersion dans l'unité des données. |
| **H0** | Hypothèse nulle. Le statu quo. Ce qu'on cherche à rejeter. |
| **H1** | Hypothèse alternative. Ce qu'on veut montrer. |
| **IC** | Intervalle de confiance. Plage de valeurs plausibles pour un paramètre. |
| **IQR** | Interquartile Range. Q3 - Q1. La plage du milieu 50%. |
| **KDE** | Kernel Density Estimation. Histogramme lissé, estimation non-paramétrique de la densité. |
| **Kurtosis** | Mesure l'épaisseur des queues. = 0 pour une normale. > 0 = queues lourdes. |
| **MAD** | Median Absolute Deviation. Version robuste de l'écart-type. |
| **MLE** | Maximum Likelihood Estimation. Méthode pour estimer les paramètres d'une distribution. |
| **Normale** | Distribution en cloche, symétrique. La plus courante en stats. |
| **Outlier** | Valeur anormalement éloignée du reste des données. |
| **p-value** | Probabilité d'observer un résultat aussi extrême si H0 est vraie. Petite = on rejette H0. |
| **QQ-plot** | Graphique quantile-quantile. Compare les quantiles des données aux quantiles théoriques. |
| **Quantile** | Valeur en dessous de laquelle une proportion q des données se trouve. |
| **Skewness** | Mesure l'asymétrie. = 0 si symétrique. > 0 = queue à droite. |
| **Taille d'effet** | Mesure de l'ampleur d'un effet, indépendante de la taille de l'échantillon. |
| **Test paramétrique** | Test qui suppose une forme de distribution (souvent normale). |
| **Test non-paramétrique** | Test qui ne suppose aucune forme de distribution. Plus robuste, moins puissant. |
| **Variance** | Moyenne des carrés des écarts à la moyenne. Mesure la dispersion. |

---

*Ce guide accompagne la lib `polars_stats`. Pour la doc technique de chaque fonction, voir les docstrings dans le code source.*