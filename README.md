# Reconnaissance acoustique de frappes clavier — POC local (macOS)

Pipeline Python pour mesurer **si un CNN peut reconstituer ce que je tape à
partir du son de mon clavier**, en local, sur mes propres enregistrements.

Le résultat visé n'est pas « un gros chiffre » : c'est **l'écart entre la
précision sur de la prose anglaise et la précision sur des chaînes aléatoires**.
Cet écart mesure la part du score qui vient de la redondance de la langue et non
de l'acoustique. Le chiffre sur l'aléatoire est le seul qui décrit vraiment ce
que le micro discrimine.

> Portée : POC de recherche personnelle, un seul sujet (moi), mon matériel, mes
> frappes, tout en local. Rien ici ne capture le clavier d'un tiers : `capture.py`
> ne fonctionne que sur la machine où il tourne, avec la permission
> Accessibilité accordée explicitement, et affiche à l'écran le texte à taper.

---

## 0. Démarrage rapide : un seul programme

Sur le Mac, une fois les dépendances installées (§2) et les permissions
accordées :

```bash
python run.py            # menu guidé
python run.py all        # enregistre les sessions → entraîne → évalue → démo live
```

`run.py` enchaîne tout : il te fait enregistrer plusieurs sessions (en te
rappelant de déplacer le micro entre elles), construit le split **par session**
automatiquement, lance `preprocess → train → eval → baseline`, puis ouvre la
**démo live** où tu tapes et vois en temps réel ce que le modèle croit que tu
écris. Sous-commandes utiles :

```bash
python run.py collect --sessions 6 --keys 800   # enregistrer seulement
python run.py pipeline                            # (ré)entraîner + évaluer sur les données existantes
python run.py live                                # démo temps réel avec le modèle entraîné
```

Les sections suivantes détaillent chaque étape et ce qu'il faut savoir pour que
la mesure soit honnête. Le mode live est décrit au §6bis.

---

## 1. Contrainte méthodologique : split PAR SESSION

Deux appuis de la même touche dans **une même session** sont quasi identiques
(même micro, même position, même pièce, même posture). Un split aléatoire par
fenêtre mettrait des quasi-doublons des deux côtés de la barrière et mesurerait
de la **mémorisation**, pas de la reconnaissance — typiquement 20 à 40 points de
précision fantômes.

Donc :

* on entraîne sur un sous-ensemble de **sessions entières** ;
* on sélectionne le modèle sur une (ou des) session de **validation**, elle aussi
  entière et disjointe ;
* on teste sur des sessions **entièrement tenues à l'écart**, enregistrées à un
  autre moment, avec le micro légèrement déplacé.

`train.py` et `eval.py` refusent de démarrer (`SystemExit`) si un `session_id`
apparaît dans deux splits. Il n'existe aucune option de split aléatoire dans ce
dépôt, volontairement.

## 2. Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

macOS, deux permissions à accorder **à l'application terminal** (Terminal,
iTerm2, VS Code…), pas à `python` :

* **Réglages Système › Confidentialité et sécurité › Accessibilité** — obligatoire
  pour le hook clavier global de `pynput`. Sans elle, `capture.py` enregistre
  l'audio et zéro frappe (il vous prévient).
* **Micro** — demandé au premier lancement.

Après avoir modifié une permission, relancer le terminal.

## 3. Protocole de collecte

Objectif : **≥ 5000 frappes étiquetées sur ≥ 4 sessions**. En pratique, viser
**6 à 8 sessions de ~800 frappes** (10–15 min chacune) : il en faut assez pour
garder 2 sessions de test (une prose, une aléatoire) et 1 de validation, sans
ruiner le jeu d'entraînement.

Plan type (8 sessions, 4 prose / 4 aléatoire) :

| jour | session | mode | position micro |
|---|---|---|---|
| 1 | `prose_…_micA` | prose | A |
| 1 | `random_…_micA` | random | A |
| 2 | `prose_…_micB` | prose | B (déplacé de ~10 cm) |
| 2 | `random_…_micB` | random | B |
| 3 | `prose_…_micC` | prose | C (angle différent) |
| 3 | `random_…_micC` | random | C |
| 4 | `prose_…_micD` | prose | D |
| 4 | `random_…_micD` | random | D |

Règles :

* **Déplacer légèrement le micro entre les sessions** (quelques centimètres, un
  angle différent), et enregistrer à des moments différents. C'est ce qui rend le
  test honnête : si le modèle ne survit pas à un déplacement du micro, il n'a
  appris qu'un canal, pas un clavier.
* Garder le **même clavier et la même posture de frappe** : c'est la
  discrimination des touches qu'on mesure, pas le changement de matériel.
* Taper à son rythme naturel, **ne pas corriger les fautes** : chaque keydown
  physique est étiqueté, une faute de frappe reste une donnée valide.
* Noter la position du micro dans `--tag` — elle finit dans le `session_id`.
* Alterner les deux modes sur des sessions **séparées** (une session = un mode),
  sinon la comparaison prose/aléatoire se fait à conditions acoustiques
  inégales.
* **Ne jamais mélanger les sessions au split.** Une session est atomique : soit
  train, soit val, soit test, jamais deux.

```bash
python capture.py --mode prose  --tag micA --target-keys 800
python capture.py --mode random --tag micA --target-keys 800
```

Chaque run produit `data/raw/<session_id>/` : `audio.wav` (48 kHz mono),
`keys.csv` (keydown, `timestamp,key,session_id,mode`), `events.csv` (tous les
événements, `timestamp,event,key,session_id,mode`) et `meta.json`.

### Enregistrement libre : `record.py`

`capture.py` affiche un texte à taper (utile pour équilibrer les touches et
séparer proprement prose/aléatoire). `record.py` fait l'inverse : **aucun
prompt**, il enregistre le micro et **tous les événements clavier** (down *et*
up) pendant que vous tapez ce que vous voulez, jusqu'à `Ctrl-C`.

```bash
python record.py                      # frappe libre, mode=free, jusqu'à Ctrl-C
python record.py --duration 600        # arrêt auto après 10 min
python record.py --mode prose --tag micB  # étiqueter une session libre
```

Même discipline d'horloge et même bip de sync que `capture.py`, et même format
de sortie : `keys.csv` alimente directement `preprocess.py`, `events.csv`
conserve la séquence complète down/up (utile pour les temps de maintien et
l'inter-frappe). Les sessions `mode=free` ne comptent pas dans la comparaison
prose/aléatoire de `eval.py` — pour l'expérience, garder `capture.py` ; `record.py`
sert à capturer de la frappe réelle « au fil de l'eau ».

> **Vie privée** : `record.py` est un enregistreur de frappe **global** — il logue
> **tout** ce que vous tapez pendant qu'il tourne (mots de passe, autres fenêtres
> comprises) dans des fichiers en clair. À n'utiliser que sur votre machine, pour
> vos propres frappes. `data/` est git-ignoré ; évitez de taper des secrets
> pendant un enregistrement.

### Synchronisation audio / frappes

Les deux flux sont datés avec **la même horloge monotone**
(`time.monotonic()`) : aucune sensibilité aux ajustements d'horloge murale
(NTP, veille, changement d'heure). Côté audio, l'ancrage vient des timestamps
ADC de PortAudio convertis une fois dans le référentiel monotone, puis
re-checkpointés ~2×/s (`clock_checkpoints` dans `meta.json`) — ce qui rend le
mapping robuste aux buffers perdus en cours de session.

Filet de sécurité : un **bip de sync 1 kHz** est joué et logué au début de
chaque session. `preprocess.py` le retrouve dans l'enregistrement et affiche le
décalage résiduel. Attendu : quelques millisecondes. Au-delà de ~25 ms il
prévient ; `--apply-sync-correction` applique alors le décalage mesuré (à
n'utiliser que s'il est franchement grand, la chaîne haut-parleur→micro coûtant
elle-même ~10 ms).

## 4. Prétraitement

```bash
python preprocess.py            # data/raw -> data/processed
```

Pour chaque keydown : fenêtre de **250 ms** (50 ms avant l'appui, 200 ms après —
elle couvre le transitoire d'appui *et* le relâchement), puis mel-spectrogramme
(`n_mels=64`, `n_fft=1024`, `hop=128` → 64×94), en dB, z-scoré par fenêtre.
Sortie : `data/processed/<session_id>.npz` (+ `index.json`).

Réglages utiles : `--pre-ms/--post-ms`, `--n-mels`, `--keys`
(`letters+space` par défaut ; `printable`, `all`, ou une liste
`a,b,c,space`).

> Note d'honnêteté : les fenêtres sont placées grâce aux **timestamps système**.
> On contourne donc volontairement le problème de segmentation aveugle qu'aurait
> un vrai attaquant. Ce POC mesure « l'acoustique distingue-t-elle les touches »,
> pas « peut-on retrouver les frappes dans un enregistrement non étiqueté ».

## 5. Entraînement

```bash
python train.py \
  --train-sessions prose_…_micA random_…_micA prose_…_micB random_…_micB \
  --val-sessions   prose_…_micC \
  --test-sessions  prose_…_micD random_…_micD \
  --epochs 60
```

ou avec un fichier de split (recommandé, il documente l'expérience) :

```json
{ "train": ["…micA", "…micA_r", "…micB", "…micB_r"],
  "val":   ["…micC"],
  "test":  ["…micD", "…micD_r"] }
```

```bash
python train.py --split-file splits.json --epochs 60
```

CNN : 3 blocs (conv-BN-ReLU ×2 + maxpool, largeurs 32/64/128) puis tête dense
128 → classes, ~0.5 M paramètres. AdamW + cosine, augmentation légère
(décalage temporel ±16 ms, jitter de gain, masques fréquence/temps). Les
statistiques de normalisation sont calculées **sur le train seulement** et
stockées dans le checkpoint. Le meilleur modèle (top-1 validation) est
sauvegardé dans `models/keycnn.pt`, la courbe dans `models/keycnn_trainlog.csv`.

Les chiffres de validation servent **uniquement** à choisir l'époque. Ce ne sont
pas les résultats de l'expérience.

### Baseline k-NN sur MFCC

`baseline_knn.py` fournit un point de comparaison simple et non paramétrique : le
CNN ne « vaut » que ce qu'il gagne au-dessus de ce baseline. Chaque fenêtre est
résumée par un descripteur **MFCC** (moyenne + écart-type des MFCC et de leurs
deltas dans le temps), puis classée par plus proches voisins.

```bash
python baseline_knn.py --split-file splits.json --results-dir results
```

Les MFCC sont la DCT des log-mel déjà calculés par `preprocess.py` : le baseline
voit donc **exactement les mêmes fenêtres, la même config de features et le même
split par session** que le CNN — la comparaison est honnête. `k` est réglé sur la
session de validation (jamais sur le test), les features sont standardisées sur le
train seulement, et il n'y a là non plus aucun split aléatoire.

Sortie : `results/baseline_knn.json` (top-1/top-5 prose vs aléatoire, par
session). Si `results/eval_results.json` du CNN est présent dans le même dossier,
un tableau **CNN vs k-NN** est affiché directement :

```
regime       CNN top-1   kNN top-1   CNN top-5   kNN top-5
prose            …%          …%          …%          …%
random           …%          …%          …%          …%
```

Le chiffre qui compte reste l'aléatoire : c'est là que se lit ce que chaque modèle
extrait vraiment de l'acoustique, sans béquille linguistique.

## 6. Évaluation — le cœur de l'expérience

```bash
python eval.py --checkpoint models/keycnn.pt          # sessions de test du checkpoint
python fetch_lm_corpus.py                             # texte public pour le n-gram
python eval.py --checkpoint models/keycnn.pt --lm-corpus corpus/lm_train.txt
```

Sortie (`results/`) :

* **top-1 et top-5 par caractère, séparément pour la prose et pour l'aléatoire**,
  plus l'écart `prose − aléatoire` en points — le résultat central ;
* détail par session ;
* **matrices de confusion** (`confusion_{all,prose,random}.{png,csv}`) et une
  analyse de voisinage physique : proportion d'erreurs tombant sur une touche
  adjacente du QWERTY vs le taux attendu par hasard, et distance moyenne des
  erreurs en « unités-touche » ;
* liste des confusions les plus fréquentes ;
* `eval_results.json` + `report.md`.

Décodage par modèle de langue (optionnel) : n-gram caractère d'ordre 5 interpolé,
recherche par faisceau sur chaque salve de frappe (une pause > 2 s coupe la
salve). Le poids du LM est réglé **sur la session de validation**, jamais sur le
test (`--lm-weight auto`). Le même décodeur est appliqué aux deux régimes : il
gonfle la prose et s'effondre sur l'aléatoire — c'est exactement la démonstration
recherchée, un « 90 % » obtenu ainsi parle de l'anglais, pas du clavier.
`eval.py` vérifie en plus que les lignes affichées pendant les sessions de test
ne se retrouvent pas telles quelles dans le corpus du LM (sinon on mesure de la
récupération de texte mémorisé).

### Comment lire les résultats

* `top-1 aléatoire` ≈ hasard (1/27 ≈ 3.7 %) → le micro/les features ne portent
  rien d'exploitable dans ces conditions.
* `top-1 aléatoire` nettement au-dessus du hasard, avec des erreurs concentrées
  sur les touches physiquement voisines → il y a bien un signal acoustique, et il
  est spatial.
* `top-1 prose` ≫ `top-1 aléatoire` → l'écart est de la statistique de l'anglais.
  À ne jamais présenter comme une performance acoustique.

## 6bis. Démo live — voir ce que le modèle « entend »

```bash
python live.py --checkpoint models/keycnn.pt     # ou: python run.py live
```

Charge le modèle entraîné, écoute le micro et le clavier en même temps, et pour
chaque frappe découpe la **même fenêtre de 250 ms** que le pipeline, la passe au
CNN et affiche sa prédiction à côté de la touche réellement tapée :

```
you h    model h    ✓  [h g j b n]  run top1  61.2% top5  88.4%
you e    model e    ✓  [e r w s d]  run top1  61.8% top5  88.9%
you l    model i    ✗  [i l k o p]  run top1  60.9% top5  89.1%
```

À l'arrêt (`Ctrl-C`), il imprime la transcription complète — ce que tu as tapé vs
ce que le modèle a « entendu » — et la précision top-1/top-5.

Points importants :

* C'est la **démo acoustique honnête** : la fenêtre est placée par le timestamp
  système de la frappe (pas de segmentation aveugle), et **aucun modèle de langue**
  n'intervient — tu vois le classifieur acoustique seul décider, touche par touche.
* Tes frappes live forment une **session inédite**, jamais vue à l'entraînement :
  c'est un vrai test de généralisation, pas de mémorisation. Attends-toi à des
  chiffres proches de l'aléatoire tenu à l'écart, pas des chiffres « prose + LM ».
* Le chemin de features live est **identique** à `preprocess.py` (mêmes mel,
  même normalisation par fenêtre puis par bin mel du checkpoint) : la démo mesure
  bien le même modèle que `eval.py`.
* Garde le **même clavier** et une **position de micro proche** de l'entraînement ;
  un micro très différent fait chuter la démo (le modèle a appris un canal en
  plus des touches).

## 6ter. App web live (et pont vers l'iPhone)

`app.py` sert la même démo que `live.py`, mais dans une **page web** — plus jolie,
et ouvrable **sur l'iPhone** (même Wi-Fi) pour mettre le résultat sur le téléphone
sans app native.

```bash
python app.py --demo                       # aperçu de l'interface, sans micro ni modèle
python app.py --checkpoint models/keycnn.pt   # temps réel (mic + clavier)
```

Au lancement il affiche deux URL :
- `http://localhost:8000` sur le Mac,
- `http://<ip-du-mac>:8000` à ouvrir dans Safari sur l'iPhone.

Le modèle tourne **sur le Mac** ; l'iPhone n'est qu'un écran. C'est le pont rapide.

### Vers une vraie app iPhone native

Ce qui se transfère vers iOS, c'est **le modèle**, pas l'interface :
- exporter le CNN vers **Core ML** (`coremltools`) pour le faire tourner on-device,
- ré-implémenter le calcul du mel-spectrogramme côté Swift (Accelerate) **à
  l'identique** de `preprocess.py`, ou le replier dans le graphe du modèle,
- **ré-entraîner avec le micro de l'iPhone** (chaque micro a sa signature, un
  modèle « micro du Mac » transfère mal),
- gérer la **segmentation aveugle** (le téléphone n'a pas les événements clavier).

Les points 3 et 4 sont les vrais chantiers de la version mobile.

## 7. Test à blanc sans micro

Pour vérifier que la chaîne tourne (features → CNN → éval) sans rien enregistrer :

```bash
python make_synthetic_sessions.py --out-dir data/synthetic --sessions 6
python preprocess.py --raw-dir data/synthetic --out-dir data/proc_syn
python train.py --processed-dir data/proc_syn \
  --train-sessions syn00_prose syn01_random syn02_prose \
  --val-sessions syn03_random --test-sessions syn04_prose syn05_random \
  --out models/syn.pt --epochs 12
python eval.py --checkpoint models/syn.pt --processed-dir data/proc_syn \
  --results-dir results/syn
```

L'audio y est fabriqué (résonances par touche + coloration par session). Les
scores obtenus ne disent **rien** de vrais claviers : c'est un test de plomberie.
Si on ajoute `--lm-corpus corpus/prose_en.txt` sur ces données synthétiques, le
LM est entraîné sur le texte même qui a été « tapé » : la prose grimpe
artificiellement (~99 %) pendant que l'aléatoire chute. C'est la démonstration en
miniature de ce que le garde-fou anti-fuite du corpus sert à éviter en vrai.

## 8. Fichiers

| fichier | rôle |
|---|---|
| `run.py` | **programme unique** : enregistre → entraîne → évalue → démo live (menu ou sous-commandes) |
| `capture.py` | enregistrement micro + clavier avec prompt, horloge monotone commune, bip de sync ; classes partagées (Recorder, KeyLogger, écriture des fichiers) |
| `record.py` | enregistrement libre : micro + **tous** les événements clavier (down/up), sans prompt |
| `preprocess.py` | fenêtres 250 ms + mel-spectrogrammes, vérification du bip |
| `train.py` | CNN, split par session, meilleur checkpoint |
| `baseline_knn.py` | baseline k-NN sur MFCC, comparable au CNN (mêmes fenêtres, même split) |
| `eval.py` | prose vs aléatoire, confusion, voisinage physique, décodage n-gram |
| `live.py` | démo temps réel (terminal) : tape et vois ce que le modèle croit que tu écris |
| `app.py` | **app web live** : même démo dans le navigateur, ouvrable aussi sur l'iPhone (même Wi-Fi) |
| `kkr_common.py` | labels canoniques, layout physique, I/O session, mapping d'horloge |
| `make_synthetic_sessions.py` | données factices pour tester la chaîne |
| `fetch_lm_corpus.py` | texte public (Gutenberg) pour le modèle de langue |
| `corpus/prose_en.txt` | texte affiché en mode prose |

## 9. Limites

* Un seul sujet, un seul clavier, une seule pièce : rien ici ne généralise à une
  autre machine ou à quelqu'un d'autre.
* Fenêtrage par timestamps système : la segmentation aveugle n'est pas traitée.
* Les touches sont repliées sur leur forme physique (`shift+a` → `a`) ; le
  décodage complet d'un texte casé n'est pas l'objet.
* Le CNN voit une frappe isolée, sans contexte de digramme ni timing entre
  frappes — deux sources d'information supplémentaires volontairement écartées
  pour que la mesure reste acoustique.
