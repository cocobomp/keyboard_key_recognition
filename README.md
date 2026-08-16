<!-- Branche « sons du quotidien ». Convention du repo : une branche = un
     projet. Le projet frère (reconnaissance acoustique de frappes clavier)
     vit sur claude/keystroke-acoustic-recognition-uzw4z5 ; cette branche en
     réutilise les patterns (split par session, Recorder, app web, run.py)
     sans toucher à ses fichiers. -->

# Reconnaissance d'événements sonores du quotidien

POC de recherche perso, 100 % local sur macOS : savoir ce qui se passe chez
moi au son. En priorité la nuit — **est-ce que je ronfle, quand, combien de
temps** — et aussi en journée : chaise qui grince, stores, porte, vaisselle,
clavier…

Un seul sujet (moi), un seul micro (MacBook, iPhone possible pour l'UI),
aucune donnée ne quitte la machine.

## La chaîne

```
record_audio.py   capture 16 kHz mono FLAC, chunks de 10 min, caffeinate
      │
detect_events.py  segmentation NON supervisée (seuil adaptatif sur le bruit
      │           de fond glissant) -> clips de quelques secondes
      │
embed.py          embeddings + étiquettes zéro-shot AudioSet (AST)
      │
cluster.py        familles de sons récurrents (HDBSCAN)
      │
label_ui.py       étiquetage ASSISTÉ dans le navigateur (iPhone ok) :
      │           valider un cluster entier en un geste, apprentissage
      │           actif, propagation kNN — jamais des milliers de clips
      │
train_events.py   classifieur léger sur embeddings, split PAR SESSION
      │
eval_events.py    nuits tenues à l'écart + baseline zéro-shot obligatoire
      │
night_report.py   % de ronflement, épisodes, timeline heure par heure
      │
predict_events.py (étape 2) stats horaires, chaîne de Markov, nuits atypiques
```

Tout s'enchaîne avec `python run.py` (menu) ou `python run.py <étape>`.

## Démarrage — valider SANS micro d'abord

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # torch/transformers optionnels au début

# chaîne complète sur sessions synthétiques (aucun enregistrement) :
python make_synthetic_sessions.py --short
python run.py all --raw-dir data/synthetic --backend mel --simulate-labels
```

Si `results/report.md` et `results/night_synnight03.md` sortent, la
plomberie est bonne. Ensuite seulement, enregistrer en vrai :

```bash
python run.py record          # une nuit (8 h par défaut, Ctrl-C ok)
python run.py pipeline        # detect -> embed -> cluster -> ... -> rapport
python run.py label           # l'UI d'étiquetage (aussi depuis l'iPhone)
```

## Protocole d'enregistrement

- **Viser ≥ 4 nuits et ≥ 3 sessions de jour.** En dessous, le split
  train/val/test par session n'a pas assez de matière et les chiffres ne
  veulent rien dire.
- **Ne jamais déplacer le micro pendant une session.** Le varier ENTRE les
  sessions (chevet, commode, autre coin de la chambre) est au contraire
  souhaitable : le modèle apprend le son, pas la position du micro.
- Toujours utiliser `--tag` pour noter la position (`--tag chevet`).
- La nuit : Mac branché sur secteur, `record_audio.py` lance `caffeinate`
  tout seul pour empêcher la veille. Une nuit ≈ 0,3–0,5 Go en FLAC 16 kHz
  (l'espace disque est vérifié au lancement).
- **Ne JAMAIS mélanger les sessions dans le split.** Deux ronflements de la
  même nuit (même micro, même position, même état) sont quasi identiques :
  un split aléatoire par fenêtre mesurerait de la mémorisation, pas de la
  reconnaissance. Le split est construit par `run.py split` et un garde-fou
  (`check_disjoint`) fait échouer train/eval si une session apparaît dans
  deux splits.

## Vie privée — ce qui est stocké, et où

Un enregistreur qui tourne la nuit capte aussi les conversations. Règles :

| Donnée | Où | Contenu | Suppression |
|---|---|---|---|
| Audio brut complet | `data/raw/<session>/chunk_*.flac` | TOUT, y compris la parole | `detect_events.py --delete-raw` (auto après extraction) |
| Clips d'événements | `data/events/<session>/clips/*.flac` | quelques secondes par événement détecté | `embed.py --privacy embeddings-only` |
| Embeddings + temps | `data/embeddings/*.npz`, `data/events/*/events.csv` | vecteurs + horodatages, pas d'audio | `rm -r data/` |
| Labels | `data/labels.json` | mes étiquettes | éditable/révocable dans l'UI |
| Modèle, résultats | `models/`, `results/` | classifieur, rapports | `rm -r models results` |

- Par défaut, seul l'audio **des événements détectés** est conservé au-delà
  de la détection : lancer `detect_events.py --delete-raw` supprime les
  chunks bruts (recommandé dès que la détection d'une session est validée).
- `embed.py --privacy embeddings-only` supprime aussi les clips : il ne
  reste que les embeddings et les horodatages (l'UI affiche alors les
  spectrogrammes… s'ils ont été mis en cache avant, sinon rien d'audible).
- `embed.py --drop-speech` écarte les événements que le modèle AudioSet
  classe comme parole (score > 0,5) et supprime leurs clips.
- `data/`, `models/`, `results/`, `splits.json` sont git-ignorés. Rien
  n'est uploadé nulle part ; l'UI web n'écoute que sur le réseau local.

## Étiqueter sans se ruiner la vie

`label_ui.py` (ou `run.py label`) ouvre une page locale, aussi accessible
depuis l'iPhone sur le même Wi-Fi (l'URL est affichée au lancement) :

1. le système **propose** une étiquette (zéro-shot AudioSet : « Snoring »
   → ronflement, dès le premier soir, sans aucun label fourni) ;
2. ✅ valider / ✏️ corriger / ⏭️ passer — un **cluster entier** se valide en
   un geste (« ces 300 sons = ronflement ») ;
3. l'**apprentissage actif** ne présente que les cas incertains ou
   représentatifs (marge du classifieur quand un modèle existe) ;
4. chaque label est **propagé** aux voisins dans l'espace d'embedding
   (provenance `propagated`, toujours révocable d'un clic) ;
5. tout est sauvé au fil de l'eau dans `data/labels.json`.

Sur le synthétique, ~12 gestes couvrent ~98 % du corpus — c'est l'ordre de
grandeur attendu en vrai pour les sons fréquents.

## Lire les résultats

- `results/report.md` — précision/rappel/F1 par classe **sur les nuits de
  test uniquement**, en deux colonnes : baseline zéro-shot seule vs modèle
  adapté à mes labels (la différence est ce que l'adaptation apporte), plus
  les métriques **par épisode** (un épisode de ronflement compté juste s'il
  est détecté au bon moment, IoU ≥ 0.3 ou début à ±2 s).
- `results/night_<session>.md/.png` — le rapport d'une nuit : % de temps à
  ronfler, nombre d'épisodes, plus long épisode, timeline heure par heure.
- `results/confusion.png` — qui est confondu avec qui.
- `python run.py predict` — stats horaires, transitions de Markov, nuits
  atypiques (étape 2, à prendre avec des pincettes).

## Attentes réalistes (honnêtes)

Ce qui devrait bien marcher :
- **Le ronflement.** Son fort, long, périodique, des centaines d'exemplaires
  par nuit, et AudioSet a une classe « Snoring » dédiée : le zéro-shot le
  repère dès le premier soir et l'adaptation affine. C'est le cas d'usage
  taillé pour cette chaîne.
- La détection d'événements elle-même (une chambre la nuit est très
  silencieuse, le seuil adaptatif a la vie facile).

Ce qui sera fragile :
- **Les événements rares et brefs** (stores, chaise, porte) : quelques
  exemplaires par semaine → F1 par classe très bruité, clusters qui
  fusionnent porte/vaisselle. Prévoir plusieurs semaines de données avant
  d'avoir des chiffres stables, et ne pas sur-interpréter avant.
- Les scores obtenus sur les sessions **synthétiques** valident la
  plomberie, pas les performances réelles.
- La distinction ronflement / respiration forte / frottement de draps est
  intrinsèquement floue — attendre des confusions entre ces classes.
- `--drop-speech` dépend du détecteur AudioSet : bon sur de la parole
  franche, moins sur des murmures. Ne pas le considérer comme une garantie.

## Fichiers

| Script | Rôle |
|---|---|
| `sed_common.py` | constantes, sessions, temps, `check_disjoint`, labels |
| `record_audio.py` | capture chunkée FLAC (pattern Recorder du projet clavier) |
| `detect_events.py` | segmentation non supervisée à seuil adaptatif |
| `embed.py` | embeddings AST/mel + zéro-shot (`embed_batch()` remplaçable) |
| `cluster.py` | HDBSCAN/k-means + résumé par cluster |
| `label_ui.py` | app web d'étiquetage assisté (pattern app.py du clavier) |
| `train_events.py` | classifieur sur embeddings, split par session |
| `eval_events.py` | éval nuits tenues à l'écart + baseline zéro-shot |
| `night_report.py` | rapport d'une nuit + timeline |
| `predict_events.py` | étape 2 : stats horaires, Markov, nuits atypiques |
| `make_synthetic_sessions.py` | sessions de synthèse pour valider sans micro |
| `simulate_labeling.py` | rejoue l'étiquetage depuis la vérité terrain (test) |
| `run.py` | menu / sous-commandes qui enchaînent tout |
