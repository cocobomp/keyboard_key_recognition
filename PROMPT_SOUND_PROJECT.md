# Prompt de démarrage — branche « son » (reconnaissance d'événements sonores)

Copier-coller le bloc ci-dessous dans une **nouvelle session Claude Code** sur ce repo.
Il démarre le projet frère du projet clavier, sur sa propre branche.

---

Construis un pipeline Python de reconnaissance d'événements sonores du quotidien, en local sur macOS.

Contexte : POC de recherche perso, je suis le seul sujet, tout tourne sur ma machine, avec mon micro (MacBook, éventuellement iPhone plus tard). Objectif : savoir ce qui se passe chez moi au son. En priorité la nuit — **est-ce que je ronfle, quand, combien de temps** — et aussi en journée : chaise qui grince, stores qu'on ouvre, porte, vaisselle, clavier, etc.

Ce repo contient déjà un projet frère sur une autre branche (reconnaissance acoustique de frappes clavier). Ne le casse pas, mais réutilise ses patterns (voir §7).

## 1. Contrainte méthodologique NON NÉGOCIABLE

Le split train/test se fait **PAR SESSION D'ENREGISTREMENT** (par nuit), jamais par fenêtre aléatoire. Deux ronflements de la même nuit (même micro, même position, même état physiologique) sont quasi identiques ; un split aléatoire mesurerait de la mémorisation, pas de la reconnaissance. Entraîne sur un sous-ensemble de nuits/sessions, teste sur des nuits entièrement tenues à l'écart. Un garde-fou doit faire échouer le programme si une session apparaît dans deux splits.

## 2. Contrainte vie privée NON NÉGOCIABLE

Un enregistreur qui tourne toute la nuit capte aussi les conversations. Donc :
- Par défaut **ne pas conserver l'audio brut complet** : ne garder que les **segments d'événements** détectés (quelques secondes chacun) + les features.
- Un mode `--privacy embeddings-only` qui ne stocke que les embeddings/mel et supprime l'audio.
- Détection de la parole (les modèles AudioSet la donnent gratuitement) + une option `--drop-speech` qui écarte ces segments.
- Tout reste local, `data/` git-ignoré, aucun upload.
- Le README doit dire clairement ce qui est stocké et où.

## 3. Contrainte d'étiquetage (le cœur du projet)

Je ne dois **jamais** avoir à étiqueter des milliers de clips à la main. Le système doit :
1. **Proposer** des étiquettes tout seul dès le départ (zéro étiquette fournie par moi),
2. me laisser **confirmer / corriger** ce qu'il propose,
3. me permettre d'étiqueter **un groupe entier en un geste**,
4. ne me solliciter que sur les cas **incertains ou représentatifs** (apprentissage actif),
5. **propager** mes quelques étiquettes aux sons voisins automatiquement.

Autrement dit : semi-supervisé + apprentissage actif, pas du tout-manuel.

## 4. Pipeline attendu

### `record_audio.py` — capture longue durée
- 16 kHz mono (ce qu'attendent les modèles audio pré-entraînés), FLAC de préférence (moitié moins lourd que WAV).
- Enregistrement d'une nuit entière (8 h) découpé en fichiers de ~10 min : évite le fichier géant et survit à un crash.
- Horloge monotone commune (`time.monotonic()`) + horodatage UTC pour situer chaque événement dans la nuit.
- Gérer : empêcher la veille du Mac (`caffeinate`), reprise après erreur, vérification de l'espace disque au lancement (une nuit ≈ 0,5–1 Go), arrêt propre au `Ctrl-C` ou après `--duration`.
- Sortie par session : `data/raw/<session_id>/chunk_XXX.flac` + `meta.json` (horloge, device, durée, config).

### `detect_events.py` — segmentation NON supervisée
- Trouver « il s'est passé quelque chose » sans aucune étiquette : énergie/RMS avec seuil adaptatif sur le bruit de fond glissant (une chambre la nuit est très silencieuse, le seuil doit s'adapter), ou détection d'onsets.
- Filtres : durée min/max, fusion des événements trop proches, marge avant/après.
- Sortie : un CSV/Parquet d'événements (t_start, t_end, énergie, chemin du clip) + les clips extraits.

### `embed.py` — embeddings + étiquettes zéro-shot
**C'est la brique qui rend l'étiquetage assisté possible.** Utilise un modèle audio pré-entraîné sur AudioSet pour obtenir, par événement : (a) un embedding, (b) des **étiquettes proposées gratuitement**.
- AudioSet contient déjà les classes « Snoring », « Breathing », « Creak », « Squeak », « Door », « Speech », « Typing »… donc le système sait proposer « ronflement » **dès le premier soir**.
- Recommandé en premier : **AST** (`MIT/ast-finetuned-audioset-10-10-0.4593` via `transformers`) — c'est du PyTorch, et torch fonctionne déjà sur ma machine (Apple Silicon, device `mps`).
- Alternatives si ça coince : PANNs (CNN14), YAMNet (TensorFlow Hub — attention, TF sur Apple Silicon est plus pénible), OpenL3.
- Documente le choix et rends-le remplaçable (une fonction `embed_batch()` isolée).

### `cluster.py` — regroupement
- Clustering des embeddings (HDBSCAN de préférence, sinon k-means) → familles de sons récurrents.
- Pour chaque cluster : taille, étiquette zéro-shot dominante, exemples représentatifs (les plus proches du centroïde).
- But : pouvoir dire « tous ces 300 sons = ronflement » en un clic.

### `label_ui.py` — l'app d'étiquetage assisté
Petite app web locale (reprends le pattern `app.py` de la branche clavier : serveur HTTP simple + page autonome, ouvrable aussi depuis l'iPhone sur le même Wi-Fi).
- Affiche un cluster ou un événement avec **l'étiquette proposée**, et je fais ✅ valider / ✏️ corriger / ⏭️ passer.
- **Validation par lot** d'un cluster entier.
- **Apprentissage actif** : me présenter en priorité les cas incertains + diversifiés, pas tout le corpus.
- **Propagation** aux plus proches voisins dans l'espace d'embedding, avec possibilité de revenir dessus.
- Écoute du clip dans le navigateur + spectrogramme/waveform.
- Étiquettes sauvegardées de façon incrémentale (`data/labels.json`), **toujours corrigibles**.
- Doit être utile avec **zéro étiquette au départ** comme avec des centaines.

### `train_events.py`
- Classifieur léger **sur les embeddings** (régression logistique ou petit MLP) : quelques centaines d'exemples suffisent grâce au pré-entraînement.
- Split par session obligatoire + garde-fou anti-chevauchement.
- Gérer le fort déséquilibre de classes (le fond sonore domine tout).

### `eval_events.py`
- Sur des **nuits tenues à l'écart**.
- Métriques par classe : précision / rappel / F1, matrice de confusion.
- Métriques **temporelles** (event-based) : un épisode de ronflement détecté au bon moment, pas seulement des fenêtres isolées.
- **Baseline obligatoire** : le modèle zéro-shot pré-entraîné seul, sans mes étiquettes. Montrer ce que l'adaptation perso apporte réellement par-dessus.

### `night_report.py`
- Rapport lisible d'une nuit : % de temps passé à ronfler, timeline heure par heure, nombre d'épisodes, durée du plus long épisode, autres événements notables.
- Sortie Markdown/HTML + figure timeline.

### `predict_events.py` (étape 2, à faire seulement quand le reste marche)
- Modélisation temporelle de la séquence d'événements : anticiper le prochain événement, repérer les nuits anormales.
- Commencer simple (statistiques par heure, chaînes de Markov) avant tout modèle séquentiel.

## 5. Programme unique

Un `run.py` comme sur la branche clavier : menu / sous-commandes qui enchaînent enregistrer → détecter → embarquer → clusteriser → étiqueter → entraîner → évaluer → rapport.

## 6. Vérification sans micro

Fournis un générateur de sessions synthétiques (comme `make_synthetic_sessions.py` côté clavier) : des sons fabriqués (ronflements périodiques, grincements, silence, bruit de fond) pour valider toute la chaîne sans enregistrer. Teste le pipeline de bout en bout dessus avant de me faire enregistrer quoi que ce soit.

## 7. Réutilisation de la branche clavier

Va lire la branche du projet clavier et reprends :
- la discipline de split par session et son garde-fou (`check_disjoint` dans `kkr_common.py`),
- le pattern `Recorder` (callback audio, horloge monotone, checkpoints de frames),
- le pattern d'app web live (`app.py`) pour l'UI d'étiquetage,
- la structure `run.py`.
Ne modifie pas les fichiers du projet clavier.

## 8. Livrables

Scripts séparés, `requirements.txt`, et un README décrivant : le protocole d'enregistrement (combien de nuits viser, où poser le micro, ne pas le déplacer en cours de nuit mais varier entre les nuits, ne jamais mélanger les sessions au split), ce qui est stocké et la vie privée, et comment lire les résultats.

Viser : **≥ 4 nuits** et **≥ 3 sessions de jour**.

## 9. Attentes réalistes

Dis-moi honnêtement ce qui marchera bien et ce qui sera fragile. Le ronflement est un son fort, périodique et long → ça devrait bien marcher. Des événements rares et brefs (stores, chaise) auront très peu d'exemples → prévois-le et dis-le, ne gonfle pas les chiffres.
