# Pipeline détaillé

## Les 3 étapes (InsightFace buffalo_l)

1. **Détection**: RetinaFace-10GF (`det_10g.onnx`) → bbox + 5 landmarks + det_score.
2. **Alignement + attributs**: 2d106/3d68 landmarks → pose (yaw/pitch/roll),
   genderage (non utilisé ici).
3. **Reconnaissance**: ResNet50@WebFace600K (`w600k_r50.onnx`) → embedding 512-D
   (ArcFace). L2-normalisé dans `engine._normalize` pour que le produit scalaire
   égale la similarité cosinus.

`FaceEngine.analyze_path` lit l'image via `cv2.imread` (BGR) et retourne une liste
de `DetectedFace` (frozen dataclass: bbox, pose, det_score, embedding).

## Pose (ordre InsightFace)

`face.pose` est `[pitch, yaw, roll]` en degrés. On mappe explicitement dans
`engine.analyze_image`. Interprétation:
- yaw = rotation gauche/droite (le paramètre critique)
- pitch = haut/bas
- roll = inclinaison (tête penchée)

Fiabilité embedding selon |yaw|: < 30° bon, 30-45° dégradé, > 45° non fiable. La
pose est **stockée mais pas filtrée** au load (choix utilisateur): filtrer à la
requête si besoin en lisant les colonnes yaw/pitch/roll de la table `faces`.

## Stockage (database.py)

- `images(id, path UNIQUE, mtime)`: dédup par mtime → `image_is_current` évite de
  ré-indexer un fichier inchangé.
- `faces(id, image_id, model_name, x1..y2, yaw/pitch/roll, det_score)`.
- `vec_faces` (sqlite-vec vec0): `face_id PK, embedding FLOAT[512]
  distance_metric=cosine`.

### Recherche KNN

```sql
SELECT ... FROM vec_faces v JOIN faces f JOIN images i
WHERE v.embedding MATCH ? AND k = ? AND f.model_name = ?
ORDER BY v.distance
```

sqlite-vec renvoie la **distance cosinus** (0 = identique). On convertit
`similarity = 1 - distance` et on filtre `>= threshold`. `_dedupe_by_image` garde
le meilleur visage par image (une image multi-personnes ne remonte qu'une fois pour
la personne cherchée).

**k obligatoire**: sqlite-vec exige `k = N` dans la clause WHERE pour un KNN.
`search(limit=None)` (défaut) pose `k = COUNT(*) vec_faces`, soit tous les visages,
puis filtre par seuil côté Python. Pas de cap fixe (l'ancien `limit=200` tronquait
silencieusement une personne présente dans >200 images). `k` doit couvrir toute la
table car le filtre `model_name` est appliqué APRÈS le KNN.

**--limit (group)**: plafonne les matchs de *reconnaissance* (meilleurs d'abord).
Défaut illimité (`DEFAULT_LIMIT=None`, même pattern que `DEFAULT_THRESHOLD`). Le
forcing se propage à travers l'ensemble reconnu COMPLET (avant cap), et les matchs
forcés ne sont jamais tronqués par `--limit`.

## Sélection du visage requête (service.select_face)

- `--coords X,Y`: visage dont la box contient le point, sinon le plus proche du
  centre; jamais de prompt.
- 1 seul visage, pas de coords: pris directement.
- N visages, pas de coords: retourne None → la CLI affiche la table et demande.
- `--face N`: index explicite, court-circuite tout.

## Gotchas

- **stdout pollué**: InsightFace fait des `print()` au chargement et à l'inférence.
  `_silence_stdout()` redirige stdout→stderr pendant `prepare` et `get`, sinon
  `group --json` produit du JSON invalide. Ne jamais retirer ce garde.
- **Cache modèle**: InsightFace lit l'argument `root=` (pas `INSIGHTFACE_HOME`).
  On passe `root=~/.cache/models/insightface`; les modèles vont dans
  `<root>/models/buffalo_l/`. Au 1er run, download ~330 Mo.
- **Python 3.12 imposé**: pas de wheels InsightFace/onnxruntime pour 3.13+ à ce
  jour. `requires-python = ">=3.12,<3.13"`, uv provisionne 3.12 automatiquement.
- **numpy < 2.0**: onnxruntime/insightface ne suivent pas encore numpy 2 partout.
- **Embeddings non comparables entre modèles**: chaque recherche filtre sur
  `model_name`. Deux modèles = deux espaces vectoriels distincts.
- **Seuil**: 0.40 par défaut (permissif). Pour de la vérification d'identité,
  monter vers 0.5-0.6 pour écraser les faux positifs (accepter un imposteur = pire
  cas). À calibrer sur tes propres données.

## Forced links (force-group)

But: lier manuellement des images non reconnues automatiquement (profil vs face,
angle > 45°). Modèle: on force des **personnes**, pas des paires. Choix design:

- **Stockage**: table `forced_links(path_a, path_b)`, clé = chemin (survit au
  re-load). `add_forced_clique` insère toutes les paires (clique) avec ordre
  normalisé lo<hi et INSERT OR IGNORE (idempotent).
- **Groupes**: `service._connected_components` fait un union-find sur les arêtes à
  la requête. Deux cliques qui partagent une image fusionnent. Transitif.
- **1 visage exigé**: `force_group` refuse une image à 0 ou N visages
  (ForceGroupError). Désambiguïsation nette: 1 image = 1 personne.
- **Propagation** (`find_matches`, use_forcing=True): reconnaissance normale → set
  A. Toute composante forcée qui touche la requête OU un membre de A ajoute tous ses
  chemins. Les chemins forcés hors A deviennent des MatchRow(forced=True,
  similarity=1.0, bbox/pose depuis `face_meta_for_path` ou None si non indexé).
  C'est l'intérêt: forcer {face, profil}, reconnaître face → profil remonte aussi.
- **--no-forcing**: court-circuite, résultats reconnaissance purs.

## replace-path (réécriture de chemins)

Le chemin étant la clé, déplacer la collection casse tout → il faut réécrire.

- **Plan** (`plan_path_replace`): `re.compile` + `re.sub(pattern, repl, path)` sur
  tous les chemins de `images` et `forced_links`. `<repl>` = backrefs Python
  standard. Retourne RewritePlan(changes, unchanged).
- **Collisions**: deux chemins distincts qui donnent la même cible, ou une cible qui
  existe déjà → ReplacePathError (refuse, ne corrompt pas UNIQUE(path)).
- **Application** (`apply_path_rewrites`): UPDATE images + forced_links.path_a +
  path_b dans une seule transaction (`with self._conn`), rollback sur erreur.
- **--dry-run**: affiche la table old→new, n'écrit rien.
- Regex invalide → ReplacePathError à la compilation.

## Cohérence des chemins (gotcha critique)

`load` ET `force_group` stockent `Path.resolve()` (absolu). Sinon, sur macOS,
`/tmp/x` (load) et `/private/tmp/x` (resolve dans force-group) divergent et la
propagation échoue silencieusement. Toute nouvelle entrée de chemin DOIT resolve().

## Tests

- `test_database.py`: round-trip réel sqlite-vec (add, search exact ~1.0, seuil,
  isolation par modèle).
- `test_service.py`: `select_face` (coords/single/multi) et `_dedupe_by_image`.
- `test_cli.py`: flux load/group/info avec `FaceEngine` mocké (FakeEngine), donc
  aucun download modèle. sqlite-vec reste réel.
- `test_forcing.py`: union-find, clique, ordre normalisé, face_meta_for_path.
- `test_forcing_service.py`: propagation à travers reconnaissance, --no-forcing,
  isolation des groupes non liés.
- `test_replace_path.py`: plan, backrefs, application images+liens, no-op, regex
  invalide, collisions (fusion + cible existante).
- `test_cli.py`: force-group (1 visage / multi rejet), group forced vs --no-forcing,
  replace-path dry-run puis apply, info forced_links.
- Non couvert: internes `engine.py` (nécessite le vrai modèle, validé
  manuellement, dont force-group + propagation + replace-path end-to-end) et
  branches prompt/table interactives.
