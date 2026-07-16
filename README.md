# face-rec

Reconnaissance faciale sur une collection d'images. Deux temps:

1. **`load`** indexe une collection: pour chaque image, détecte tous les visages,
   calcule un embedding (InsightFace / ArcFace, vecteur 512-D) et stocke tout dans
   SQLite (recherche vectorielle via `sqlite-vec`).
2. **`group`** prend une image requête (dans ou hors collection), détecte ses
   visages, en sélectionne un, puis retourne toutes les images de la collection
   contenant la même personne, triées par similarité.

Une image peut contenir plusieurs personnes: l'unité indexée est le **visage**, pas
l'image.

## Précision

Le modèle par défaut est `buffalo_l` (détecteur RetinaFace-10GF + reconnaissance
ResNet50@WebFace600K, embedding ArcFace 512-D), l'état de l'art open source. La
décision "même personne" repose sur la **similarité cosinus** entre embeddings, pas
sur des mesures géométriques du visage.

La pose (yaw/pitch/roll) de chaque visage est stockée: fiable en frontal à
trois-quarts (< 30°), utilisable jusqu'à ~45°, non fiable au-delà. Pour un usage
type vérification d'identité, privilégier des images frontales et relever le seuil.

## Installation

Nécessite Python 3.12 (les wheels InsightFace/onnxruntime ne couvrent pas encore
3.13+). `uv` gère la version automatiquement.

```bash
make sync              # installe les dépendances dans .venv
make install           # installe la commande face-rec globalement (uv tool)
```

Au premier `load`, InsightFace télécharge le pack de modèles (~330 Mo) dans
`~/.cache/models/insightface`.

## Usage

```bash
# 1. Indexer une collection (récursif)
face-rec load ~/Photos/album

# 2. Trouver les images contenant une personne
face-rec group requete.jpg                      # 1 visage: direct
face-rec group requete.jpg                      # N visages: table + choix interactif
face-rec group requete.jpg --face 0             # choisir un visage par index, sans prompt
face-rec group requete.jpg --coords 320,240     # choisir le visage le plus proche du pixel
face-rec group requete.jpg --threshold 0.5      # seuil de similarité (défaut 0.40)
face-rec group requete.jpg --json               # sortie JSON (chemins + similarité + pose)
face-rec group requete.jpg --no-forcing         # ignorer les liens manuels force-group

# 3. Lier manuellement des images (profil + face non reconnus automatiquement)
face-rec force-group face.jpg profil.jpg        # 2+ images, 1 visage chacune, = même personne

# 4. Réécrire les chemins en base (après déplacement de la collection)
face-rec replace-path '^/old/' '/new/' --dry-run   # prévisualiser
face-rec replace-path '^/old/' '/new/'             # appliquer

# 5. État de la base
face-rec info
```

### Liens manuels (force-group)

La reconnaissance échoue sur les visages de profil vs face (angle > 45°). Pour ces
cas, on valide manuellement l'identité: `force-group` déclare que plusieurs images
montrent la même personne, indépendamment du seuil. Chaque image doit contenir
**exactement un visage**.

Les liens sont:
- **transitifs**: `force-group a b` puis `force-group b c` ⇒ a, b, c sont la même
  personne (groupes fusionnés par composantes connexes).
- **propagés à travers la reconnaissance**: si `group` reconnaît `a` et que `a` est
  lié à `profil`, alors `profil` remonte aussi, marqué `forced`.
- **persistants**: stockés par chemin de fichier, survivent à un re-`load`.

`face-rec group` combine par défaut reconnaissance + liens forcés. `--no-forcing`
restreint aux résultats de reconnaissance pure.

### Déplacer la collection (replace-path)

Comme le chemin est la clé (embeddings et liens forcés), déplacer/renommer les
fichiers sur le disque impose de réécrire les chemins en base. `replace-path`
applique une substitution regex Python (`re.sub`) à tous les chemins d'images et de
liens. `<repl>` supporte les backreferences (`\1`, `\g<name>`). Toujours
prévisualiser avec `--dry-run`. Les collisions (deux chemins fusionnés, ou cible
déjà existante) sont détectées et refusées.

### Options clés

| Option | Rôle |
|--------|------|
| `--db PATH` | Chemin de la base SQLite (défaut `faces.db`) |
| `--model NAME` | Pack de modèles InsightFace (défaut `buffalo_l`) |
| `--threshold F` | Seuil de similarité cosinus 0..1 (défaut 0.40) |
| `--coords X,Y` | Sélection du visage par pixel (pas de prompt) |
| `--face N` | Sélection du visage par index (pas de prompt) |
| `--reindex` | Ré-indexe même les fichiers inchangés |
| `--no-forcing` | (group) ignore les liens manuels force-group |
| `--dry-run` | (replace-path) prévisualise sans écrire |

Les chemins sont stockés **résolus (absolus)** au `load` et au `force-group`, pour
qu'ils coïncident (sur macOS `/tmp` devient `/private/tmp`).

### Changer de modèle

Le nom du modèle est stocké par embedding. Les embeddings de deux modèles ne sont
pas comparables: `group` ne compare qu'entre embeddings du même modèle. Changer de
`--model` implique de ré-indexer (ou d'indexer sous un autre tag dans la même base).

## Développement

```bash
make check     # lint + format-check + typecheck + security + tests+coverage
make test      # tests seuls
```

## Structure

Voir `CLAUDE.md`.
