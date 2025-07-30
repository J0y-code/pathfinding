import json
import os
import heapq
import re
import sys
import random
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3, NodePath
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Essayez d'importer le module astar
try:
    import astar_module
except ImportError:
    print("[Warning]: (astar_module.pyd) is 'not founded'. Utilisation of a __python__ implementation")
    # Implémentation de secours de l'algorithme A*
    def astar_implementation(start, goal, graph, point_dict):
        """Implémentation Python de l'algorithme A*"""
        if start not in graph or goal not in graph:
            return None

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float('inf') for node in graph}
        g_score[start] = 0
        f_score = {node: float('inf') for node in graph}
        f_score[start] = heuristic(point_dict[start], point_dict[goal])

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in graph[current]:
                tentative_g_score = g_score[current] + heuristic(point_dict[current], point_dict[neighbor])

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(point_dict[neighbor], point_dict[goal])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def heuristic(p1, p2):
        """Calcule la distance de Manhattan entre deux points"""
        return sum(abs(a - b) for a, b in zip(p1, p2))

    class AstarModule:
        @staticmethod
        def astar(start, goal, graph, point_dict):
            return astar_implementation(start, goal, graph, point_dict)

    astar_module = AstarModule()
    sys.modules['astar_module'] = astar_module  # Ajouter au cache des modules

# PFSParser
class PFSParser:
    def __init__(self, filename):
        self.filename = filename
        self.points = {}
        self.graph = {}

    def load(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        current_point_id = None
        current_point = {}
        in_point_block = False

        for line in lines:
            line = line.strip()
            if line.startswith("<Point>"):
                current_point_id = int(line.split()[1])
                current_point = {}
                in_point_block = True
                continue

            if in_point_block:
                if line.startswith("<x>"):
                    current_point['x'] = float(re.search(r"{(.*?)}", line).group(1))
                elif line.startswith("<y>"):
                    current_point['y'] = float(re.search(r"{(.*?)}", line).group(1))
                elif line.startswith("<z>"):
                    current_point['z'] = float(re.search(r"{(.*?)}", line).group(1))
                elif line.startswith("<Tag>"):
                    tag_match = re.search(r"\[(.*?)\]", line)
                    if tag_match:
                        current_point['tag'] = tag_match.group(1)
                elif line == "}":
                    self.points[current_point_id] = {
                        'pos': (
                            current_point.get('x', 0.0),
                            current_point.get('y', 0.0),
                            current_point.get('z', 0.0),
                        ),
                        'tag': current_point.get('tag', None)
                    }
                    self.graph[current_point_id] = []
                    current_point_id = None
                    in_point_block = False

        crossroad_id = None
        for line in lines:
            line = line.strip()
            if line.startswith("<Crossroad>"):
                crossroad_id = int(re.search(r"{(\d+)}", line).group(1))
                self.graph.setdefault(crossroad_id, [])
            elif line.startswith("<Peripheral>") and crossroad_id is not None:
                pid = int(re.search(r"{(\d+)}", line).group(1))
                if pid not in self.graph[crossroad_id]:
                    self.graph[crossroad_id].append(pid)
                if crossroad_id not in self.graph.setdefault(pid, []):
                    self.graph[pid].append(crossroad_id)

        return self.points, self.graph


# NextPointNet
class NextPointNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=1,
                 num_hidden_layers=3, activation='leakyrelu',
                 dropout_rate=0.2, batch_norm=False):
        """
        Réseau neuronal amélioré et configurable pour prédire le prochain point.

        Args:
            input_dim: dimension d'entrée (par défaut 6: pos AI + pos cible)
            hidden_dim: nombre de neurones dans les couches cachées
            output_dim: dimension de sortie (nombre de classes)
            num_hidden_layers: nombre de couches cachées
            activation: fonction d'activation ('relu', 'leakyrelu', 'tanh', etc.)
            dropout_rate: taux de dropout (0 pour désactiver)
            batch_norm: booléen pour activer BatchNorm
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = max(1, num_hidden_layers)  # Au moins une couche cachée
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm

        # Définir la fonction d'activation
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()  # valeur par défaut

        # Construire le réseau
        layers = []

        # Couche d'entrée
        layers.append(nn.Linear(input_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Couches cachées
        for _ in range(self.num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Couche de sortie
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------- Ai améliorée --------------------
class Ai:
    def __init__(self, points, graph, model=None, start_id=None):
        self.debug = False
        self.Ai_node = NodePath("Ai_n")
        self.Ai_node.reparentTo(render)
        self.model = model
        if self.model:
            self.model.reparentTo(self.Ai_node)
        self.parent = None
        self.points = points
        self.graph = graph
        self.start_id = start_id
        self.goal_id = None
        self.current_target_index = 0
        self.path = []
        self.interrupt_path = False
        self.start_pos = None
        self.spawn_pos = Vec3(self.Ai_node.getPos())
        if start_id:
            self.start_pos = self.points[self.start_id]['pos']
            self.Ai_node.setPos(*self.start_pos)
            self.spawn_pos = Vec3(self.start_pos) if self.start_pos else Vec3(0, 0, 0)

        # Ai parameters
        self.speed = 5.0
        self.aitarg = None
        self.wander_radius = 2
        self.use_spawn_as_center = True
        self.last_targets = []

        # Configuration par défaut du réseau neuronal
        self.nn_config = {
            'input_dim': 6,
            'hidden_dim': 64,
            'output_dim': 1,
            'num_hidden_layers': 2,
            'activation': 'leakyrelu',
            'dropout_rate': 0.2,
            'batch_norm': False,
            'learning_rate': 0.01,
            'weight_decay': 1e-4,
            'batch_size': 64,  # Assurez-vous que cette clé est présente
            'epochs': 500,
            'optimizer': 'adamw',
            'scheduler_step': 10,
            'scheduler_gamma': 0.9,
            'gradient_clip': 1.0
        }

    def configure_network(self, config=None, **kwargs):
        """
        Configure les paramètres du réseau neuronal.

        Peut être appelé de deux manières:
        1. ai.configure_network(hidden_dim=128, activation='relu')
        2. ai.configure_network({'hidden_dim': 128, 'activation': 'relu'})
        """
        if config is not None:
            if not isinstance(config, dict):
                raise ValueError("Le paramètre config doit être un dictionnaire")
            kwargs.update(config)

        for key, value in kwargs.items():
            if key in self.nn_config:
                # Validation des paramètres
                if key in ['hidden_dim', 'input_dim', 'output_dim',
                          'num_hidden_layers', 'batch_size', 'epochs', 'scheduler_step']:
                    if not isinstance(value, int) or value <= 0:
                        raise ValueError(f"{key} doit être un entier positif")

                elif key in ['dropout_rate', 'learning_rate', 'weight_decay',
                            'scheduler_gamma', 'gradient_clip']:
                    if not isinstance(value, (int, float)) or value < 0:
                        raise ValueError(f"{key} doit être un nombre positif")

                elif key in ['activation', 'optimizer']:
                    if not isinstance(value, str):
                        raise ValueError(f"{key} doit être une chaîne de caractères")

                elif key == 'batch_norm':
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} doit être un booléen")

                self.nn_config[key] = value
            else:
                print(f"Avertissement: paramètre inconnu {key}")

    def save_config(self, filepath="ai_config.json"):
        """Sauvegarde la configuration dans un fichier JSON."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'nn_config': self.nn_config,
                # autres paramètres à sauvegarder...
            }, f, indent=4)

    def load_config(self, filepath="ai_config.json"):
        """Charge la configuration depuis un fichier JSON."""
        import json
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config = json.load(f)
            self.configure_network(config.get('nn_config', {}))
            return True
        else:
            print(f"Fichier de configuration {filepath} non trouvé.")
            return False

    def heuristic(self, p1, p2):
        return sum(abs(a - b) for a, b in zip(p1, p2))

    def astar(self, start, goal):
        point_dict = {pid: list(data['pos']) for pid, data in self.points.items()}
        return astar_module.astar(start, goal, self.graph, point_dict)

    def set_new_goal(self, pid):
        if not self.points or pid not in self.points:
            if self.debug:
                print("Invalid point.")
            return

        current_pos = Vec3(self.Ai_node.getPos())
        closest_id = min(
            self.points,
            key=lambda i: (Vec3(*self.points[i]['pos']) - current_pos).lengthSquared()
        )

        if pid == closest_id:
            if self.debug:
                print("Goal point is the actual position.")
            return

        self.start_id = closest_id
        self.goal_id = pid
        new_path = self.astar(self.start_id, self.goal_id)

        if new_path:
            if self.debug:
                print(f"New path : {' -> '.join(map(str, new_path))}")
            self.path = new_path
            self.current_target_index = 0
            self.interrupt_path = True
        else:
            if self.debug:
                print("No path found.")
            self.path = []

    def move_along_path(self, task):
        if not self.path or self.current_target_index >= len(self.path) - 1:
            if self.debug:
                print('No path found')
            return task.cont

        if self.interrupt_path:
            self.interrupt_path = False
            self.current_target_index = 0

        pos_current = Vec3(self.Ai_node.getPos())
        pos_target = Vec3(*self.points[self.path[self.current_target_index + 1]]['pos'])
        direction = pos_target - pos_current
        distance = direction.length()

        if distance < 0.1:
            self.current_target_index += 1
        else:
            direction.normalize()
            move_step = direction * globalClock.getDt() * self.speed
            self.Ai_node.setPos(pos_current + move_step)

        return task.cont

    def check_cible_movement(self, cible):
        self.cible = cible
        global player_last_pos
        current_pos = self.cible.getPos(render)

        if (current_pos - player_last_pos).lengthSquared() > 0.0001:
            if self.debug:
                print("Le joueur a bougé.")
                print(f"[DEBUG] Player moved: {moved}")
            player_last_pos = current_pos
        return True

    def clpToobject(self, cible):
        if hasattr(cible, 'model'):
            cible = cible

        player_pos = cible.getPos(render)
        closest_id = min(
            self.points,
            key=lambda pid: (Vec3(*self.points[pid]['pos']) - player_pos).lengthSquared()
        )
        if self.debug:
            print(f"Closest point to player is ID {closest_id} at {self.points[closest_id]['pos']}")
        return closest_id

    def follow(self, task):
        if not self.aitarg:
            return task.again

        closest_to_player = self.clpToobject(self.aitarg)
        self.set_new_goal(closest_to_player)
        return task.again
    
    def set_wander_radius(self, radius):
        """Change le rayon de wandering dynamiquement."""
        self.wander_radius = radius

    def wanderer(self, high_pid=None, task=None):
        if not self.points:
            if self.debug:
                print("[WARN] No points available.")
            return task.again if task else None

        center = self.spawn_pos if self.use_spawn_as_center else Vec3(self.Ai_node.getPos())
        valid_ids = [
            pid for pid, data in self.points.items()
            if (Vec3(*data['pos']) - center).length() <= self.wander_radius
        ]

        if high_pid is not None:
            valid_ids = [pid for pid in valid_ids if pid <= high_pid]
            if not valid_ids:
                if self.debug:
                    print(f"[WARN] No valid points under ID {high_pid} in radius")
                return task.again if task else None

        candidates = [pid for pid in valid_ids if pid not in self.last_targets[-5:]]
        if not candidates:
            candidates = valid_ids

        target_id = random.choice(candidates)
        if self.debug:
            print(f"[AI] Wandering to ID {target_id} within radius {self.wander_radius} at center {center}")
        self.set_new_goal(target_id)

        self.last_targets.append(target_id)
        if len(self.last_targets) > 10:
            self.last_targets.pop(0)

        return task.again if task else None

    def _initialize_class_mappings(self):
        """Initialise les mappages entre les IDs de points et les classes."""
        if not hasattr(self, 'points') or not self.points:
            print("Erreur : Aucun point défini pour créer les mappages ID/Classe")
            return False

        # Obtenir tous les IDs de points uniques
        all_point_ids = sorted(self.points.keys())

        # Créer les mappages
        self.unique_ids = all_point_ids
        self.id_to_class = {pid: idx for idx, pid in enumerate(self.unique_ids)}
        self.class_to_id = {idx: pid for idx, pid in enumerate(self.unique_ids)}
        self.num_classes = len(self.unique_ids)

        # Mettre à jour la dimension de sortie dans la configuration
        if 'output_dim' in self.nn_config:
            self.nn_config['output_dim'] = self.num_classes

        print(f"[AI] Mappages ID/Classe initialisés pour {self.num_classes} points")
        return True

    def _verify_mappings(self):
        """Vérifie que tous les points du graphe sont couverts par les mappages."""
        if not hasattr(self, 'id_to_class') or not self.id_to_class:
            self._initialize_class_mappings()

        # Vérifier que tous les points du graphe sont couverts
        all_points = set(self.points.keys())
        mapped_points = set(self.id_to_class.keys())
        missing_points = all_points - mapped_points

        if missing_points:
            print(f"Avertissement : {len(missing_points)} points non mappés trouvés")
            self._update_class_mappings(list(missing_points))
            return False
        return True

    def _update_class_mappings(self, new_ids):
        """
        Met à jour les mappages ID/Classe avec de nouveaux IDs.
        Conserve les indices des anciens points pour maintenir la cohérence.
        """
        if not hasattr(self, 'unique_ids'):
            self.unique_ids = []
        if not hasattr(self, 'id_to_class'):
            self.id_to_class = {}
        if not hasattr(self, 'class_to_id'):
            self.class_to_id = {}

        # Ajouter les nouveaux IDs qui ne sont pas déjà présents
        existing_ids = set(self.unique_ids)
        new_ids_to_add = [pid for pid in new_ids if pid not in existing_ids]

        if new_ids_to_add:
            # Ajouter les nouveaux IDs à la fin de la liste
            self.unique_ids.extend(new_ids_to_add)
            # Mettre à jour les mappages
            start_index = len(self.unique_ids) - len(new_ids_to_add)
            for index, pid in enumerate(new_ids_to_add, start=start_index):
                self.id_to_class[pid] = index
                self.class_to_id[index] = pid

            self.num_classes = len(self.unique_ids)
            print(f"Mappages mis à jour. {len(new_ids_to_add)} nouveaux points ajoutés. Total: {self.num_classes} classes")

            # Mettre à jour la configuration du réseau
            if 'output_dim' in self.nn_config:
                self.nn_config['output_dim'] = self.num_classes
        else:
            print("Aucun nouvel ID à ajouter aux mappages")

    def _generate_training_data(self):
        """
        Génère le dataset d'entraînement complet.
        Retourne True si la génération a réussi, False sinon.
        """
        print("[AI] Début de la génération des données d'entraînement")

        # Vérifier et initialiser les mappages
        if not hasattr(self, 'id_to_class'):
            self._initialize_class_mappings()
        self._verify_mappings()

        # Préparer les structures de données
        self.X_data = []
        self.y_data = []
        self.data_gen_stats = {
            'total_attempts': 0,
            'valid_samples': 0,
            'path_length_stats': [],
            'zone_stats': {},
            'error_count': 0
        }

        # Configuration de la génération
        target_samples = 50
        batch_size = 200
        max_attempts = target_samples * 5  # Limite pour éviter une boucle infinie

        # Identifier les zones si applicable
        self.graph_zones = self._identify_zones()

        valid_ids = list(self.points.keys())
        if not valid_ids:
            print("[ERREUR] Aucun point valide trouvé dans le graphe")
            return False

        # Stratégies de génération avec leurs poids
        strategies = [
            (self._generate_random_pair, 0.3),
            (self._generate_close_pair, 0.2),
            (self._generate_far_pair, 0.2),
            (self._generate_zone_pair if self.graph_zones else None, 0.2 if self.graph_zones else 0),
            (self._generate_path_pair, 0.1)
        ]

        # Filtrer les stratégies invalides et normaliser les poids
        valid_strategies = [(s, w) for s, w in strategies if s is not None and w > 0]
        if not valid_strategies:
            print("Aucune stratégie de génération valide trouvée!")
            return False

        # Normaliser les poids
        total_weight = sum(w for _, w in valid_strategies)
        if total_weight <= 0:
            total_weight = 1
        # Normaliser les poids pour qu'ils sommement à 1
        strategies_weights = [(s, w/total_weight) for s, w in valid_strategies]

        # Génération des échantillons
        print(f"Génération de {target_samples} échantillons avec {len(valid_strategies)} stratégies...")
        while self.data_gen_stats['valid_samples'] < target_samples and self.data_gen_stats['total_attempts'] < max_attempts:
            # Sélectionner une stratégie aléatoirement pondérée
            selected_strategy = random.choices(
                [s for s, _ in strategies_weights],
                weights=[w for _, w in strategies_weights],
                k=1
            )[0]

            # Générer un échantillon avec la stratégie sélectionnée
            sample = self._generate_sample_with_strategy(selected_strategy, valid_ids)

            if sample is not None:
                ai_id, player_id, path, next_id = sample

                # Créer l'échantillon d'entrée (positions AI + positions cible)
                ai_pos = self.points[ai_id]['pos']
                player_pos = self.points[player_id]['pos']
                input_vec = list(ai_pos) + list(player_pos)

                # Ajouter à notre dataset
                self.X_data.append(input_vec)
                self.y_data.append(self.id_to_class[next_id])

                # Statistiques
                self.data_gen_stats['valid_samples'] += 1
                self.data_gen_stats['path_length_stats'].append(len(path))

                # Mettre à jour les statistiques de zone si applicable
                if self.graph_zones:
                    ai_zone = self._get_zone_for_point(ai_id)
                    player_zone = self._get_zone_for_point(player_id)
                    for zone in [ai_zone, player_zone]:
                        if zone:
                            self.data_gen_stats['zone_stats'][zone] = \
                                self.data_gen_stats['zone_stats'].get(zone, 0) + 1

            self.data_gen_stats['total_attempts'] += 1

            # Afficher le progrès toutes les 100 tentatives
            if self.data_gen_stats['total_attempts'] % 100 == 0:
                progress = self.data_gen_stats['valid_samples'] / target_samples * 100
                print(f"Progrès: {progress:.1f}% ({self.data_gen_stats['valid_samples']}/{target_samples})")

        # Vérifier que nous avons assez d'échantillons
        if self.data_gen_stats['valid_samples'] < target_samples:
            print(f"Avertissement: Seulement {self.data_gen_stats['valid_samples']} échantillons générés sur {target_samples} visés")
            if self.data_gen_stats['valid_samples'] == 0:
                print("Échec de la génération de données - aucun échantillon valide!")
                return False

        # Calculer quelques statistiques finales
        avg_path_length = sum(self.data_gen_stats['path_length_stats']) / len(self.data_gen_stats['path_length_stats']) \
            if self.data_gen_stats['path_length_stats'] else 0
        print(f"\nGénération terminée. Statistiques:")
        print(f"- Échantillons valides: {self.data_gen_stats['valid_samples']}/{self.data_gen_stats['total_attempts']}")
        print(f"- Taux de succès: {self.data_gen_stats['valid_samples']/self.data_gen_stats['total_attempts']*100:.1f}%")
        print(f"- Longueur moyenne des chemins: {avg_path_length:.1f}")
        if self.graph_zones:
            print("- Couverture des zones:")
            for zone, count in self.data_gen_stats['zone_stats'].items():
                print(f"  {zone}: {count} échantillons")
        return True

    def _generate_sample_with_strategy(self, strategy, valid_ids):
        """
        Génère un échantillon d'entraînement en utilisant la stratégie spécifiée.
        Retry plusieurs fois si nécessaire.
        """
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Appeler la stratégie pour obtenir une paire de points et un chemin
                ai_id, player_id, path = strategy(valid_ids, self.graph_zones)

                if not path or len(path) < 2:
                    continue  # Échantillon invalide, réessayer

                next_id = path[1]

                # Vérifier que tous les points du chemin sont dans nos mappages
                missing_points = [pid for pid in path if pid not in self.id_to_class]
                if missing_points:
                    self._update_class_mappings(missing_points)
                    # Après mise à jour des mappages, nous devons remettre à jour next_id
                    if next_id in self.id_to_class:
                        return ai_id, player_id, path, next_id
                    continue

                # Vérifier que next_id est toujours valide après les mises à jour
                if next_id not in self.id_to_class:
                    print(f"Erreur: next_id {next_id} toujours absent après mise à jour des mappages")
                    continue

                return ai_id, player_id, path, next_id

            except Exception as e:
                print(f"Erreur lors de la génération d'échantillon (attempt {attempt+1}): {str(e)}")
                if attempt == max_attempts - 1:
                    return None

        return None

    def _generate_random_pair(self, valid_ids, zones=None):
        """Génère une paire aléatoire de points."""
        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            ai_id = random.choice(valid_ids)
            player_id = random.choice(valid_ids)

            if ai_id != player_id:
                path = self.astar(ai_id, player_id)
                if path and len(path) >= 2:
                    return ai_id, player_id, path
            attempts += 1

        return None, None, None

    def _generate_close_pair(self, valid_ids, zones=None):
        """Génère une paire de points relativement proches."""
        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            ai_id = random.choice(valid_ids)

            # Trouver des points proches
            distances = []
            for pid in valid_ids:
                if pid == ai_id:
                    continue
                dist = self._calculate_distance(
                    self.points[ai_id]['pos'],
                    self.points[pid]['pos']
                )
                distances.append((dist, pid))

            if not distances:
                attempts += 1
                continue

            # Trier par distance et sélectionner un point dans le milieu
            distances.sort()
            middle_index = len(distances) // 2
            _, player_id = distances[middle_index]

            path = self.astar(ai_id, player_id)
            if path and len(path) >= 2:
                return ai_id, player_id, path

            attempts += 1

        return self._generate_random_pair(valid_ids, zones)

    def _generate_far_pair(self, valid_ids, zones=None):
        """Génère une paire de points relativement éloignés."""
        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            ai_id = random.choice(valid_ids)

            # Trouver des points éloignés
            distances = []
            for pid in valid_ids:
                if pid == ai_id:
                    continue
                dist = self._calculate_distance(
                    self.points[ai_id]['pos'],
                    self.points[pid]['pos']
                )
                distances.append((dist, pid))

            if not distances:
                attempts += 1
                continue

            # Trier par distance décroissante et sélectionner un point éloigné
            distances.sort(reverse=True)
            _, player_id = distances[len(distances)//4]  # Prendre un point dans le quart supérieur

            path = self.astar(ai_id, player_id)
            if path and len(path) >= 2:
                return ai_id, player_id, path

            attempts += 1

        return self._generate_random_pair(valid_ids, zones)

    def _generate_zone_pair(self, valid_ids, zones=None):
        """Génère une paire de points dans des zones différentes."""
        if not self.graph_zones or len(self.graph_zones) < 2:
            return self._generate_random_pair(valid_ids, zones)

        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            # Sélectionner deux zones différentes
            zone1, zone2 = random.sample(self.graph_zones, 2)

            # Trouver des points dans chaque zone
            zone1_points = [pid for pid in valid_ids if self._get_zone_for_point(pid) == zone1]
            zone2_points = [pid for pid in valid_ids if self._get_zone_for_point(pid) == zone2]

            if not zone1_points or not zone2_points:
                attempts += 1
                continue

            ai_id = random.choice(zone1_points)
            player_id = random.choice(zone2_points)

            path = self.astar(ai_id, player_id)
            if path and len(path) >= 2:
                return ai_id, player_id, path

            attempts += 1

        return self._generate_random_pair(valid_ids, zones)

    def _generate_path_pair(self, valid_ids, zones=None):
        """Génère une paire de points avec un chemin relativement long."""
        attempts = 0
        max_attempts = 10

        while attempts < max_attempts:
            ai_id = random.choice(valid_ids)

            # Trouver un point éloigné selon la longueur du chemin
            path_lengths = []
            for pid in valid_ids:
                if pid == ai_id:
                    continue
                path = self.astar(ai_id, pid)
                if path:
                    path_lengths.append((len(path), pid))

            if not path_lengths:
                attempts += 1
                continue

            # Trier par longueur de chemin décroissante
            path_lengths.sort(reverse=True)
            # Prendre un point parmi les 20% avec les chemins les plus longs
            candidate_index = min(len(path_lengths)//5, len(path_lengths)-1)
            _, player_id = path_lengths[candidate_index]

            path = self.astar(ai_id, player_id)
            if path and len(path) >= 2:
                return ai_id, player_id, path

            attempts += 1

        return self._generate_random_pair(valid_ids, zones)

    def _identify_zones(self):
        """Identifie les zones dans le graphe (si des tags sont disponibles ou via division spatiale)."""
        # Première tentative: utiliser des tags si disponibles
        zones = set()
        for pid, data in self.points.items():
            if 'tag' in data and data['tag']:
                zones.add(data['tag'])

        if zones:
            return sorted(zones)

        # Deuxième tentative: diviser le terrain en quadrants
        if len(self.points) > 100:  # Seulement pour les grands graphes
            try:
                # Collecter toutes les positions
                all_x = []
                all_y = []
                for data in self.points.values():
                    pos = data['pos']
                    if len(pos) >= 2:  # S'assurer que nous avons x et y
                        all_x.append(pos[0])
                        all_y.append(pos[1])

                if all_x and all_y:
                    min_x, max_x = min(all_x), max(all_x)
                    min_y, max_y = min(all_y), max(all_y)

                    # Diviser en 4 quadrants si la zone est suffisamment grande
                    if (max_x - min_x) > 50 and (max_y - min_y) > 50:
                        mid_x = (min_x + max_x) / 2
                        mid_y = (min_y + max_y) / 2

                        # Créer des zones et les assigner aux points
                        zone_names = [
                            "Quadrant_SO", "Quadrant_NO",
                            "Quadrant_SE", "Quadrant_NE"
                        ]
                        for pid, data in self.points.items():
                            x, y, *_ = data['pos']
                            if x <= mid_x and y <= mid_y:
                                zone = zone_names[0]
                            elif x <= mid_x and y > mid_y:
                                zone = zone_names[1]
                            elif x > mid_x and y <= mid_y:
                                zone = zone_names[2]
                            else:
                                zone = zone_names[3]
                            data['zone'] = zone  # Ajouter un attribut zone

                        return zone_names

            except Exception as e:
                print(f"Erreur lors de la création des zones: {e}")

        # Si aucune zone n'est identifiée
        return None

    def _get_zone_for_point(self, pid):
        """Retourne la zone d'un point donné."""
        if pid not in self.points:
            return None

        data = self.points[pid]
        if 'zone' in data:
            return data['zone']
        elif 'tag' in data:
            return data['tag']
        return None

    def _calculate_distance(self, p1, p2):
        """Calcule la distance euclidienne entre deux points."""
        return sum((a - b)**2 for a, b in zip(p1, p2))**0.5

    def _verify_data_integrity(self):
        """
        Vérifie l'intégrité des données générées:
        - Tous les points référencés existent
        - Toutes les classes sont valides
        - Aucune donnée corrompue
        """
        if not hasattr(self, 'X_data') or not hasattr(self, 'y_data'):
            return False

        print("[AI] Vérification de l'intégrité des données...")

        # Vérifier que tous les points référencés existent
        valid = True
        invalid_count = 0

        # Pour X_data, chaque échantillon est une liste concatenant deux positions
        # Pour y_data, chaque valeur est un indice de classe

        # Vérifier que tous les indices de classe dans y_data sont valides
        max_class_index = self.num_classes - 1
        y_data_valid = all(0 <= y <= max_class_index for y in self.y_data)

        if not y_data_valid:
            print("Erreur: Des indices de classe invalides dans y_data")
            valid = False

        # Vérifier que toutes les classes référencées existent dans notre mappage
        missing_classes = [y for y in self.y_data if y not in self.class_to_id]
        if missing_classes:
            print(f"Erreur: {len(missing_classes)} indices de classe invalides (max {max_class_index})")
            valid = False

        if valid:
            print("[AI] Vérification de l'intégrité des données réussie")
        else:
            print("[ERREUR] Problèmes d'intégrité des données détectés")

        return valid

    def _prepare_training(self):
        """
        Prépare tout ce qui est nécessaire pour l'entraînement:
        - Convertit les données en tenseurs
        - Initialise le modèle
        - Configure l'optimiseur et la fonction de perte
        """
        print("[AI] Préparation des outils d'entraînement...")

        # Vérifier l'intégrité des données
        if not self._verify_data_integrity():
            print("[ERREUR] Problèmes avec les données d'entraînement")
            return False

        # Convertir les données en tenseurs
        try:
            X_tensor = torch.tensor(self.X_data, dtype=torch.float32)
            y_tensor = torch.tensor(self.y_data, dtype=torch.long)

            # Mettre à jour l'état d'entraînement
            self._train_state['X_data'] = X_tensor
            self._train_state['y_data'] = y_tensor
            self._train_state['data_size'] = len(X_tensor)
        except Exception as e:
            print(f"Erreur lors de la conversion des données en tenseurs: {e}")
            return False

        # Créer le modèle
        try:
            self.next_point_to_aitarg_model = NextPointNet(
                input_dim=self.nn_config['input_dim'],
                hidden_dim=self.nn_config['hidden_dim'],
                output_dim=self.nn_config['output_dim'],
                num_hidden_layers=self.nn_config['num_hidden_layers'],
                activation=self.nn_config['activation'],
                dropout_rate=self.nn_config['dropout_rate'],
                batch_norm=self.nn_config['batch_norm']
            )
            self._train_state['model_created'] = True
        except Exception as e:
            print(f"Erreur lors de la création du modèle: {e}")
            return False

        # Configurer l'optimiseur
        try:
            params = self.next_point_to_aitarg_model.parameters()
            lr = self.nn_config['learning_rate']
            wd = self.nn_config['weight_decay']

            if self.nn_config['optimizer'].lower() == 'adam':
                optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
            elif self.nn_config['optimizer'].lower() == 'adamw':
                optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
            elif self.nn_config['optimizer'].lower() == 'sgd':
                optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
            else:
                optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

            self._train_state['optimizer'] = optimizer
        except Exception as e:
            print(f"Erreur lors de la configuration de l'optimiseur: {e}")
            return False

        # Configurer la fonction de perte (avec pondération si nécessaire)
        try:
            # Calculer les poids de classe si nécessaire
            if hasattr(self, 'y_data') and self.y_data:
                y_array = np.array(self.y_data)
                unique_classes, counts = np.unique(y_array, return_counts=True)

                if len(unique_classes) > 1:
                    # Calculer les poids inverses
                    class_weights = 1. / counts
                    class_weights = class_weights / np.sum(class_weights)  # Normaliser
                    class_weights = torch.tensor(class_weights, dtype=torch.float32)

                    # Créer un dictionnaire pour vérifier les indices de classe
                    max_class_index = max(unique_classes)
                    if max_class_index >= len(class_weights):
                        # Ajuster si nécessaire (si des classes sont manquantes)
                        class_weights = torch.ones(max_class_index + 1, dtype=torch.float32)
                        # Mais dans notre cas, les indices de classe devraient correspondre

                    self._train_state['criterion'] = torch.nn.CrossEntropyLoss(weight=class_weights)
                else:
                    self._train_state['criterion'] = torch.nn.CrossEntropyLoss()
        except Exception as e:
            print(f"Erreur lors de la configuration de la fonction de perte: {e}")
            self._train_state['criterion'] = torch.nn.CrossEntropyLoss()

        # Configurer le scheduler
        try:
            self._train_state['scheduler'] = torch.optim.lr_scheduler.StepLR(
                self._train_state['optimizer'],
                step_size=self.nn_config['scheduler_step'],
                gamma=self.nn_config['scheduler_gamma']
            )
        except Exception as e:
            print(f"Avertissement: Impossible de configurer le scheduler: {e}")
            self._train_state['scheduler'] = None

        print("[AI] Préparation terminée. Tout est prêt pour l'entraînement.")
        return True

    def _train_epoch(self):
        """
        Exécute une époque complète d'entraînement.
        Retourne la perte moyenne pour l'époque.
        """
        if not self._train_state.get('model_created', False):
            print("[ERREUR] Le modèle n'est pas prêt pour l'entraînement")
            return None
        model = self.next_point_to_aitarg_model
        criterion = self._train_state['criterion']
        optimizer = self._train_state['optimizer']
        scheduler = self._train_state.get('scheduler', None)
        batch_size = self._train_state.get('batch_size', 64)  # Utiliser une valeur par défaut si nécessaire
        data_size = self._train_state['data_size']
        X_data = self._train_state['X_data']
        y_data = self._train_state['y_data']
        # Mode entraînement
        model.train()
        total_loss = 0.0
        num_batches = 0
        # Mélanger les données au début de chaque époque
        indices = torch.randperm(data_size)
        X_shuffled = X_data[indices]
        y_shuffled = y_data[indices]
        # Entraînement par batches
        for start_idx in range(0, data_size, batch_size):
            end_idx = min(start_idx + batch_size, data_size)
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            # Clipper les gradients si configuré
            if 'gradient_clip' in self.nn_config and self.nn_config['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.nn_config['gradient_clip']
                )
            optimizer.step()
            # Statistiques
            total_loss += loss.item() * (end_idx - start_idx)
            num_batches += 1
            # Afficher le progrès occasionnellement
            if start_idx % (batch_size * 10) == 0:  # Toutes les 10 batches
                current_loss = total_loss / (end_idx) if end_idx > 0 else 0
                progress = (end_idx / data_size) * 100
                print(f"Progrès: {progress:.1f}% - Perte actuelle: {current_loss:.4f}")
        # Calculer la perte moyenne pour l'époque
        epoch_loss = total_loss / data_size if data_size > 0 else 0
        # Exécuter le scheduler
        if scheduler:
            scheduler.step()
        return epoch_loss


    def _train_model(self, epochs):
        """
        Entraîne le modèle pour un nombre donné d'époques.
        Gère l'arrêt anticipé et enregistre les statistiques.
        """
        print(f"\n[AI] Début de l'entraînement du modèle pour {epochs} époques")

        best_loss = float('inf')
        patience = 5  # Nombre d'époques sans amélioration avant arrêt
        epochs_without_improvement = 0
        history = {
            'epoch_loss': [],
            'val_loss': []  # Pourrait être utilisé pour la validation (non implémenté ici)
        }

        # Boucle d'entraînement
        for epoch in range(epochs):
            epoch_start_time = time.time() if 'time' in sys.modules else 0

            # Entraînement d'une époque
            epoch_loss = self._train_epoch()

            if epoch_loss is None:
                print("Erreur lors de l'entraînement de l'époque")
                break

            # Enregistrer les statistiques
            history['epoch_loss'].append(epoch_loss)

            # Afficher les statistiques de l'époque
            if isinstance(epoch_loss, (int, float)):
                epoch_time = time.time() - epoch_start_time if 'time' in sys.modules else 0
                print(f"Époque {epoch+1}/{epochs} - Perte: {epoch_loss:.4f}"
                    f"{f' - Temps: {epoch_time:.2f}s' if epoch_time else ''}")

                # Vérifier l'amélioration pour l'arrêt anticipé
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_without_improvement = 0
                    # Sauvegarder le modèle si c'est le meilleur jusqu'à présent
                    self.save_model()
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Arrêt anticipé après {epochs_without_improvement} époques sans amélioration")
                        break

        print("\nEntraînement terminé")
        return history

    def train(self, task, epochs=None):
        """
        Méthode principale pour l'entraînement du modèle.
        Gère la génération des données et l'entraînement du modèle.
        """
        # Initialiser si nécessaire
        if not hasattr(self, '_train_state'):
            if not self.aitarg:
                print("[AI Train] Aucun aitarg défini.")
                return task.done
            # Nombre d'époques par défaut
            epochs = epochs if epochs is not None else self.nn_config.get('epochs', 50)
            # Initialiser les mappages ID/Classe
            self._initialize_class_mappings()
            # Préparer les structures de données
            self._train_state = {
                'epoch': 0,
                'epochs': epochs,
                'data_generated': False,
                'training_phase': 'init',  # init -> data_gen -> training -> complete
                'batch_size': self.nn_config.get('batch_size', 64)  # Ajouter batch_size ici
            }
        # Étape 1: Génération des données si pas encore faite
        if not self._train_state.get('data_generated', False):
            print("\n[Phase 1/3] Génération des données d'entraînement")
            success = self._generate_training_data()
            if not success:
                print("[ERREUR] Échec de la génération des données")
                return task.done
            print("\n[Phase 2/3] Préparation de l'entraînement")
            if not self._prepare_training():
                print("[ERREUR] Échec de la préparation de l'entraînement")
                return task.done
            self._train_state['data_generated'] = True
            self._train_state['training_phase'] = 'ready_to_train'
            return task.cont  # Continuer pour passer à l'entraînement
        # Étape 2: Entraînement du modèle
        if self._train_state['training_phase'] == 'ready_to_train':
            print("\n[Phase 3/3] Entraînement du modèle")
            history = self._train_model(self._train_state['epochs'])
            self._train_state['training_phase'] = 'complete'
            self._train_state['training_history'] = history
            return task.done  # Fin de l'entraînement
        return task.cont  # Par défaut, continuer (bien que normalement nous devrions être terminés)


    def save_model(self, filepath="next_point_model.pth"):
        """Sauvegarde le modèle entraîné avec sa configuration."""
        if not hasattr(self, 'next_point_to_aitarg_model'):
            print("[ERREUR] Aucun modèle à sauvegarder")
            return False

        try:
            checkpoint = {
                'model_state_dict': self.next_point_to_aitarg_model.state_dict(),
                'id_to_class': self.id_to_class,
                'class_to_id': self.class_to_id,
                'nn_config': self.nn_config,
                'optimizer_state_dict': self._train_state['optimizer'].state_dict() if hasattr(self, '_train_state') and 'optimizer' in self._train_state else None,
                'epoch': self._train_state.get('epoch', 0),
                'loss_history': self._train_state.get('epoch_stats', []),
                'training_history': self._train_state.get('training_history', {})
            }
            torch.save(checkpoint, filepath)
            print(f"[AI] Modèle sauvegardé dans {filepath}")
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du modèle: {e}")
            return False

    def load_model(self, filepath="next_point_model.pth"):
        """Charge un modèle sauvegardé."""
        if not os.path.exists(filepath):
            print(f"Fichier modèle introuvable: {filepath}")
            return False

        try:
            checkpoint = torch.load(filepath)

            # Charger la configuration
            if 'nn_config' in checkpoint:
                self.nn_config.update(checkpoint['nn_config'])

            # Mettre à jour les mappages ID/Classe
            if 'id_to_class' in checkpoint:
                self.id_to_class = checkpoint['id_to_class']
                self.class_to_id = checkpoint.get('class_to_id', {v: k for k, v in self.id_to_class.items()})
                self.unique_ids = list(sorted(self.id_to_class.keys()))
                self.num_classes = len(self.unique_ids)
                print(f"Mappages ID/Classe chargés avec {self.num_classes} classes")

            # Créer le modèle
            self.next_point_to_aitarg_model = NextPointNet(
                input_dim=self.nn_config.get('input_dim', 6),
                hidden_dim=self.nn_config.get('hidden_dim', 64),
                output_dim=self.nn_config.get('output_dim', self.num_classes),
                num_hidden_layers=self.nn_config.get('num_hidden_layers', 2),
                activation=self.nn_config.get('activation', 'leakyrelu'),
                dropout_rate=self.nn_config.get('dropout_rate', 0.2),
                batch_norm=self.nn_config.get('batch_norm', False)
            )

            # Charger les poids du modèle
            self.next_point_to_aitarg_model.load_state_dict(checkpoint['model_state_dict'])

            # Charger l'état de l'optimiseur si présent
            if 'optimizer_state_dict' in checkpoint and hasattr(self, '_train_state'):
                if 'optimizer' in self._train_state:
                    self._train_state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

            # Charger l'historique si présent
            if 'training_history' in checkpoint:
                if not hasattr(self, '_train_state'):
                    self._train_state = {}
                self._train_state['training_history'] = checkpoint['training_history']

            print(f"Modèle et configuration chargés depuis {filepath}")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False


    def predict(self):
        if not self.aitarg or not hasattr(self, 'next_point_to_aitarg_model'):
            if self.debug:
                print("[ML] No aitarg or model not available.")
            return None
        ai_pos = self.Ai_node.getPos(render)
        targ_pos = self.aitarg.getPos(render)
        # Créer le vecteur d'entrée [x_ai, y_ai, z_ai, x_target, y_target, z_target]
        input_tensor = torch.tensor([[ai_pos.x, ai_pos.y, ai_pos.z,
                                    targ_pos.x, targ_pos.y, targ_pos.z]],
                                    dtype=torch.float32)
        with torch.no_grad():
            logits = self.next_point_to_aitarg_model(input_tensor)
            predicted_class = torch.argmax(logits, dim=1).item()
        predicted_id = self.class_to_id.get(predicted_class, None)
        if self.debug:
            print(f"[ML] Predicted class: {predicted_class}, point ID: {predicted_id}")
        return predicted_id if predicted_id in self.points else None

    def ML_seek(self, task):
        """
        Panda3D task: Predict the next point to move towards the current aitarg using the trained model,
        and set it as the new goal. This is similar to follow, but uses only the ML prediction.
        """
        if not self.aitarg:
            return task.again
        predicted_next_id = self.predict()
        if predicted_next_id in self.points:
            self.set_new_goal(predicted_next_id)
            if self.debug:
                print(f"[AI] ML predicted next point: {predicted_next_id}")
        else:
            if self.debug:
                print("[AI] ML prediction invalid, no move.")
        return task.again

# Exemple d'utilisation avec la nouvelle configuration
if __name__ == "__main__":
    env = PFSParser("level1.pfs")
    points, graph = env.load()
    app = ShowBase()
    model = loader.loadModel('playertest.egg')
    ai = Ai(points, graph, model, start_id=240)

    # Configuration avancée du réseau neuronal
    ai.configure_network(
        hidden_dim=128,
        num_hidden_layers=3,
        activation='leakyrelu',
        dropout_rate=0.3,
        batch_norm=True,
        learning_rate=0.001,
        epochs=200,
        batch_size=64,
        optimizer='adamw',
        scheduler_step=10,
        scheduler_gamma=0.85,
        gradient_clip=1.0
    )

    # Sauvegarder la configuration pour une utilisation future
    ai.save_config("ai_config.json")

    # Créer une cible
    ai.aitarg = loader.loadModel('target.egg')
    ai.aitarg.setPos(10, 10, 0)
    ai.aitarg.reparentTo(render)

    # Configurez d'autres paramètres de l'IA
    ai.speed = 5
    ai.debug = True
    ai.wander_radius = 25
    ai.use_spawn_as_center = False

    # Démarrer les tâches
    app.taskMgr.add(ai.move_along_path, "MoveTask")
    app.taskMgr.add(lambda task: ai.train(task), "TrainTask")

    # Planifier le passage à ML_seek après l'entraînement
    def start_ml_seek(task):
        if hasattr(ai, 'next_point_to_aitarg_model'):
            app.taskMgr.add(ai.ML_seek, "MLSeekTask")
        return task.done

    app.taskMgr.do_method_later(20, start_ml_seek, "StartMLSeekTask")

    app.run()
