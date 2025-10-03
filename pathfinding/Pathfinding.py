from panda3d.core import NodePath, Vec3, Point3, LineSegs
import re
import math
import numpy as np
import heapq
import random

def build_neighbor_distances(graph, points):
    """
    Pré-calcul des distances entre chaque point et ses voisins.
    Retourne un dict {point_id: {neighbor_id: distance}}.
    """
    # Convertir toutes les positions en array numpy
    positions = {pid: np.array(data['pos'], dtype=np.float32) for pid, data in points.items()}

    neighbor_distances = {}
    for pid, neighbors in graph.items():
        neighbor_distances[pid] = {}
        pos1 = positions[pid]
        for n in neighbors:
            pos2 = positions[n]
            neighbor_distances[pid][n] = np.linalg.norm(pos1 - pos2)

    return neighbor_distances


def astar_precomputed(start, goal, graph, points, neighbor_distances):
    """
    A* sur un graphe avec distances pré-calculées.
    Retourne le chemin sous forme de liste de point IDs.
    """
    if start not in graph or goal not in graph:
        return None

    positions = {pid: np.array(points[pid]['pos'], dtype=np.float32) for pid in points}

    open_heap = [(0, start)]
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score = {node: float('inf') for node in graph}
    f_score[start] = np.linalg.norm(positions[start] - positions[goal])

    open_set = {start}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        open_set.discard(current)

        if current == goal:
            # Reconstruction du chemin
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in graph[current]:
            tentative_g = g_score[current] + neighbor_distances[current][neighbor]
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + np.linalg.norm(positions[neighbor] - positions[goal])
                f_score[neighbor] = f
                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f, neighbor))
                    open_set.add(neighbor)

    return None



# ================================
# Construire graphe hiérarchique des subareas
# ================================
def build_subarea_graph(points, areas, graph):
    """
    Crée un graphe où chaque subarea est un nœud,
    et deux subareas sont connectées si elles partagent des points voisins.
    """
    point_to_subarea = {pid: data['subarea'] for pid, data in points.items()}
    subarea_graph = {}

    # Initialisation
    for area_name, subareas in areas.items():
        for sub_name in subareas.keys():
            subarea_graph[sub_name] = set()

    # Parcours des connexions pour relier les subareas
    for pid, neighbors in graph.items():
        sa1 = point_to_subarea[pid]
        for n in neighbors:
            sa2 = point_to_subarea[n]
            if sa1 != sa2:
                subarea_graph[sa1].add(sa2)
                subarea_graph[sa2].add(sa1)

    return {k: list(v) for k, v in subarea_graph.items()}


# ================================
# A* hiérarchique
# ================================
def astar_hierarchical(start_pid, goal_pid, points, graph, neighbor_distances, areas, point_to_subarea):
    """
    1. Chemin entre subareas
    2. Chemins locaux à l'intérieur de chaque subarea
    """
    start_sa = point_to_subarea[start_pid]
    goal_sa = point_to_subarea[goal_pid]

    # Si start_sa == goal_sa, pas besoin de hiérarchie
    if start_sa == goal_sa:
        return astar_precomputed(start_pid, goal_pid, graph, points, neighbor_distances)

    # 1️ Construire graphe hiérarchique
    subarea_graph = build_subarea_graph(points, areas, graph)

    # 2️ A* sur subareas (graphe réduit)
    open_heap = [(0, start_sa)]
    came_from = {}
    g_score = {sa: float('inf') for sa in subarea_graph}
    g_score[start_sa] = 0
    f_score = {sa: float('inf') for sa in subarea_graph}
    f_score[start_sa] = 0

    while open_heap:
        _, current_sa = heapq.heappop(open_heap)
        if current_sa == goal_sa:
            # reconstruction chemin subarea
            sa_path = [current_sa]
            while current_sa in came_from:
                current_sa = came_from[current_sa]
                sa_path.append(current_sa)
            sa_path.reverse()
            break
        for neighbor_sa in subarea_graph[current_sa]:
            tentative_g = g_score[current_sa] + 1
            if tentative_g < g_score[neighbor_sa]:
                came_from[neighbor_sa] = current_sa
                g_score[neighbor_sa] = tentative_g
                f_score[neighbor_sa] = tentative_g
                heapq.heappush(open_heap, (f_score[neighbor_sa], neighbor_sa))
    else:
        return None  # pas de chemin entre subareas

    # 3️ Chemins locaux
    full_path = []
    current_point = start_pid
    for idx, sa in enumerate(sa_path):
        # Points candidats dans cette subarea
        candidate_points = [pid for pid, psa in point_to_subarea.items() if psa == sa]

        # Si c’est la dernière subarea, on vise goal_pid
        target_points = [goal_pid] if sa == goal_sa else candidate_points

        # Choisir le point le plus proche de current_point
        best_target = min(target_points, key=lambda p: np.linalg.norm(
            np.array(points[p]['pos']) - np.array(points[current_point]['pos'])
        ))

        # Chemin A* local
        local_path = astar_precomputed(current_point, best_target, graph, points, neighbor_distances)
        if not local_path:
            return None
        if full_path:
            local_path = local_path[1:]  # éviter de répéter le dernier point
        full_path.extend(local_path)
        current_point = best_target

    return full_path



# PFSParser
class PFSParser:
    def __init__(self, filename):
        self.filename = filename
        self.areas = {}   # {area_name: {subarea_name: [point_ids]}}
        self.points = {}  # {id: {'pos': (x,y,z), 'tag': ..., 'area': ..., 'subarea': ...}}
        self.graph = {}   # {id: [neighbor_ids]}

    def load(self):
        with open(self.filename, 'r', encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        current_area = None
        current_subarea = None
        current_point_id = None
        current_point = {}
        in_point_block = False

        # --- Parsing principal ---
        for line in lines:
            line = line.strip()

            # --- Détection des zones ---
            if line.startswith("<Area>"):
                # Exemple : <Area> [Plan.002] {
                match = re.search(r"\[(.*?)\]", line)
                if match:
                    current_area = match.group(1)
                    self.areas.setdefault(current_area, {})
                continue

            if line.startswith("<SubArea>"):
                match = re.search(r"\[(.*?)\]", line)
                if match:
                    current_subarea = match.group(1)
                    self.areas[current_area].setdefault(current_subarea, [])
                continue

            # --- Détection d’un Point ---
            if line.startswith("<Point>"):
                match = re.search(r"<Point>\s+(\d+)", line)
                if match:
                    current_point_id = int(match.group(1))
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
                    # Fin du point
                    self.points[current_point_id] = {
                        'pos': (
                            current_point.get('x', 0.0),
                            current_point.get('y', 0.0),
                            current_point.get('z', 0.0),
                        ),
                        'tag': current_point.get('tag', None),
                        'area': current_area,
                        'subarea': current_subarea
                    }
                    self.graph[current_point_id] = []
                    if current_area and current_subarea:
                        self.areas[current_area][current_subarea].append(current_point_id)

                    current_point_id = None
                    in_point_block = False

        # --- Parsing des connexions ---
        crossroad_id = None
        in_connections = False

        for line in lines:
            line = line.strip()

            if line.startswith("<Relate>"):
                in_connections = True
                continue
            if in_connections and line == "}":
                in_connections = False
                continue

            if in_connections:
                if line.startswith("<Crossroad>"):
                    crossroad_id = int(re.search(r"{(\d+)}", line).group(1))
                    self.graph.setdefault(crossroad_id, [])
                elif line.startswith("<Peripheral>") and crossroad_id is not None:
                    pid = int(re.search(r"{(\d+)}", line).group(1))
                    # Ajouter l’arête bidirectionnelle
                    if pid not in self.graph[crossroad_id]:
                        self.graph[crossroad_id].append(pid)
                    if crossroad_id not in self.graph.setdefault(pid, []):
                        self.graph[pid].append(crossroad_id)

        return self.areas, self.points, self.graph



from scipy.spatial import KDTree
import time
# ================================
# Classe AI
# ================================
class Ai:
    def __init__(self, points, graph, areas=None, model=None, start_id=None, world=None, use_bullet=True):
        self.use_bullet = use_bullet
        self.world = world

        # Données du graphe et subareas
        self.points = points
        self.graph = graph
        self.areas = areas or {}
        self.point_to_subarea = {pid: data['subarea'] for pid, data in points.items()}

        # KDTree pour snap rapide
        self.point_ids = list(points.keys())
        self.positions_array = np.array([points[pid]['pos'] for pid in self.point_ids], dtype=np.float32)
        self.kdtree = KDTree(self.positions_array)

        # Node & model
        self.Ai_node = NodePath("Ai_n")
        self.Ai_node.reparentTo(render)
        self.model = model
        if self.model:
            self.model.reparentTo(self.Ai_node)

        self.start_id = start_id
        if start_id:
            self.Ai_node.setPos(*self.points[start_id]['pos'])

        self.goal_id = None
        self.path = []
        self.current_target_index = 0
        self.interrupt_path = False

        # Paramètres
        self.speed = 5.0
        self.aitarg = None
        self.wander_radius = 50
        self.debug = False
        self.debug_lines = []

        # Pré-calcul distances
        self.neighbor_distances = build_neighbor_distances(graph, points)

        # Graphe hiérarchique subarea (pré-calculé)
        self.subarea_graph = build_subarea_graph(points, self.areas, graph)

        # Cooldown recalcul chemin
        self._last_path_time = 0
        self._path_cooldown = 1.0

    # -------------------------
    # Snap vers le graphe
    # -------------------------
    def snap_to_graph(self, position):
        dist, idx = self.kdtree.query(position)
        return self.point_ids[idx]

    # -------------------------
    # A* hiérarchique (fonction externe)
    # -------------------------
    def astar(self, start, goal):
        return astar_hierarchical(
            start,
            goal,
            self.points,
            self.graph,
            self.neighbor_distances,
            self.areas,
            self.point_to_subarea
        )

    # -------------------------
    # Debug line
    # -------------------------
    def debug_line(self, pos1, pos2, color=(1, 0, 0, 1)):
        ls = LineSegs()
        ls.setThickness(2.0)
        ls.setColor(*color)
        ls.moveTo(pos1)
        ls.drawTo(pos2)
        node = ls.create()
        np_node = render.attachNewNode(node)
        np_node.setBin("fixed", 0)
        np_node.setDepthTest(False)
        self.debug_lines.append(np_node)

        def remove_node(task):
            if np_node in self.debug_lines:
                np_node.removeNode()
                self.debug_lines.remove(np_node)
            return task.done

        taskMgr.doMethodLater(0.5, remove_node, "remove_debug_line")
        return np_node

    # -------------------------
    # Ligne libre (raycast)
    # -------------------------
    def is_clear_line(self, id1, id2):
        pos1 = Point3(*self.points[id1]['pos'])
        pos2 = Point3(*self.points[id2]['pos'])
        if self.use_bullet and self.world:
            offsets = [Vec3(0, 0, 0), Vec3(0.3, 0, 0), Vec3(-0.3, 0, 0)]
            for o in offsets:
                if self.world.rayTestClosest(pos1 + o, pos2 + o).hasHit():
                    return False
        return True

    # -------------------------
    # Lissage chemin
    # -------------------------
    def smooth_path(self, path):
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = min(i + 10, len(path) - 1)  # regarder max 10 points en avant pour limiter raycasts
            while j > i + 1:
                if self.is_clear_line(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    # ================================
    # Définir un nouvel objectif
    # ================================
    def set_new_goal(self, pid):
        if pid not in self.points: return
        closest_id = self.snap_to_graph(self.Ai_node.getPos())
        self.start_id = closest_id
        self.goal_id = pid
        raw_path = self.astar(self.start_id, self.goal_id)
        if raw_path:
            self.path = self.smooth_path(raw_path)
            self.current_target_index = 0
            self.interrupt_path = True

    # ================================
    # Déplacement le long du chemin
    # ================================
    def move_along_path(self, task):
        if not self.path or self.current_target_index >= len(self.path):
            return task.cont
        pos = Vec3(self.Ai_node.getPos())

        # Look-ahead
        target_index = self.current_target_index
        for j in range(self.current_target_index + 1, len(self.path)):
            if self.is_clear_line(self.path[self.current_target_index], self.path[j]):
                target_index = j
            else: break

        # Bouger vers les points accessibles
        while self.current_target_index <= target_index and self.current_target_index < len(self.path):
            target = Vec3(*self.points[self.path[self.current_target_index]]['pos'])
            direction = target - pos
            dist = direction.length()
            if dist > 0.001:
                step = direction.normalized() * self.speed * globalClock.getDt()
                if step.length() > dist: step = direction
                self.Ai_node.setPos(pos + step)
                pos = Vec3(self.Ai_node.getPos())
            if dist >= 0.5: break
            self.current_target_index += 1

        # Debug chemin
        if self.debug:
            for i in range(len(self.path)-1):
                p1 = Vec3(*self.points[self.path[i]]['pos'])
                p2 = Vec3(*self.points[self.path[i+1]]['pos'])
                self.debug_line(p1,p2,(1,1,0,1))

        return task.cont

    # ================================
    # Suivi d’une cible
    # ================================
    def follow(self, task):
        if not self.aitarg: return task.again
        player_pos = self.aitarg.getPos(render)
        now = time.time()
        if not hasattr(self, "last_player_pos") or (player_pos - self.last_player_pos).length() > .5:
            if now - self._last_path_time > self._path_cooldown:
                closest_to_player = self.snap_to_graph(player_pos)
                if self.goal_id != closest_to_player or not self.path:
                    self.set_new_goal(closest_to_player)
                    self._last_path_time = now
            self.last_player_pos = Vec3(player_pos)
        self.move_along_path(task)
        self.Ai_node.lookAt(self.aitarg)
        self.Ai_node.setP(0)
        return task.again

    # -------------------------
    # Générer un point aléatoire à l'intérieur du rayon
    # -------------------------
    def random_wander_point(self):
        # Position actuelle de l'AI
        pos = Vec3(self.Ai_node.getPos())

        # Générer un offset aléatoire dans un cercle 2D
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, self.wander_radius)
        offset = Vec3(math.cos(angle) * radius, math.sin(angle) * radius, 0)
        target_pos = pos + offset

        # Snap au point le plus proche du graphe
        target_id = self.snap_to_graph(target_pos)
        return target_id

    # -------------------------
    # Tâche Wander
    # -------------------------
    def wander(self, task):
        # Si pas de chemin ou arrivé à destination, choisir un nouveau point
        if not self.path or self.current_target_index >= len(self.path):
            wander_pid = self.random_wander_point()
            self.set_new_goal(wander_pid)

        # Déplacer le long du chemin
        self.move_along_path(task)
        return task.cont
