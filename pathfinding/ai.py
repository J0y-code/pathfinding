from panda3d.core import NodePath, Vec3, Point3, LineSegs
import numpy as np
from scipy.spatial import KDTree
import random, math, os, time, threading, queue

from .ai_utils import AiUtils
from .astar import astar_precomputed, astar_hierarchical, build_neighbor_distances
from .subarea_graph import build_subarea_graph, precompute_gateways
from .parser import PFSParser
from .profiler import *

class Ai:
    fast_dist = staticmethod(AiUtils.fast_dist)
    fast_dist_cached = staticmethod(AiUtils.fast_dist_cached)

    def __init__(self, points, graph, areas=None, model=None, start_id=None, world=None, use_bullet=True):
        # Chargement / génération du graphe optimisé
        cache_file = "graph_cache.npz"
        if os.path.exists(cache_file):
            print("[CACHE] Chargement du graphe pré-calculé...")
            data = np.load(cache_file, allow_pickle=True)
            self.neighbor_distances = data["neighbor_distances"].item()
        else:
            print("[CACHE] Aucun cache trouvé, génération...")
            self.neighbor_distances = build_neighbor_distances(graph, points)
            print("[CACHE] Sauvegarde du graphe optimisé...")
            np.savez_compressed(cache_file, neighbor_distances=self.neighbor_distances)

        self.use_bullet = use_bullet
        self.world = world

        # Données du graphe
        self.points = points
        self.graph = graph
        self.areas = areas or {}
        self.point_to_subarea = {pid: data['subarea'] for pid, data in points.items()}

        # KDTree pour snap rapide
        self.point_ids = list(points.keys())
        self.positions_array = np.array([points[pid]['pos'] for pid in self.point_ids], dtype=np.float32)
        self.kdtree = KDTree(self.positions_array)

        # Positions numpy pour A*
        self.positions = {pid: np.array(data['pos'], dtype=np.float32) for pid, data in points.items()}

        # Node principal
        self.Ai_node = NodePath("Ai_n")
        self.Ai_node.reparentTo(render)
        self.model = model
        if self.model:
            self.model.reparentTo(self.Ai_node)

        # Position initiale
        self.start_id = start_id
        if start_id:
            self.Ai_node.setPos(*self.points[start_id]['pos'])

        # État courant
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

        # Graphe hiérarchique
        self.subarea_graph = build_subarea_graph(points, self.areas, graph)
        self.gateways = precompute_gateways(points, graph, self.point_to_subarea)

        # Cooldown de recalcul
        self._last_path_time = 0
        self._path_cooldown = 0.2

        # ======================
        # THREADING PATHFINDING
        # ======================
        self.path_requests = queue.Queue()
        self.path_results = queue.Queue()
        self._path_thread = threading.Thread(target=self._path_worker, daemon=True)
        self._path_thread.start()

        # Task Panda3D pour surveiller les résultats
        taskMgr.add(self._check_path_results, f"check_path_results_{id(self)}")

    # ---------------------------------------------------------------------
    # THREAD WORKER
    # ---------------------------------------------------------------------
    def _path_worker(self):
        """Thread dédié au pathfinding (A* hiérarchique)."""
        while True:
            try:
                item = self.path_requests.get()
                if item is None:
                    break
                start, goal = item
                path = self.astar(start, goal)
                self.path_results.put((start, goal, path))
            except Exception as e:
                print(f"[AI Thread] Erreur de pathfinding: {e}")

    # ---------------------------------------------------------------------
    # RÉCEPTION DES RÉSULTATS THREADÉS
    # ---------------------------------------------------------------------
    def _check_path_results(self, task):
        """Vérifie périodiquement si un chemin calculé est prêt."""
        try:
            while True:
                start, goal, path = self.path_results.get_nowait()
                if start == self.start_id and goal == self.goal_id and path:
                    self.path = self.smooth_path(path)
                    self.current_target_index = 0
                    self.interrupt_path = True
                    print(f"[AI] Nouveau chemin reçu ({len(self.path)} points)")
        except queue.Empty:
            pass
        return task.cont

    # ---------------------------------------------------------------------
    # FONCTIONS GRAPHE
    # ---------------------------------------------------------------------
    def snap_to_graph(self, position):
        pos = (position.getX(), position.getY(), position.getZ()) if hasattr(position, "getX") else tuple(position)
        if hasattr(self, "_last_snap") and Ai.fast_dist(pos, self._last_snap[0]) < 0.2:
            return self._last_snap[1]
        dist, idx = self.kdtree.query(pos)
        pid = self.point_ids[idx]
        self._last_snap = (pos, pid)
        return pid

    def astar(self, start, goal):
        """Appelle ton A* hiérarchique optimisé (fonction externe)."""
        return astar_hierarchical(
            start,
            goal,
            self.points,
            self.graph,
            self.neighbor_distances,
            self.areas,
            self.point_to_subarea,
            subarea_graph=self.subarea_graph,
            gateways=self.gateways,
            positions=self.positions
        )

    # ---------------------------------------------------------------------
    # VISUALISATION DEBUG
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # CHEMIN
    # ---------------------------------------------------------------------
    def is_clear_line(self, id1, id2):
        pos1 = Point3(*self.points[id1]['pos'])
        pos2 = Point3(*self.points[id2]['pos'])
        if self.use_bullet and self.world:
            offsets = [Vec3(0, 0, 0), Vec3(0.3, 0, 0), Vec3(-0.3, 0, 0)]
            for o in offsets:
                if self.world.rayTestClosest(pos1 + o, pos2 + o).hasHit():
                    return False
        return True

    def smooth_path(self, path):
        """Lissage intelligent du chemin (raycasts rapides)."""
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = min(i + 10, len(path) - 1)
            while j > i + 1:
                if self.is_clear_line(path[i], path[j]):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    # ---------------------------------------------------------------------
    # NOUVEL OBJECTIF
    # ---------------------------------------------------------------------
    def set_new_goal(self, pid):
        if pid not in self.points:
            return
        closest_id = self.snap_to_graph(self.Ai_node.getPos())
        self.start_id = closest_id
        self.goal_id = pid
        print(f"[AI] Requête de chemin envoyée ({self.start_id} → {self.goal_id})")
        self.path_requests.put((self.start_id, self.goal_id))

    # ---------------------------------------------------------------------
    # DÉPLACEMENT
    # ---------------------------------------------------------------------
    def move_along_path(self, task):
        if not self.path or self.current_target_index >= len(self.path):
            return task.cont

        pos = Vec3(self.Ai_node.getPos())

        target_index = self.current_target_index
        for j in range(self.current_target_index + 1, len(self.path)):
            if self.is_clear_line(self.path[self.current_target_index], self.path[j]):
                target_index = j
            else:
                break

        while self.current_target_index <= target_index and self.current_target_index < len(self.path):
            target = Vec3(*self.points[self.path[self.current_target_index]]['pos'])
            direction = target - pos
            dist = direction.length()
            if dist > 0.001:
                step = direction.normalized() * self.speed * globalClock.getDt()
                if step.length() > dist:
                    step = direction
                self.Ai_node.setPos(pos + step)
                pos = Vec3(self.Ai_node.getPos())
            if dist >= 0.5:
                break
            self.current_target_index += 1

        if self.debug:
            for i in range(len(self.path) - 1):
                p1 = Vec3(*self.points[self.path[i]]['pos'])
                p2 = Vec3(*self.points[self.path[i + 1]]['pos'])
                self.debug_line(p1, p2, (1, 1, 0, 1))

        return task.cont

    # ---------------------------------------------------------------------
    # FOLLOW & WANDER
    # ---------------------------------------------------------------------
    def follow(self, task):
        if not self.aitarg:
            return task.again

        player_pos = self.aitarg.getPos(render)
        now = time.time()

        if not hasattr(self, "last_player_pos") or (player_pos - self.last_player_pos).length() > 0.5:
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

    def random_wander_point(self):
        pos = Vec3(self.Ai_node.getPos())
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, self.wander_radius)
        offset = Vec3(math.cos(angle) * radius, math.sin(angle) * radius, 0)
        target_pos = pos + offset
        target_id = self.snap_to_graph(target_pos)
        return target_id

    def wander(self, task):
        if not self.path or self.current_target_index >= len(self.path):
            wander_pid = self.random_wander_point()
            self.set_new_goal(wander_pid)
        self.move_along_path(task)
        return task.cont
