# astar.py (modifié)
import heapq
import numpy as np
from .profiler import *
from .subarea_graph import build_subarea_graph
from .ai_utils import AiUtils

@profile
def build_neighbor_distances(graph, points):
    positions = {pid: np.array(data['pos'], dtype=np.float32) for pid, data in points.items()}
    fastdist = AiUtils.fast_dist
    neighbor_distances = {}
    for pid, neighbors in graph.items():
        neighbor_distances[pid] = {n: fastdist(positions[pid], positions[n]) for n in neighbors}
    return neighbor_distances

@profile
def astar_precomputed(start, goal, graph, points, neighbor_distances, positions=None):
    """
    positions: optional dict {pid: np.array([...],dtype=float32)} to avoid rebuilding each call.
    """
    if start not in graph or goal not in graph:
        return None

    # use precomputed numpy positions if provided, otherwise build lightweight view (only tuples -> np.array on demand)
    if positions is None:
        positions = {pid: np.array(points[pid]['pos'], dtype=np.float32) for pid in points}

    fastdist = AiUtils.fast_dist
    open_heap = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: fastdist(positions[start], positions[goal])}
    open_set = {start}

    pos_goal = positions[goal]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current not in open_set:
            continue
        open_set.discard(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        cur_g = g_score[current]
        pos_cur = positions[current]

        # iterate neighbors (neighbor_distances uses precomputed distances)
        for neighbor, dist in neighbor_distances[current].items():
            tentative_g = cur_g + dist
            if tentative_g < g_score.get(neighbor, 1e9):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + fastdist(positions[neighbor], pos_goal)
                f_score[neighbor] = f
                heapq.heappush(open_heap, (f, neighbor))
                open_set.add(neighbor)
    return None

@profile
def astar_hierarchical(start_pid, goal_pid, points, graph, neighbor_distances,
                       areas, point_to_subarea,
                       subarea_graph=None, gateways=None, positions=None):
    """
    Accept precomputed subarea_graph and gateways to avoid rebuilding them every call.
    gateways is expected as mapping (sa, neighbor_sa) -> list of gateway point ids (precomputed).
    positions: optional dict {pid: np.array([...])} passed to astar_precomputed.
    """

    start_sa = point_to_subarea[start_pid]
    goal_sa = point_to_subarea[goal_pid]
    if start_sa == goal_sa:
        return astar_precomputed(start_pid, goal_pid, graph, points, neighbor_distances, positions=positions)

    if subarea_graph is None:
        subarea_graph = build_subarea_graph(points, areas, graph)

    open_heap = [(0, start_sa)]
    came_from = {}
    g_score = {sa: float('inf') for sa in subarea_graph}
    g_score[start_sa] = 0
    f_score = {sa: float('inf') for sa in subarea_graph}
    f_score[start_sa] = 0

    while open_heap:
        _, current_sa = heapq.heappop(open_heap)
        if current_sa == goal_sa:
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
        return None

    # Chemins locaux — utiliser gateways quand disponible
    full_path = []
    current_point = start_pid
    for idx, sa in enumerate(sa_path):
        # si c'est la subarea finale, target = goal_pid
        if sa == goal_sa:
            target_points = [goal_pid]
        else:
            # choisir les gateways reliant sa -> next_sa (si gateways fournis)
            next_sa = sa_path[idx + 1]
            if gateways is not None:
                # gateways may be stored both directions
                key = (sa, next_sa)
                key_rev = (next_sa, sa)
                candidate_gateways = gateways.get(key) or gateways.get(key_rev) or []
                if not candidate_gateways:
                    # fallback: tous les points de la subarea (lent)
                    candidate_gateways = [pid for pid, psa in point_to_subarea.items() if psa == sa]
            else:
                candidate_gateways = [pid for pid, psa in point_to_subarea.items() if psa == sa]
            target_points = candidate_gateways

        # choisir le gateway le plus proche du point courant
        best_target = min(target_points,
                          key=lambda p: np.linalg.norm(np.array(points[p]['pos'], dtype=np.float32) -
                                                       np.array(points[current_point]['pos'], dtype=np.float32)))
        local_path = astar_precomputed(current_point, best_target, graph, points, neighbor_distances, positions=positions)
        if not local_path:
            return None
        if full_path:
            local_path = local_path[1:]
        full_path.extend(local_path)
        current_point = best_target

    return full_path
