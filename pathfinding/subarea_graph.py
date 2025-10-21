from .profiler import *

@profile
def build_subarea_graph(points, areas, graph):
    # Préparer un mapping rapide point → subarea
    point_to_subarea = {}
    for pid, data in points.items():
        point_to_subarea[pid] = data['subarea']

    # Pré-allouer les sous-aires connues
    subarea_graph = {}
    for area in areas.values():
        for sub in area.keys():
            subarea_graph[sub] = set()

    # Boucle principale
    for pid, neighbors in graph.items():
        sa1 = point_to_subarea[pid]
        for n in neighbors:
            sa2 = point_to_subarea[n]
            if sa1 != sa2:
                subarea_graph[sa1].add(sa2)
                subarea_graph[sa2].add(sa1)

    # Conversion en listes pour sérialisation
    for k in subarea_graph:
        subarea_graph[k] = list(subarea_graph[k])
    return subarea_graph


@profile
def precompute_gateways(points, graph, point_to_subarea):
    """
    Retourne mapping {(sa, neighbor_sa): [point_ids,...], ...}
    Pour chaque arête reliant deux subareas, on collecte les points côté A et côté B.
    """
    gateways = {}

    for pid, neighbors in graph.items():
        psa = point_to_subarea.get(pid)
        if psa is None:
            continue
        for n in neighbors:
            nsa = point_to_subarea.get(n)
            if nsa is None or nsa == psa:
                continue
            key = (psa, nsa)
            key_rev = (nsa, psa)
            # ajouter pid comme gateway pour la paire (psa, nsa)
            gateways.setdefault(key, set()).add(pid)
            # ajouter n comme gateway pour la paire (nsa, psa)
            gateways.setdefault(key_rev, set()).add(n)

    # convertir sets → listes
    for k in list(gateways.keys()):
        gateways[k] = list(gateways[k])
    return gateways

