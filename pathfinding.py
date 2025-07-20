__author__ = "Nathan Pflieger-Chakma"

"""attention the code is not finished, 
there will be modifications, 
I advise you to check if the repository has been updated, 
you can also modify the code if you want."""


import re
import heapq

import sys, os
from panda3d.core import Vec3, NodePath
import astar_module

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
        
        
        
        
class Ai:
    def __init__(self, points, graph, model=None, start_id=None):
        # model initialisation
        self.debug = False
        self.Ai_node = NodePath("Ai_n")
        self.Ai_node.reparentTo(render)
        self.model = model
        self.model.reparentTo(self.Ai_node)
        
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
            self.spawn_pos = Vec3(self.start_pos) if self.start_pos else Vec3(0,0,0)

        #Ai parameters
        self.speed = 5.0
        self.aitarg = None
        
        # wandering parameters
        # Par défaut, on considère rayon 50 unités (à adapter)
        self.wander_radius = 2
        # Si tu veux choisir si le rayon est autour du spawn ou de la position actuelle
        self.use_spawn_as_center = True  # True = spawn, False = position actuelle
        self.last_targets = []


    def set_new_goal(self, pid):
        if not self.points or pid not in self.points:
            if self.debug:
                print("invalid point.")
            return

        current_pos = Vec3(self.Ai_node.getPos())
        closest_id = min(
            self.points,
            key=lambda i: (Vec3(*self.points[i]['pos']) - current_pos).lengthSquared()
        )

        if pid == closest_id:
            if self.debug:
                print("goal point is the actual position.")
            return

        self.start_id = closest_id
        self.goal_id = pid
        new_path = self.astar(self.start_id, self.goal_id)

        if new_path:
            if self.debug:
                print(f"New path : {' -> '.join(map(str, new_path))}")
            self.path = new_path
            self.current_target_index = 0
            self.interrupt_path = True  # Signal to restart movement immediately
        else:
            if self.debug:
                print("No path found.")
            self.path = []
            

    def check_cible_movement(self, cible):
        self.cible = cible
        global player_last_pos
        current_pos = self.cible.getPos(render)

        if (current_pos - player_last_pos).lengthSquared() > 0.0001:
            if self.debug:
                print("Le joueur a bougé.")
                print(f"[DEBUG] Player moved: {moved}")
            player_last_pos = current_pos  # Met à jour pour la prochaine frame
        return True
    

    def clpToobject(self, cible):
        # Si c'est un Player, on récupère self.model
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


    def move_along_path(self, task):
        if not self.path or self.current_target_index >= len(self.path) - 1:
            if self.debug:
                print('no path found')
            return task.cont

        if self.interrupt_path:
            # Recommencer le chemin dès que possible
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


    def heuristic(self, p1, p2):
        return sum(abs(a - b) for a, b in zip(p1, p2))

    def astar(self, start, goal):
        point_dict = {pid: list(data['pos']) for pid, data in self.points.items()}
        return astar_module.astar(start, goal, self.graph, point_dict)
        
    def follow(self, task):
        if not self.aitarg:
            return task.again

        # Suivre le joueur en permanence (toutes les 0.5s)
        closest_to_player = self.clpToobject(self.aitarg)
        self.set_new_goal(closest_to_player)

        return task.again


    def wanderer(self, high_pid=None, task=None):
        import random
        if not self.points:
            if self.debug:
                print("[WARN] No points available.")
            return task.again if task else None

        # Choix du centre de recherche
        if self.use_spawn_as_center:
            center = self.spawn_pos
        else:
            center = Vec3(self.Ai_node.getPos())

        # Filtrer les points dans le rayon
        valid_ids = []
        for pid, data in self.points.items():
            pos = Vec3(*data['pos'])
            if (pos - center).length() <= self.wander_radius:
                valid_ids.append(pid)

        if high_pid is not None:
            valid_ids = [pid for pid in valid_ids if pid <= high_pid]
            if not valid_ids:
                if self.debug:
                    print(f"[WARN] No valid points under ID {high_pid} in radius")
                return task.again if task else None

        # Eviter de revenir sur les derniers points (optionnel)
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

        
if __name__ == "__main__":
    env = PFSParser("level1.pfs")
    points, graph = env.load()

    app = ShowBase()
    model = loader.loadModel('playertest.egg')

    ai = Ai(points, graph, model, start_id=240)
    ai.speed = 5
    ai.debug = True
    ai.wander_radius = 25  # rayon élargi
    ai.use_spawn_as_center = False

    app.taskMgr.add(ai.move_along_path, "MoveTask")
    app.taskMgr.do_method_later(5, ai.wanderer, "WanderTask", extraArgs=[400], appendTask=True)
    app.run()
