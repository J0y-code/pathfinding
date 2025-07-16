__author__ = "Nathan Pflieger-Chakma"

"""attention the code is not finished, 
there will be modifications, 
I advise you to check if the repository has been updated, 
you can also modify the code if you want."""


import re
import heapq

import sys, os
from panda3d.core import Vec3


from panda3d.core import (
    Point3, LineSegs, NodePath, LColor, CollisionTraverser, CollisionNode,
    CollisionRay, CollisionHandlerQueue, CollisionSphere, BitMask32, DirectionalLight, Vec3
)

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
        self.Ai_node = NodePath("Ai_n")
        self.Ai_node.reparentTo(render)
        self.model = model
        self.model.reparentTo(self.Ai_node)
        
        #
        self.points = points
        self.graph = graph
        self.start_id = start_id
        self.goal_id = None
        self.current_target_index = 0
        self.path = []
        self.interrupt_path = False

        #Ai parameters
        self.speed = 5.0
        self.aitarg = None

    def set_new_goal(self, pid):
        if not self.points or pid not in self.points:
            print("invalid point.")
            return

        current_pos = Vec3(self.Ai_node.getPos())
        closest_id = min(
            self.points,
            key=lambda i: (Vec3(*self.points[i]['pos']) - current_pos).lengthSquared()
        )

        if pid == closest_id:
            print("goal point is the actual position.")
            return

        self.start_id = closest_id
        self.goal_id = pid
        new_path = self.astar(self.start_id, self.goal_id)

        if new_path:
            print(f"New path : {' -> '.join(map(str, new_path))}")
            self.path = new_path
            self.current_target_index = 0
            self.interrupt_path = True  # Signal to restart movement immediately
        else:
            print("No path found.")
            self.path = []

            
            

    def check_cible_movement(self, cible):
        self.cible = cible
        global player_last_pos
        current_pos = self.cible.getPos(render)

        if (current_pos - player_last_pos).lengthSquared() > 0.0001:
            print("Le joueur a bougé.")
            print(f"[DEBUG] Player moved: {moved}")
            player_last_pos = current_pos  # Met à jour pour la prochaine frame
        return True
    

    def clpToobject(self, cible):
        # Si c'est un Player, on récupère self.model
        if hasattr(cible, 'model'):
            cible = cible.model

        player_pos = cible.getPos(render)

        closest_id = min(
            self.points,
            key=lambda pid: (Vec3(*self.points[pid]['pos']) - player_pos).lengthSquared()
        )

        print(f"Closest point to player is ID {closest_id} at {self.points[closest_id]['pos']}")
        return closest_id



    def move_along_path(self, task):
        if not self.path or self.current_target_index >= len(self.path) - 1:
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
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float('inf') for node in self.graph}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.graph}
        f_score[start] = self.heuristic(self.points[start]['pos'], self.points[goal]['pos'])

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for neighbor in self.graph.get(current, []):
                tentative_g = g_score[current] + self.heuristic(
                    self.points[current]['pos'], self.points[neighbor]['pos']
                )
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(
                        self.points[neighbor]['pos'], self.points[goal]['pos']
                    )
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None
        
    def follow(self, task):
        if not self.aitarg:
            return task.again

        # Suivre le joueur en permanence (toutes les 0.5s)
        closest_to_player = self.clpToobject(self.aitarg)
        self.set_new_goal(closest_to_player)

        return task.again

    def wanderer(self, task):
        from random import randint
        self.set_new_goal(randint(1,400))

        return task.again

    


        
if __name__ == "__main__":
    PFS_env = PFSParser("level1.pfs")
    points, graph = PFS_env.load()
    

    
    from direct.showbase.ShowBase import *
    from panda3d.core import *
    from direct.task import Task
    from random import randint
    from time import sleep
    
    Test = ShowBase()
    model = loader.loadModel('playertest.egg')

    ai = Ai(points=points, graph=graph, model=model, start_id=1)
    ai.speed = 5

    ai.set_new_goal(240)
    Test.taskMgr.add(ai.move_along_path, "MoveTask")

    Test.taskMgr.do_method_later(5, ai.wanderer, "MoveTask")
    
    Test.run()
