__author__ = "Nathan Pflieger-Chakma"

"""
Attention: the code is not finished.
There will be modifications.
I advise you to check if the repository has been updated.
You can also modify the code if you want.
"""

import heapq
import re
import sys
import os
import random

from direct.showbase.ShowBase import *
from panda3d.core import Vec3, NodePath

import astar_module
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- PFSParser --------------------
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

# -------------------- NextPointNet (PyTorch) --------------------
class NextPointNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, output_dim=1):
        # input_dim=6: AI position (x,y,z) + target position (x,y,z)
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# -------------------- Ai --------------------
class Ai:
    def __init__(self, points, graph, model=None, start_id=None):
        self.debug = False
        self.Ai_node = NodePath("Ai_n")
        self.Ai_node.reparentTo(render)
        self.model = model
        self.model.reparentTo(self.Ai_node)
        self.parent = None # Will be set later by the parent app

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

        # Wandering parameters
        self.wander_radius = 2
        self.use_spawn_as_center = True
        self.last_targets = []

    # -------- Pathfinding & Movement --------
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

    def heuristic(self, p1, p2):
        return sum(abs(a - b) for a, b in zip(p1, p2))

    def astar(self, start, goal):
        point_dict = {pid: list(data['pos']) for pid, data in self.points.items()}
        return astar_module.astar(start, goal, self.graph, point_dict)

    # -------- Player Tracking --------
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

    # -------- Wandering --------
    
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

    # -------- Machine Learning --------

    def train(self, task, epochs=500):
        """
        Non-blocking Panda3D task for training the model to follow aitarg.
        While training, the AI follows the target using the classic algorithm.
        After training, the ML prediction is activated automatically.
        """
        # Start following the target with the classic algorithm during training
        if not hasattr(self, '_follow_task_started'):
            self._follow_task_started = True
            # Start follow task if not already running
            if hasattr(self, 'parent'):
                self._follow_task = self.parent.taskMgr.add(self.follow, "FollowTask")
            else:
                print("Warning: self.app not set, can't start follow task.")

        # Initialize training state on first call
        if not hasattr(self, '_train_state'):
            if not self.aitarg:
                print("No aitarg set for training.")
                return task.done
            
            aitarg_pos = self.aitarg.getPos(render)

            
            # Récupération de tous les ID uniques utilisés dans le graphe
            unique_ids = sorted(set(
                neighbor
                for pid in self.points
                for neighbor in self.graph[pid]
            ))

            # Mappings pour passer entre identifiants et classes
            self.id_to_class = {pid: idx for idx, pid in enumerate(unique_ids)}
            self.class_to_id = {idx: pid for pid, idx in self.id_to_class.items()}


            X = []
            y = []

            for pid, data in self.points.items():
                ai_pos = data['pos']
                for neighbor in self.graph[pid]:
                    X.append(list(ai_pos) + [aitarg_pos.x, aitarg_pos.y, aitarg_pos.z])
                    y.append(self.id_to_class[neighbor])  # ✅ Classification: index de classe

            # Convertir en tensors
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)  # CrossEntropyLoss nécessite long


            self.next_point_to_aitarg_model = NextPointNet(input_dim=6)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.next_point_to_aitarg_model.parameters(), lr=0.01)

            self._train_state = {
                'X': X,
                'y': y,
                'criterion': criterion,
                'optimizer': optimizer,
                'epoch': 0,
                'epochs': epochs
            }

        state = self._train_state
        # Train one epoch per frame
        state['optimizer'].zero_grad()
        outputs = self.next_point_to_aitarg_model(state['X'])
        loss = state['criterion'](outputs, state['y'])
        loss.backward()
        state['optimizer'].step()
        if self.debug and state['epoch'] % 10 == 0:
            print(f"[Aitarg Task] Epoch {state['epoch']}, Loss: {loss.item():.4f}")

        state['epoch'] += 1
        if state['epoch'] >= state['epochs']:
            print("Training task finished.")
            del self._train_state
            # Stop follow task and start ML prediction task
            if hasattr(self, '_follow_task'):
                self.parent.taskMgr.remove(self._follow_task)
            self.parent.taskMgr.do_method_later(0, self.ML_seek, "PredictTask", appendTask=True)
            return task.done
        return task.cont
    
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
            logits = self.next_point_to_aitarg_model(input_tensor)  # → sortie brute
            predicted_class = torch.argmax(logits, dim=1).item()    # → index de classe avec probabilité max

        # Convertir la classe vers un ID de point réel
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

# exemple of usage
if __name__ == "__main__":
    env = PFSParser("level1.pfs")
    points, graph = env.load()

    app = ShowBase()
    model = loader.loadModel('playertest.egg')

    ai = Ai(points, graph, model, start_id=240)
    ai.speed = 5
    ai.debug = True
    ai.wander_radius = 25
    ai.use_spawn_as_center = False

    # Example: Set aitarg (target NodePath)
    ai.aitarg = loader.loadModel('target.egg')
    ai.aitarg.setPos(10, 10, 0)
    ai.aitarg.reparentTo(render)

    # Start threaded training as a Panda3D task
    app.taskMgr.add(ai.move_along_path, "MoveTask")
    app.taskMgr.add(lambda task: ai.train(task, epochs=500), "TrainTask")



    app.run()
