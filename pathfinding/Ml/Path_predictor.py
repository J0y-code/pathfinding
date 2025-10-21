import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import heapq

class PathPredictorNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super(PathPredictorNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PathPredictor:
    def __init__(self, points, graph):
        self.points = points
        self.graph = graph
        self.model = PathPredictorNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95

    def collect_data(self, player_path):
        for i in range(len(player_path) - 1):
            state = player_path[i]
            next_state = player_path[i + 1]
            self.memory.append((state, next_state))

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([torch.FloatTensor(state) for state, _ in batch])
        next_states = torch.stack([torch.FloatTensor(next_state) for _, next_state in batch])

        predictions = self.model(states)
        loss = self.criterion(predictions, next_states)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict_path(self, current_state):
        with torch.no_grad():
            current_state_tensor = torch.FloatTensor(current_state)
            prediction = self.model(current_state_tensor)
        return prediction.numpy()

    def plan_ambush(self, current_state):
        predicted_next_state = self.predict_path(current_state)
        predicted_next_point = self.find_closest_point(predicted_next_state)
        ambush_path = self.astar(current_state, predicted_next_point)
        return ambush_path

    def find_closest_point(self, state):
        state = np.array(state)
        closest_point = None
        min_distance = float('inf')
        for point_id, point_data in self.points.items():
            point = np.array(point_data['pos'])
            distance = np.linalg.norm(state - point)
            if distance < min_distance:
                min_distance = distance
                closest_point = point_id
        return closest_point

    def astar(self, start, goal):
        point_dict = {pid: list(data['pos']) for pid, data in self.points.items()}
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {node: float('inf') for node in self.graph}
        g_score[start] = 0
        f_score = {node: float('inf') for node in self.graph}
        f_score[start] = self.heuristic(point_dict[start], point_dict[goal])

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
            for neighbor in self.graph[current]:
                tentative_g_score = g_score[current] + self.heuristic(point_dict[current], point_dict[neighbor])
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(point_dict[neighbor], point_dict[goal])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def heuristic(self, p1, p2):
        return sum(abs(a - b) for a, b in zip(p1, p2))
