import re
from .profiler import *

class PFSParser:
    def __init__(self, filename):
        self.filename = filename
        self.areas = {}   # {area_name: {subarea_name: [point_ids]}}
        self.points = {}  # {id: {'pos': (x,y,z), 'tag': ..., 'area': ..., 'subarea': ...}}
        self.graph = {}   # {id: [neighbor_ids]}

    @profile
    def load(self):
        # --- Regex pré-compilées pour vitesse ---
        re_brackets = re.compile(r"\[(.*?)\]")
        re_braces_num = re.compile(r"{(\d+)}")
        re_braces_float = re.compile(r"{(.*?)}")
        re_point_id = re.compile(r"<Point>\s+(\d+)")

        current_area = None
        current_subarea = None
        current_point_id = None
        current_point = {}
        in_point_block = False
        in_connections = False
        crossroad_id = None

        # --- Lecture en flux (pas tout en mémoire) ---
        with open(self.filename, 'r', encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # --- Détection des zones principales ---
                if line.startswith("<Area>"):
                    match = re_brackets.search(line)
                    if match:
                        current_area = match.group(1)
                        self.areas.setdefault(current_area, {})
                    continue

                if line.startswith("<SubArea>"):
                    match = re_brackets.search(line)
                    if match:
                        current_subarea = match.group(1)
                        self.areas[current_area].setdefault(current_subarea, [])
                    continue

                # --- Bloc Point ---
                if line.startswith("<Point>"):
                    match = re_point_id.search(line)
                    if match:
                        current_point_id = int(match.group(1))
                        current_point.clear()
                        in_point_block = True
                    continue

                if in_point_block:
                    if line.startswith("<x>"):
                        current_point['x'] = float(re_braces_float.search(line).group(1))
                    elif line.startswith("<y>"):
                        current_point['y'] = float(re_braces_float.search(line).group(1))
                    elif line.startswith("<z>"):
                        current_point['z'] = float(re_braces_float.search(line).group(1))
                    elif line.startswith("<Tag>"):
                        tag_match = re_brackets.search(line)
                        if tag_match:
                            current_point['tag'] = tag_match.group(1)
                    elif line == "}":
                        # Fin du point
                        self.points[current_point_id] = {
                            'pos': (
                                current_point.get('x', 0.0),
                                current_point.get('y', 0.0),
                                current_point.get('z', 0.0)
                            ),
                            'tag': current_point.get('tag'),
                            'area': current_area,
                            'subarea': current_subarea
                        }
                        self.graph.setdefault(current_point_id, [])
                        if current_area and current_subarea:
                            self.areas[current_area][current_subarea].append(current_point_id)
                        in_point_block = False
                    continue

                # --- Bloc des connexions ---
                if line.startswith("<Relate>"):
                    in_connections = True
                    continue
                if in_connections and line == "}":
                    in_connections = False
                    crossroad_id = None
                    continue

                if in_connections:
                    if line.startswith("<Crossroad>"):
                        match = re_braces_num.search(line)
                        if match:
                            crossroad_id = int(match.group(1))
                            self.graph.setdefault(crossroad_id, [])
                    elif line.startswith("<Peripheral>") and crossroad_id is not None:
                        match = re_braces_num.search(line)
                        if match:
                            pid = int(match.group(1))
                            if pid not in self.graph[crossroad_id]:
                                self.graph[crossroad_id].append(pid)
                            self.graph.setdefault(pid, [])
                            if crossroad_id not in self.graph[pid]:
                                self.graph[pid].append(crossroad_id)

        return self.areas, self.points, self.graph



class PFSParserFast:
    def __init__(self, filename):
        self.filename = filename
        self.areas = {}   # {area_name: {subarea_name: [point_ids]}}
        self.points = {}  # {id: {'pos': (x,y,z), 'tag': ..., 'area': ..., 'subarea': ...}}
        self.graph = {}   # {id: [neighbor_ids]}

    def load(self):
        with open(self.filename, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()

        current_area = None
        current_subarea = None
        current_point_id = None
        current_point = {}
        in_point_block = False
        in_connections = False
        crossroad_id = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # --- Zones ---
            if line.startswith("<Area>"):
                current_area = line[line.find("[")+1:line.find("]")]
                self.areas.setdefault(current_area, {})
                continue
            if line.startswith("<SubArea>"):
                current_subarea = line[line.find("[")+1:line.find("]")]
                self.areas[current_area].setdefault(current_subarea, [])
                continue

            # --- Points ---
            if line.startswith("<Point>"):
                current_point_id = int(line.split()[1])
                current_point.clear()
                in_point_block = True
                continue

            if in_point_block:
                if line.startswith("<x>"):
                    current_point['x'] = float(line[line.find("{")+1:line.find("}")])
                elif line.startswith("<y>"):
                    current_point['y'] = float(line[line.find("{")+1:line.find("}")])
                elif line.startswith("<z>"):
                    current_point['z'] = float(line[line.find("{")+1:line.find("}")])
                elif line.startswith("<Tag>"):
                    start = line.find("[")+1
                    end = line.find("]")
                    current_point['tag'] = line[start:end] if start != -1 and end != -1 else None
                elif line == "}":
                    self.points[current_point_id] = {
                        'pos': (current_point.get('x',0.0),
                                current_point.get('y',0.0),
                                current_point.get('z',0.0)),
                        'tag': current_point.get('tag'),
                        'area': current_area,
                        'subarea': current_subarea
                    }
                    self.graph.setdefault(current_point_id, [])
                    if current_area and current_subarea:
                        self.areas[current_area][current_subarea].append(current_point_id)
                    in_point_block = False
                continue

            # --- Connexions ---
            if line.startswith("<Relate>"):
                in_connections = True
                continue
            if in_connections and line == "}":
                in_connections = False
                crossroad_id = None
                continue
            if in_connections:
                if line.startswith("<Crossroad>"):
                    crossroad_id = int(line[line.find("{")+1:line.find("}")])
                    self.graph.setdefault(crossroad_id, [])
                elif line.startswith("<Peripheral>") and crossroad_id is not None:
                    pid = int(line[line.find("{")+1:line.find("}")])
                    if pid not in self.graph[crossroad_id]:
                        self.graph[crossroad_id].append(pid)
                    self.graph.setdefault(pid, [])
                    if crossroad_id not in self.graph[pid]:
                        self.graph[pid].append(crossroad_id)

        return self.areas, self.points, self.graph
