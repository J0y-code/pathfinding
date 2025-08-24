# Pathfinding with Panda3D

This module implements a 3D pathfinding system using the A* algorithm in a Panda3D environment. It allows AI-controlled entities to follow a target (like the player) by dynamically calculating paths through a graph loaded from a `.pfs` file.

## Features

- A* pathfinding between graph nodes
- Player-following AI that dynamically updates its goal
- `.pfs` file parser for navigation points and connections
- Smooth AI movement with frame-based interpolation
- Periodic path recalculation based on player position

## Components

- `PFSParser`: Parses `.pfs` files to extract point positions and graph structure
- `Ai`: Core AI controller
  - `clpToobject()`: Finds the closest graph node to the player or target
  - `set_new_goal(pid)`: Sets a new goal and computes the path
  - `astar(start, goal)`: Calculates the shortest path using A*
  - `move_along_path(task)`: Moves the AI along the path over time
  - `updateAi(task)`: Re-evaluates the player's position and updates the goal

## How to create a psf file

1.import as an add-on "PSF_exporter.py" for blender 4.x

2.create a flat ground (plan) and subdivise into multiples faces, and select the ground

3.go to export and find export_scene.pds and select selected only and export

/!\ other objects in the scene that are not selected will be considered as obstacles.

PS: sorry if there are some element in french

## a .pfs file

```pfs
<Area> [name] {
  <Group_of_Points> {
    <Point> 1 {
      <x> {0}
      <y> {0}
      <z> {0}
      <Tag> [start]
    }
    <Point> 2 {
      <x> {5}
      <y> {0}
      <z> {0}
    }
  }
  <Relate> {
    <Crossroad> {1}
    <Peripheral> {2}
    <Cost> {1}
  }
}
```

## Python Usage Example

```python
from direct.showbase.ShowBase import ShowBase
from pathfinding import PFSParser, Ai
from panda3d.core import Vec3

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the player model
        self.player = loader.loadModel("models/player.egg")
        self.player.reparentTo(render)
        self.player.setPos(0, 0, 0)

        # Load navigation graph
        parser = PFSParser("level1.pfs")
        points, graph = parser.load()

        # Initialize the AI
        ai_model = loader.loadModel("models/enemy.egg")
        self.enemy = Ai(points=points, graph=graph, model=ai_model, start_id=1)
        self.enemy.aitarg = self.player

        # Start AI behavior
        self.taskMgr.do_method_later(0.5, self.enemy.follow, "UpdateAITask")
        self.taskMgr.add(self.enemy.move_along_path, "MoveAIPathTask")

        # Optional: Move the player slightly every frame
        self.taskMgr.add(self.move_player, "MovePlayer")

    def move_player(self, task):
        self.player.setX(self.player.getX() + 0.05)
        return task.cont

app = MyApp()
app.run()
```

## Demo Video

[Download and watch the demo video](video/samples_wandering.mp4)

[Demo animation](video/samples_wandering.gif)
