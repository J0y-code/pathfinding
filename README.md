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

## Example `.pfs` Format

```pfs
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
<Crossroad> {1}
<Peripheral> {2}


## Example usage

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3
from pathfinding import PFSParser, Ai

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load player model (to be followed)
        self.player = loader.loadModel("player.egg")
        self.player.reparentTo(render)
        self.player.setPos(0, 0, 0)

        # Load pathfinding data
        parser = PFSParser("Assets/graphs/level1.pfs")
        points, graph = parser.load()

        # Load AI model
        ai_model = loader.loadModel("enemy.egg")
        self.enemy = Ai(points=points, graph=graph, model=ai_model, start_id=1)
        self.enemy.aitarg = self.player  # Set player as the AI's target

        # Add tasks
        self.taskMgr.do_method_later(0.5, self.enemy.updateAi, "UpdateAITask")
        self.taskMgr.add(self.enemy.move_along_path, "MoveAIPathTask")

        # Optional: move player for demo purposes
        self.taskMgr.add(self.move_player, "MovePlayerTask")

    def move_player(self, task):
        pos = self.player.getPos()
        self.player.setPos(pos + Vec3(0.05, 0, 0))  # Move slowly in X
        return task.cont

app = MyApp()
app.run()

