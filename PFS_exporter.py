bl_info = {
    "name": "Export PFS A*",
    "author": "Nathan Pflieger-Chakma / ChatGPT",
    "version": (2, 1),
    "blender": (4, 0, 0),
    "location": "File > Export > PFS Pathfinding",
    "description": "Export navigation data (.pfs) with edge-based and extended adjacency connections",
    "category": "Import-Export",
}

import bpy
from bpy.types import Operator
from bpy.props import StringProperty
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.io_utils import ExportHelper


class ExportPFS(Operator, ExportHelper):
    bl_idname = "export_scene.pfs"
    bl_label = "Export to PFS"
    filename_ext = ".pfs"
    filter_glob: StringProperty(default="*.pfs", options={'HIDDEN'})

    def execute(self, context):
        depsgraph = context.evaluated_depsgraph_get()
        ignore_collection = bpy.data.collections.get("obstacles")
        bidirectional_links = True

        active = context.active_object
        if not active or active.type != 'MESH':
            self.report({'ERROR'}, "Please select a mesh as navigation source.")
            return {'CANCELLED'}

        # --- Préparer navmesh
        eval_obj = active.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        mesh.transform(eval_obj.matrix_world)

        # Générer les points (centre de faces)
        points = []
        for i, poly in enumerate(mesh.polygons, start=1):
            center = sum((mesh.vertices[v].co for v in poly.vertices), Vector()) / len(poly.vertices)
            tag = active.get("tag", None)
            points.append((i, center.copy(), tag))

        # --- Obstacles
        obstacles = []
        for obj in context.visible_objects:
            if obj == active or obj.type != 'MESH':
                continue
            if ignore_collection and obj in ignore_collection.objects:
                continue

            obs_eval = obj.evaluated_get(depsgraph)
            obs_mesh = obs_eval.to_mesh()
            obs_mesh.transform(obs_eval.matrix_world)
            obs_bvh = BVHTree.FromPolygons(
                [v.co.copy() for v in obs_mesh.vertices],
                [p.vertices[:] for p in obs_mesh.polygons]
            )
            obstacles.append(obs_bvh)
            obs_eval.to_mesh_clear()

        # --- Connexions par arêtes
        links = set()
        for poly in mesh.polygons:
            a_id = poly.index + 1
            a_loc = sum((mesh.vertices[v].co for v in poly.vertices), Vector()) / len(poly.vertices)

            for edge in poly.edge_keys:
                adjacent_faces = [
                    p for p in mesh.polygons
                    if p.index != poly.index and edge[0] in p.vertices and edge[1] in p.vertices
                ]

                for adj in adjacent_faces:
                    b_id = adj.index + 1
                    b_loc = sum((mesh.vertices[v].co for v in adj.vertices), Vector()) / len(adj.vertices)
                    direction = (b_loc - a_loc).normalized()
                    distance = (b_loc - a_loc).length

                    blocked = any(bvh.ray_cast(a_loc, direction, distance)[0] for bvh in obstacles)
                    if not blocked:
                        links.add((a_id, b_id, distance))
                        if bidirectional_links:
                            links.add((b_id, a_id, distance))

        # --- Connexions adjacentes (second niveau)
        adjacency = {}
        for a, b, dist in links:
            adjacency.setdefault(a, set()).add(b)

        extra_links = set()
        for a in adjacency:
            neighbors = list(adjacency[a])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    b, c = neighbors[i], neighbors[j]
                    common = adjacency.get(b, set()) & adjacency.get(c, set())
                    for d in common:
                        if d != a and d not in adjacency[a]:
                            a_loc = next(loc for pid, loc, _ in points if pid == a)
                            d_loc = next(loc for pid, loc, _ in points if pid == d)
                            dist = (d_loc - a_loc).length
                            extra_links.add((a, d, dist))
                            if bidirectional_links:
                                extra_links.add((d, a, dist))

        links |= extra_links

        # --- Export fichier
        try:
            with open(self.filepath, 'w') as f:
                f.write("<< Exported by Blender PFS Exporter >>\n")
                area_name = active.name
                f.write(f"<Area> [{area_name}] {{\n")
                f.write("  <Group_of_Points> {\n")
                for pid, loc, tag in points:
                    f.write(f"    <Point> {pid} {{\n")
                    f.write(f"      <x> {{{loc.x}}}\n")
                    f.write(f"      <y> {{{loc.y}}}\n")
                    f.write(f"      <z> {{{loc.z}}}\n")
                    if tag:
                        f.write(f"      <Tag> [{tag}]\n")
                    f.write("    }\n")
                f.write("  }\n")
                f.write("  <Connections> {\n")
                for a, b, dist in sorted(links):
                    f.write("    <Relate> {\n")
                    f.write(f"      <Crossroad> {{{a}}}\n")
                    f.write(f"      <Peripheral> {{{b}}}\n")
                    #f.write(f"      <Cost> {{{dist}}}\n")
                    f.write("    }\n")
                f.write("  }\n")
                f.write("}\n")
        except Exception as e:
            self.report({'ERROR'}, f"Write error: {e}")
            return {'CANCELLED'}

        eval_obj.to_mesh_clear()
        self.report({'INFO'}, f"File exported: {self.filepath}")
        return {'FINISHED'}


# --- ENREGISTREMENT
def menu_func_export(self, context):
    self.layout.operator(ExportPFS.bl_idname, text="PFS Pathfinding (.pfs)")

def register():
    bpy.utils.register_class(ExportPFS)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(ExportPFS)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
