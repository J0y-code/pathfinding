bl_info = {
    "name": "Export PFS SubAreas",
    "author": "J0ytheC0de",
    "version": (2, 0),
    "blender": (4, 0, 0),
    "location": "File > Export > PFS Pathfinding",
    "description": "Export navmesh grouped by sub-areas (material-based) with edge and visibility links",
    "category": "Import-Export",
}

import bpy
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, FloatProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector
from mathutils.bvhtree import BVHTree

class ExportPFSSubAreasUnified(Operator, ExportHelper):
    bl_idname = "export_scene.pfs_subareas_unified"
    bl_label = "Export to PFS (SubAreas, unified syntax)"
    filename_ext = ".pfs"
    filter_glob: StringProperty(default="*.pfs", options={'HIDDEN'})

    use_edge_links: BoolProperty(
        name="Edge Links",
        description="Relier les faces adjacentes par arêtes",
        default=True
    )
    use_visibility_links: BoolProperty(
        name="Visibility Links",
        description="Relier les faces visibles sans obstacle (raycast)",
        default=False
    )
    visibility_max_distance: FloatProperty(
        name="Max Distance",
        description="Distance maximale pour créer un lien de visibilité",
        default=5.0,
        min=0.1,
        max=1000.0
    )

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Please select a mesh.")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        mesh.transform(eval_obj.matrix_world)

        # --- SubAreas par matériau
        subareas = {}
        face_material_map = {}
        for poly in mesh.polygons:
            mat = obj.material_slots[poly.material_index].name if obj.material_slots else "Default"
            subareas.setdefault(mat, []).append(poly.index)
            face_material_map[poly.index] = mat

        # --- Points
        points = {}
        for poly in mesh.polygons:
            center = sum((mesh.vertices[v].co for v in poly.vertices), Vector()) / len(poly.vertices)
            points[poly.index + 1] = {
                'pos': center.copy(),
                'material': face_material_map[poly.index]
            }

        # --- Préparer obstacles
        ignore_collection = bpy.data.collections.get("obstacles")
        obstacles = []
        for obj_ in context.visible_objects:
            if obj_ == obj or obj_.type != 'MESH':
                continue
            if ignore_collection and obj_ in ignore_collection.objects:
                continue

            obs_eval = obj_.evaluated_get(depsgraph)
            obs_mesh = obs_eval.to_mesh()
            obs_mesh.transform(obs_eval.matrix_world)
            obs_bvh = BVHTree.FromPolygons(
                [v.co.copy() for v in obs_mesh.vertices],
                [p.vertices[:] for p in obs_mesh.polygons]
            )
            obstacles.append(obs_bvh)
            obs_eval.to_mesh_clear()

        # --- Connexions
        links_internal = {mat: set() for mat in subareas}
        links_external = set()

        # --- Liens par arêtes
        if self.use_edge_links:
            for poly in mesh.polygons:
                pid = poly.index + 1
                mat = face_material_map[poly.index]

                for edge in poly.edge_keys:
                    adjacent = [
                        p for p in mesh.polygons
                        if p.index != poly.index and edge[0] in p.vertices and edge[1] in p.vertices
                    ]
                    for adj in adjacent:
                        adj_id = adj.index + 1
                        adj_mat = face_material_map[adj.index]

                        a_loc = points[pid]['pos']
                        b_loc = points[adj_id]['pos']
                        direction = (b_loc - a_loc).normalized()
                        distance = (b_loc - a_loc).length

                        blocked = any(bvh.ray_cast(a_loc, direction, distance)[0] for bvh in obstacles)
                        if not blocked:
                            if mat == adj_mat:
                                links_internal[mat].add((pid, adj_id))
                            else:
                                links_external.add((pid, adj_id))

        # --- Liens par visibilité
        if self.use_visibility_links:
            point_items = list(points.items())
            for i, (pid_a, data_a) in enumerate(point_items):
                for pid_b, data_b in point_items[i+1:]:
                    a_loc = data_a['pos']
                    b_loc = data_b['pos']
                    direction = (b_loc - a_loc).normalized()
                    distance = (b_loc - a_loc).length

                    if distance > self.visibility_max_distance:
                        continue

                    blocked = any(bvh.ray_cast(a_loc, direction, distance)[0] for bvh in obstacles)
                    if not blocked:
                        mat_a = data_a['material']
                        mat_b = data_b['material']
                        if mat_a == mat_b:
                            links_internal[mat_a].add((pid_a, pid_b))
                            links_internal[mat_a].add((pid_b, pid_a))
                        else:
                            links_external.add((pid_a, pid_b))
                            links_external.add((pid_b, pid_a))

        # --- Export fichier
        try:
            with open(self.filepath, 'w') as f:
                f.write("<< Exported by Blender PFS Exporter >>\n")
                f.write(f"<Area> [{obj.name}] {{\n")

                for mat, face_ids in subareas.items():
                    f.write(f"  <SubArea> [{mat}] {{\n")
                    f.write("    <Group_of_Points> {\n")
                    for fid in face_ids:
                        pid = fid + 1
                        p = points[pid]
                        f.write(f"      <Point> {pid} {{\n")
                        f.write(f"        <x> {{{p['pos'].x}}}\n")
                        f.write(f"        <y> {{{p['pos'].y}}}\n")
                        f.write(f"        <z> {{{p['pos'].z}}}\n")
                        f.write("      }\n")
                    f.write("    }\n")

                    f.write("    <Connections> {\n")
                    for a, b in sorted(links_internal[mat]):
                        f.write("      <Relate> {\n")
                        f.write(f"        <Crossroad> {{{a}}}\n")
                        f.write(f"        <Peripheral> {{{b}}}\n")
                        f.write("      }\n")
                    f.write("    }\n")
                    f.write("  }\n")

                if links_external:
                    f.write("  <Connections> {\n")
                    for a, b in sorted(links_external):
                        f.write("    <Relate> {\n")
                        f.write(f"      <Crossroad> {{{a}}}\n")
                        f.write(f"      <Peripheral> {{{b}}}\n")
                        f.write("    }\n")
                    f.write("  }\n")

                f.write("}\n")

        except Exception as e:
            self.report({'ERROR'}, f"Write error: {e}")
            return {'CANCELLED'}

        eval_obj.to_mesh_clear()
        self.report({'INFO'}, f"File exported: {self.filepath}")
        return {'FINISHED'}

# --- ENREGISTREMENT ---
def menu_func_export(self, context):
    self.layout.operator(ExportPFSSubAreasUnified.bl_idname, text="PFS Pathfinding with SubAreas (unified) (.pfs)")

def register():
    bpy.utils.register_class(ExportPFSSubAreasUnified)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(ExportPFSSubAreasUnified)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
