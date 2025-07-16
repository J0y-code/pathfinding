bl_info = {
    "name": "Export PFS A*",
    "author": "Nathan Pflieger-Chakma",
    "version": (1, 2),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > PDS / File > Export > PDS",
    "description": "Export navigation data to PDS format with auto navpoint generation by exporting",
    "category": "Import-Export",
}

import bpy
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import FloatProperty, BoolProperty, StringProperty, PointerProperty
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.io_utils import ExportHelper

# === UI PANEL ===
class PDS_PT_Panel(Panel):
    bl_label = "PDS Tools"
    bl_idname = "PDS_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PFS'

    def draw(self, context):
        layout = self.layout
        props = context.scene.pds_props

        layout.label(text="Paramètres de génération")
        layout.prop(props, "grid_step")
        layout.prop(props, "z_offset")
        layout.prop(props, "area_name")
        layout.prop(props, "ignore_collection")
        layout.prop(props, "bidirectional_links")

        layout.separator()
        layout.label(text="Exporter")
        layout.operator("export_scene.pds")

# === PROPRIÉTÉS ===
class PDSProperties(PropertyGroup):
    grid_step: FloatProperty(name="Grid Step", default=1.0, min=0.1)
    z_offset: FloatProperty(name="Z Offset", default=0.1)
    area_name: StringProperty(name="Area Name", default="level1")
    ignore_collection: StringProperty(name="Ignore Collection", default="obstacles")
    bidirectional_links: BoolProperty(name="Connexions bidirectionnelles", default=True)

# === EXPORT OPERATOR ===
class ExportPDS(Operator, ExportHelper):
    bl_idname = "export_scene.pfs"
    bl_label = "Export to PFS"
    filename_ext = ".pfs"
    filter_glob: StringProperty(default="*.pfs", options={'HIDDEN'})

    def execute(self, context):
        props = context.scene.pds_props
        depsgraph = context.evaluated_depsgraph_get()
        ignore_coll = bpy.data.collections.get(props.ignore_collection) if props.ignore_collection else None

        active = context.active_object
        if not active or active.type != 'MESH':
            self.report({'ERROR'}, "Veuillez sélectionner un mesh comme source de navigation.")
            return {'CANCELLED'}

        eval_obj = active.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        mesh.transform(eval_obj.matrix_world)

        verts = [v.co.copy() for v in mesh.vertices]
        polygons = [p.vertices[:] for p in mesh.polygons]
        bvh_navmesh = BVHTree.FromPolygons(verts, polygons)

        # Génération des points
        points = []
        for i, poly in enumerate(mesh.polygons, start=1):
            center = sum((mesh.vertices[v].co for v in poly.vertices), Vector()) / len(poly.vertices)
            tag = active.get("tag", None)
            points.append((i, center.copy(), tag))

        eval_obj.to_mesh_clear()

        # Récupération des obstacles
        obstacles = []
        for obj in context.visible_objects:
            if obj == active or obj.type != 'MESH':
                continue
            if ignore_coll and obj.name not in ignore_coll.objects:
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

        # Connexions
        links = []
        for a_id, a_loc, _ in points:
            neighbors = sorted(
                [(b_id, b_loc) for b_id, b_loc, _ in points if b_id != a_id],
                key=lambda x: (x[1] - a_loc).length
            )[:8]
            for b_id, b_loc in neighbors:
                direction = (b_loc - a_loc).normalized()
                distance = (b_loc - a_loc).length
                blocked = any(bvh.ray_cast(a_loc, direction, distance)[0] for bvh in obstacles)
                if not blocked:
                    links.append((a_id, b_id))
                    if props.bidirectional_links:
                        links.append((b_id, a_id))

        # Export du fichier PDS
        try:
            with open(self.filepath, 'w') as f:
                f.write("<< Exported by Blender PDS Exporter >>\n")
                f.write(f"<Area> [{props.area_name}] {{\n")
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
                for a, b in links:
                    f.write("    <Relate> {\n")
                    f.write(f"      <Crossroad> {{{a}}}\n")
                    f.write(f"      <Peripheral> {{{b}}}\n")
                    f.write("    }\n")
                f.write("  }\n")
                f.write("}\n")
        except Exception as e:
            self.report({'ERROR'}, f"Erreur d'écriture : {e}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Fichier exporté : {self.filepath}")
        return {'FINISHED'}

# === ENREGISTREMENT ===
classes = [
    PDS_PT_Panel,
    PDSProperties,
    ExportPDS,
]

def menu_func_export(self, context):
    self.layout.operator(ExportPDS.bl_idname, text="PDS Pathfinding (.pfs)")

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.Scene.pds_props = PointerProperty(type=PDSProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    del bpy.types.Scene.pds_props

if __name__ == "__main__":
    register()
