import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import tempfile
import os
import trimesh
import pandas as pd
import plotly.graph_objects as go

import shutil

if shutil.which("blender") is not None:
    st.info("Blender is available. 3D model generation will be faster ")
else:
    st.warning("Blender is NOT available. 3D model generation may be slower and less accurate.")

# Constants for 3D model
WALL_HEIGHT = 2.4
WALL_THICKNESS = 0.2
DOOR_HEIGHT = 2.0
WINDOW_HEIGHT = 1.2
WINDOW_BOTTOM = 0.9
DOOR_CUTOUT_THICKNESS = 5.0  # meters, much larger than any wall

def create_3d_model(binary_image, image_shape, wall_color, wall_height_m=4.0, include_floor=False, line_thickness_px=1, floor_color="#8B4513"):
    scale_x, scale_y = 15.0 / image_shape[1], 15.0 / image_shape[0]
    vertices, faces, colors, vert_idx = [], [], [], 0
    wall_mask = (binary_image == 0).astype(np.uint8)
    if include_floor:
        ys, xs = np.where(wall_mask == 1)
        if xs.size and ys.size:
            margin = 20
            min_x, max_x = max(0, xs.min() - margin), min(image_shape[1] - 1, xs.max() + margin)
            min_y, max_y = max(0, ys.min() - margin), min(image_shape[0] - 1, ys.max() + margin)
            x1_m, x2_m = min_x * scale_x, (max_x + 1) * scale_x
            y1_m, y2_m = (image_shape[0] - (max_y + 1)) * scale_y, (image_shape[0] - min_y) * scale_y
            floor_verts = [[x1_m, y1_m, 0.0], [x2_m, y1_m, 0.0], [x2_m, y2_m, 0.0], [x1_m, y2_m, 0.0]]
            vertices.extend(floor_verts)
            faces.extend([[vert_idx, vert_idx + 1, vert_idx + 2], [vert_idx, vert_idx + 2, vert_idx + 3]])
            colors.extend([floor_color, floor_color])
            vert_idx += 4
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt = cnt.squeeze()
        if cnt.ndim != 2 or len(cnt) < 2: continue
        for i in range(len(cnt)):
            p1, p2 = cnt[i], cnt[(i + 1) % len(cnt)]
            if np.linalg.norm(p1 - p2) < 1: continue
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = np.hypot(dx, dy)
            if length == 0: continue
            nx, ny = -dy / length, dx / length
            offset = np.array([nx, ny]) * (line_thickness_px / 2.0)
            pa, pb, pc, pd = p1 + offset, p2 + offset, p2 - offset, p1 - offset
            def px_to_m(pt): return [pt[0] * scale_x, (image_shape[0] - pt[1]) * scale_y]
            pa_m, pb_m, pc_m, pd_m = px_to_m(pa), px_to_m(pb), px_to_m(pc), px_to_m(pd)
            v = [[*pa_m, 0.0], [*pb_m, 0.0], [*pc_m, 0.0], [*pd_m, 0.0],
                 [*pa_m, wall_height_m], [*pb_m, wall_height_m], [*pc_m, wall_height_m], [*pd_m, wall_height_m]]
            vertices.extend(v)
            base = vert_idx
            faces.extend([[base, base+1, base+5], [base, base+5, base+4], [base+1, base+2, base+6], [base+1, base+6, base+5],
                          [base+2, base+3, base+7], [base+2, base+7, base+6], [base+3, base, base+4], [base+3, base+4, base+7]])
            colors.extend([wall_color] * 8)
            vert_idx += 8
    return np.asarray(vertices), faces, colors

def plot_3d_model(vertices, faces, colors):
    """Create 3D visualization using Plotly"""
    if len(vertices) == 0:
        return None

    if isinstance(vertices, list):
        vertices = np.array(vertices)

    # Create a list of meshes, one for each color
    unique_colors = list(set(colors))
    meshes = []
    
    for color in unique_colors:
        # Get indices of faces with this color
        color_indices = [i for i, c in enumerate(colors) if c == color]
        if not color_indices:
            continue
        
        # Create lists of vertices for this color
        i, j, k = [], [], []
        for idx in color_indices:
            face = faces[idx]
            i.append(face[0])
            j.append(face[1])
            k.append(face[2])
        
        # Use provided color directly
        mesh_color = color
        
        # Create mesh for this color with enhanced lighting properties
        mesh = go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=i, j=j, k=k,
            color=mesh_color, opacity=1.0,
            lighting=dict(ambient=0.9, diffuse=0.8, specular=0.05, roughness=0.2),
            flatshading=True
        )
        meshes.append(mesh)
    
    fig = go.Figure(data=meshes)
    fig.update_layout(
        title="3D Floor Plan Model",
        scene=dict(
            xaxis_title="X (meters)", yaxis_title="Y (meters)", zaxis_title="Z (meters)",
            aspectmode='data',
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=2, y=2, z=2)),
            bgcolor='white'
        ),
        width=800, height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def generate_mtl_file(colors):
    """Generate MTL file content for materials"""
    mtl_content = "# Materials for Floor Plan 3D Model\n\n"
    for color in set(colors):
        r = int(color[1:3], 16) / 255.0
        g = int(color[3:5], 16) / 255.0
        b = int(color[5:7], 16) / 255.0
        material_name = color.replace('#', 'color_')
        mtl_content += f"newmtl {material_name}\n"
        mtl_content += f"Kd {r:.6f} {g:.6f} {b:.6f}\n"
        mtl_content += "illum 1\n\n"
    return mtl_content

def generate_obj_file_with_groups(vertices, faces, colors, face_types):
    obj_content = "# Floor Plan 3D Model\n"
    obj_content += "mtllib materials.mtl\n\n"
    for v in vertices:
        obj_content += f"v {v[0]:.6f} {v[2]:.6f} {v[1]:.6f}\n"
    group_map = {}
    for i, (ftype, color) in enumerate(zip(face_types, colors)):
        key = (ftype, color)
        if key not in group_map: group_map[key] = []
        group_map[key].append(i)
    for (ftype, color), idxs in group_map.items():
        obj_content += f"g {ftype}\nusemtl {color.replace('#', 'color_')}\n"
        for i in idxs:
            face = faces[i]
            obj_content += f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
    return obj_content

def save_3d_model_with_groups(vertices, faces, colors, face_types):
    try:
        # Generate OBJ and MTL content in memory only, do not save to disk
        obj_content = generate_obj_file_with_groups(vertices, faces, colors, face_types)
        mtl_content = generate_mtl_file(colors)
        # Use a timestamp for download filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return obj_content, mtl_content, timestamp
    except Exception as e:
        st.error(f"Error generating model: {e}")
        return None, None, None

def create_download_section_with_groups(vertices, faces, colors, face_types):
    st.markdown("---"); st.subheader("📥 Download 3D Model")
    obj_content, mtl_content, timestamp = save_3d_model_with_groups(vertices, faces, colors, face_types)
    if obj_content and mtl_content:
        # Provide both OBJ and MTL as downloads
        st.download_button(
            label=" .Obj File ",
            data=obj_content,
            file_name=f"model_{timestamp}.obj",
            mime="application/x-tgif"
        )
        st.download_button(
            label=" .Mtl File ",
            data=mtl_content,
            file_name="materials.mtl",
            mime="text/plain"
        )
    #st.success(f"3D model ready for download.")

def preprocess_image(image, kernel_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 400: mask[labels == i] = 255
    return cv2.bitwise_not(mask)

@st.cache_resource
def load_model(path): return YOLO(path)

def create_walls_from_boxes(detected_walls, image_shape, wall_color, detected_doors=None):
    """Creates a 3D wall mesh from detected bounding boxes, splitting wall segments at door locations."""
    all_vertices, all_faces, all_colors, all_face_types = [], [], [], []
    scale_x, scale_y = 15.0 / image_shape[1], 15.0 / image_shape[0]
    base_idx = 0
    def boxes_overlap(box1, box2, margin=5):
        ax1, ay1, ax2, ay2 = box1
        bx1, by1, bx2, by2 = box2
        return not (ax2 < bx1 - margin or ax1 > bx2 + margin or ay2 < by1 - margin or ay1 > by2 + margin)
    for wall_box in detected_walls:
        x1, y1, x2, y2 = [int(round(c)) for c in wall_box]
        wall_segments = [[x1, y1, x2, y2]]
        # Split wall at each overlapping door
        if detected_doors:
            for door_box in detected_doors:
                dx1, dy1, dx2, dy2 = [int(round(c)) for c in door_box]
                # Only split if the door overlaps this wall
                if boxes_overlap([x1, y1, x2, y2], [dx1, dy1, dx2, dy2]):
                    new_segments = []
                    for seg in wall_segments:
                        sx1, sy1, sx2, sy2 = seg
                        # Horizontal wall
                        if abs(sy1 - sy2) < abs(sx1 - sx2):
                            # If door is within wall segment horizontally
                            if dx1 > sx1 and dx2 < sx2:
                                new_segments.append([sx1, sy1, dx1, sy2])
                                new_segments.append([dx2, sy1, sx2, sy2])
                            elif dx1 > sx1 and dx1 < sx2:
                                new_segments.append([sx1, sy1, dx1, sy2])
                            elif dx2 > sx1 and dx2 < sx2:
                                new_segments.append([dx2, sy1, sx2, sy2])
                            else:
                                # Door covers or is outside this segment
                                if dx2 <= sx1 or dx1 >= sx2:
                                    new_segments.append(seg)
                        else:  # Vertical wall
                            if dy1 > sy1 and dy2 < sy2:
                                new_segments.append([sx1, sy1, sx2, dy1])
                                new_segments.append([sx1, dy2, sx2, sy2])
                            elif dy1 > sy1 and dy1 < sy2:
                                new_segments.append([sx1, sy1, sx2, dy1])
                            elif dy2 > sy1 and dy2 < sy2:
                                new_segments.append([sx1, dy2, sx2, sy2])
                            else:
                                if dy2 <= sy1 or dy1 >= sy2:
                                    new_segments.append(seg)
                    wall_segments = new_segments if new_segments else wall_segments
        for seg in wall_segments:
            sx1, sy1, sx2, sy2 = seg
            width_px, depth_px = sx2 - sx1, sy2 - sy1
            if abs(width_px) < 1 or abs(depth_px) < 1:
                continue  # skip degenerate segments
            is_horizontal = abs(width_px) > abs(depth_px)
            extents = [abs(width_px) * scale_x, WALL_THICKNESS, WALL_HEIGHT] if is_horizontal else [WALL_THICKNESS, abs(depth_px) * scale_y, WALL_HEIGHT]
            wall_part = trimesh.primitives.Box(extents=extents)
            px, py, pz = (sx1 + sx2) / 2.0 * scale_x, (image_shape[0] - (sy1 + sy2) / 2.0) * scale_y, WALL_HEIGHT / 2.0
            wall_part.apply_translation([px, py, pz])
            num_verts = len(wall_part.vertices)
            all_faces.extend([[f + base_idx for f in face] for face in wall_part.faces])
            all_vertices.extend(wall_part.vertices.tolist())
            all_colors.extend([wall_color] * len(wall_part.faces))
            all_face_types.extend(['walls'] * len(wall_part.faces))
            base_idx += num_verts
    return all_vertices, all_faces, all_colors, all_face_types

def cutout_openings_from_walls(wall_vertices, wall_faces, detected_doors, detected_windows, image_shape):
    """Subtracts door and window openings from a wall mesh."""
    if not wall_vertices or (not detected_doors and not detected_windows):
        return wall_vertices, wall_faces

    wall_mesh = trimesh.Trimesh(vertices=wall_vertices, faces=wall_faces)
    scale_x, scale_y = 15.0 / image_shape[1], 15.0 / image_shape[0]
    opening_meshes = []

    # Create boxes for door openings
    for box in detected_doors or []:
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        width_m = abs(x2 - x1) * scale_x
        is_horizontal = width_m > abs(y2 - y1) * scale_y
        extents = [width_m, DOOR_CUTOUT_THICKNESS, DOOR_HEIGHT] if is_horizontal else [DOOR_CUTOUT_THICKNESS, abs(y2-y1)*scale_y, DOOR_HEIGHT]
        px, py, pz = (x1 + x2) / 2.0 * scale_x, (image_shape[0] - (y1 + y2) / 2.0) * scale_y, DOOR_HEIGHT / 2.0
        opening = trimesh.primitives.Box(extents=extents)
        opening.apply_translation([px, py, pz])
        opening_meshes.append(opening)

    # Create boxes for window openings
    for box in detected_windows or []:
        x1, y1, x2, y2 = [int(round(c)) for c in box]
        width_m = abs(x2 - x1) * scale_x
        is_horizontal = width_m > abs(y2 - y1) * scale_y
        extents = [width_m, WALL_THICKNESS * 1.5, WINDOW_HEIGHT] if is_horizontal else [WALL_THICKNESS * 1.5, abs(y2-y1)*scale_y, WINDOW_HEIGHT]
        opening = trimesh.primitives.Box(extents=extents)
        px, py, pz = (x1 + x2) / 2.0 * scale_x, (image_shape[0] - (y1 + y2) / 2.0) * scale_y, WINDOW_BOTTOM + WINDOW_HEIGHT / 2.0
        opening.apply_translation([px, py, pz])
        opening_meshes.append(opening)

    if opening_meshes:
        combined_openings = trimesh.util.concatenate(opening_meshes)
        blender_available = shutil.which("blender") is not None
        try:
            if blender_available:
                wall_mesh = wall_mesh.difference(combined_openings, engine='blender')
                if 'st' in globals():
                    st.info("Using Blender engine for boolean operations (fastest).")
            else:
                wall_mesh = wall_mesh.difference(combined_openings)
                if 'st' in globals():
                    st.warning("Using trimesh's internal engine for boolean operations. For complex floorplans, this may be slow or may not work. For best results, run locally with Blender installed.")
        except Exception as e:
            if 'st' in globals():
                st.error("Boolean cutout failed. This is a complex architecture for trimesh. For best results, run locally with Blender installed. Openings will be skipped.")
            return wall_vertices, wall_faces  # Return original if difference fails

    return wall_mesh.vertices.tolist(), wall_mesh.faces.tolist()

def add_wall_segments_below_windows(vertices, faces, colors, face_types, detected_windows, image_shape, wall_color):
    """Creates and adds wall segments directly below detected windows."""
    if not detected_windows:
        return vertices, faces, colors, face_types

    main_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
    scale_x = 15.0 / image_shape[1]
    scale_y = 15.0 / image_shape[0]

    for box in detected_windows:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        width_m = abs(x2 - x1) * scale_x
        height_m = abs(y2 - y1) * scale_y
        is_horizontal = width_m > height_m

        extents = [width_m, WALL_THICKNESS, WINDOW_BOTTOM] if is_horizontal else [WALL_THICKNESS, width_m, WINDOW_BOTTOM]
        wall_segment = trimesh.primitives.Box(extents=extents)
        
        px = (x1 + x2) / 2.0 * scale_x
        py = (image_shape[0] - (y1 + y2) / 2.0) * scale_y
        pz = WINDOW_BOTTOM / 2.0
        
        wall_segment.apply_translation([px, py, pz])
        main_mesh = trimesh.util.concatenate([main_mesh, wall_segment])

    n_old_faces = len(faces)
    n_new_faces = len(main_mesh.faces) - n_old_faces
    colors_extended = list(colors) + [wall_color] * n_new_faces
    face_types_extended = list(face_types) + ['walls'] * n_new_faces

    return main_mesh.vertices.tolist(), main_mesh.faces.tolist(), colors_extended, face_types_extended

def add_wall_segments_above_openings(vertices, faces, colors, face_types, detected_boxes, opening_top_z, image_shape, wall_color):
    """Creates and adds wall segments above openings like doors and windows."""
    if not detected_boxes:
        return vertices, faces, colors, face_types

    main_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
    scale_x = 15.0 / image_shape[1]
    scale_y = 15.0 / image_shape[0]

    for box in detected_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        width_m = abs(x2 - x1) * scale_x
        height_m = abs(y2 - y1) * scale_y
        is_horizontal = width_m > height_m
        
        segment_height = WALL_HEIGHT - opening_top_z
        if segment_height <= 0: continue

        extents = [width_m, WALL_THICKNESS, segment_height] if is_horizontal else [WALL_THICKNESS, width_m, segment_height]
        wall_segment = trimesh.primitives.Box(extents=extents)
        
        px = (x1 + x2) / 2.0 * scale_x
        py = (image_shape[0] - (y1 + y2) / 2.0) * scale_y
        pz = opening_top_z + segment_height / 2.0
        
        wall_segment.apply_translation([px, py, pz])
        main_mesh = trimesh.util.concatenate([main_mesh, wall_segment])

    n_old_faces = len(faces)
    n_new_faces = len(main_mesh.faces) - n_old_faces
    colors_extended = list(colors) + [wall_color] * n_new_faces
    face_types_extended = list(face_types) + ['walls'] * n_new_faces

    return main_mesh.vertices.tolist(), main_mesh.faces.tolist(), colors_extended, face_types_extended

def merge_obj_at_positions(vertices, faces, colors, face_types, detected_boxes, obj_path, color_hex, face_type_label, image_shape, z_base=0.0, z_height=2.0, is_window=False):
    scale_x = 15.0 / image_shape[1]
    scale_y = 15.0 / image_shape[0]
    main_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
    for box in detected_boxes:
        try:
            obj_mesh = trimesh.load(obj_path, process=False)
            x1, y1, x2, y2 = [int(round(c)) for c in box]
            width_m = abs(x2 - x1) * scale_x
            height_m = abs(y2 - y1) * scale_y
            is_horizontal = width_m > height_m
            # Center the object
            obj_mesh.apply_translation(-obj_mesh.centroid)
            bbox = obj_mesh.bounding_box.extents
            # --- Orientation logic ---
            if is_window:
        
                obj_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
                bbox = obj_mesh.bounding_box.extents
                # Now, rotate to align with wall orientation
                if not is_horizontal:

                    obj_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
                    bbox = obj_mesh.bounding_box.extents
            else:
                # Doors: rotate to align with wall orientation
                if not is_horizontal:
                    obj_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
                    bbox = obj_mesh.bounding_box.extents
            # --- Scaling ---
            mesh_width = max(bbox[0], bbox[1])
            mesh_thickness = min(bbox[0], bbox[1])
            mesh_height = bbox[2]
            scale_w = width_m / mesh_width if mesh_width > 0 else 1.0
            scale_t = WALL_THICKNESS / mesh_thickness if mesh_thickness > 0 else 1.0
            scale_h = (z_height - z_base) / mesh_height if mesh_height > 0 else 1.0
            obj_mesh.apply_scale([scale_w, scale_t, scale_h])
            # --- Placement ---
            px = (x1 + x2) / 2.0 * scale_x
            py = (image_shape[0] - (y1 + y2) / 2.0) * scale_y
            pz = z_base + (z_height - z_base) / 2.0
            obj_mesh.apply_translation([px, py, pz])
            main_mesh = trimesh.util.concatenate([main_mesh, obj_mesh])
        except Exception as e:
            st.warning(f"Could not load/place {obj_path}: {e}")
    n_old_faces = len(faces)
    n_new_faces = len(main_mesh.faces) - n_old_faces
    colors_extended = list(colors) + [color_hex] * n_new_faces
    face_types_extended = list(face_types) + [face_type_label] * n_new_faces
    return main_mesh.vertices.tolist(), main_mesh.faces.tolist(), colors_extended, face_types_extended

st.title('2D Floorplan to 3D ')
st.sidebar.title('Controls')

uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
general_conf = st.sidebar.slider('Model Confidence', 0.1, 0.9, 0.2, 0.01)

st.sidebar.subheader('Class Confidence')
door_conf = st.sidebar.number_input('Door', 0.1, 0.9, 0.1, 0.01)
wall_conf = st.sidebar.number_input('Wall', 0.1, 0.9, 0.1, 0.01)
window_conf = st.sidebar.number_input('Window', 0.1, 0.9, 0.1, 0.01)

st.sidebar.subheader('3D Model Generation')
wall_generation_method = st.sidebar.selectbox('Wall Generation Method', ['Detection', 'Ruled Based'])

wall_color = st.sidebar.color_picker('Wall Color', value='#A9A9A9')
door_color = st.sidebar.color_picker('Door Color', value='#FF0000')
window_color = st.sidebar.color_picker('Window Color', value='#00BFFF')
base_color = st.sidebar.color_picker('Base (Floor) Color', value='#8B4513')
apply_color = st.sidebar.button("Apply Color")

if 'applied_wall_color' not in st.session_state:
    st.session_state['applied_wall_color'] = wall_color
if 'applied_door_color' not in st.session_state:
    st.session_state['applied_door_color'] = door_color
if 'applied_window_color' not in st.session_state:
    st.session_state['applied_window_color'] = window_color
if 'applied_base_color' not in st.session_state:
    st.session_state['applied_base_color'] = base_color

if apply_color:
    st.session_state['applied_wall_color'] = wall_color
    st.session_state['applied_door_color'] = door_color
    st.session_state['applied_window_color'] = window_color
    st.session_state['applied_base_color'] = base_color

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(uploaded_file.read()); img_path = tmp_file.name

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.subheader('Original Image'); st.image(image_rgb, use_container_width=True)

    model = load_model('best.pt')
    results = model(image, conf=general_conf)
    result = results[0]
    st.subheader('Model Prediction'); st.image(result.plot(), caption='Model Prediction', use_container_width=True)

    class_colors = {'door': door_color, 'wall': wall_color, 'window': window_color}
    conf_thresholds = {"door": door_conf, "wall": wall_conf, "window": window_conf}
    
    overlay_img = Image.fromarray(image_rgb).convert("RGBA")
    detected_doors, detected_windows, detected_walls = [], [], []
    
    # Filter boxes by confidence and store rounded coordinates for both overlay and 3D
    if result.boxes:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            coords = [int(round(c)) for c in box.xyxy[0].tolist()]
            if confidence >= conf_thresholds.get(class_name, 1.0):
                if class_name == "door": detected_doors.append(coords)
                elif class_name == "window": detected_windows.append(coords)
                elif class_name == "wall": detected_walls.append(coords)
    # Draw overlays using only filtered boxes
    for coords in detected_walls:
        overlay = Image.new('RGBA', overlay_img.size)
        draw = ImageDraw.Draw(overlay)
        color_with_alpha = class_colors.get("wall") + '80'
        draw.rectangle(coords, fill=color_with_alpha)
        overlay_img = Image.alpha_composite(overlay_img, overlay)
    for coords in detected_doors:
        overlay = Image.new('RGBA', overlay_img.size)
        draw = ImageDraw.Draw(overlay)
        color_with_alpha = class_colors.get("door") + '80'
        draw.rectangle(coords, fill=color_with_alpha)
        overlay_img = Image.alpha_composite(overlay_img, overlay)
    for coords in detected_windows:
        overlay = Image.new('RGBA', overlay_img.size)
        draw = ImageDraw.Draw(overlay)
        color_with_alpha = class_colors.get("window") + '80'
        draw.rectangle(coords, fill=color_with_alpha)
        overlay_img = Image.alpha_composite(overlay_img, overlay)

    st.subheader('Detections Overlay'); st.image(overlay_img, use_container_width=True)

    wall_color = st.session_state['applied_wall_color']
    door_color = st.session_state['applied_door_color']
    window_color = st.session_state['applied_window_color']
    base_color = st.session_state['applied_base_color']
    vertices, faces, colors, face_types = [], [], [], []

    if wall_generation_method == 'Ruled Based':
        st.sidebar.subheader('Ruled Based Processing')
        kernel_size = st.sidebar.slider('Kernel Size', 3, 50, 7, 2)
        wall_mask = preprocess_image(image, kernel_size)
        img_h, img_w, _ = image.shape
        scale_y = 15.0 / img_h
        for box in detected_doors:
            x1, _, x2, _ = [int(round(c)) for c in box]
            wall_mask[img_h - int(DOOR_HEIGHT / scale_y):img_h, x1:x2] = 255
        for box in detected_windows:
            x1, _, x2, _ = [int(round(c)) for c in box]
            wall_mask[img_h - int((WINDOW_BOTTOM + WINDOW_HEIGHT)/scale_y) : img_h - int(WINDOW_BOTTOM/scale_y), x1:x2] = 255
        st.subheader('Final Wall Mask'); st.image(wall_mask, use_container_width=True)
        # --- Add base (floor) as a separate mesh and group ---
        verts, fcs, cols = create_3d_model(wall_mask, image.shape, wall_color, wall_height_m=WALL_HEIGHT,
                                                  include_floor=True, line_thickness_px=WALL_THICKNESS * (image.shape[1] / 15.0), floor_color=base_color)
        # Identify which faces are the base (floor): first two faces if include_floor=True
        if include_floor := True:
            base_face_types = ['base'] * 2
            wall_face_types = ['walls'] * (len(fcs) - 2)
            face_types = base_face_types + wall_face_types
        else:
            face_types = ['walls'] * len(fcs)
        vertices, faces, colors = verts, fcs, cols
    else: # Detection-based
        if detected_walls:
            #st.info("Generating walls from detected bounding boxes.")
            vertices, faces, colors, face_types = create_walls_from_boxes(detected_walls, image.shape, wall_color, detected_doors=detected_doors)
            # Always perform a boolean cutout for doors and windows after wall creation
            if vertices:
                #st.info("Cutting out openings for doors and windows (guaranteed cutout)...")
                vertices, faces = cutout_openings_from_walls(vertices, faces, detected_doors, detected_windows, image.shape)
                colors = [wall_color] * len(faces)
                face_types = ['walls'] * len(faces)
        
            if faces:
                img_h, img_w, _ = image.shape
                margin = 20
                x1_m, x2_m = max(0, 0 - margin) * (15.0 / img_w), min(img_w - 1, img_w - 1 + margin) * (15.0 / img_w)
                y1_m, y2_m = max(0, 0 - margin) * (15.0 / img_h), min(img_h - 1, img_h - 1 + margin) * (15.0 / img_h)
                floor_verts = [[x1_m, y1_m, 0.0], [x2_m, y1_m, 0.0], [x2_m, y2_m, 0.0], [x1_m, y2_m, 0.0]]
                base_idx = len(vertices)
                vertices.extend(floor_verts)
                faces.append([base_idx, base_idx + 1, base_idx + 2])
                faces.append([base_idx, base_idx + 2, base_idx + 3])
                colors.extend([base_color, base_color])
                face_types.extend(['base', 'base'])
        else:
            st.warning("No walls detected. Cannot generate 3D model using 'Detection' method.")

    # Merge doors
    if detected_doors:
        vertices, faces, colors, face_types = merge_obj_at_positions(
            vertices, faces, colors, face_types, detected_doors,
            obj_path="door.obj", color_hex=door_color, face_type_label='doors',
            image_shape=image.shape, z_base=0.0, z_height=DOOR_HEIGHT
        )
    # Merge windows
    if detected_windows:
        vertices, faces, colors, face_types = merge_obj_at_positions(
            vertices, faces, colors, face_types, detected_windows,
            obj_path="window.obj", color_hex=window_color, face_type_label='windows',
            image_shape=image.shape, z_base=WINDOW_BOTTOM, z_height=WINDOW_BOTTOM+WINDOW_HEIGHT, is_window=True
        )
    
    # Add wall segments below windows
    if detected_windows:
        vertices, faces, colors, face_types = add_wall_segments_below_windows(
            vertices, faces, colors, face_types, detected_windows, image.shape, wall_color
        )
    
    # Add wall segments above doors and windows
    if detected_doors:
        vertices, faces, colors, face_types = add_wall_segments_above_openings(
            vertices, faces, colors, face_types, detected_doors, DOOR_HEIGHT, image.shape, wall_color
        )
    if detected_windows:
        vertices, faces, colors, face_types = add_wall_segments_above_openings(
            vertices, faces, colors, face_types, detected_windows, WINDOW_BOTTOM + WINDOW_HEIGHT, image.shape, wall_color
        )

    fig = plot_3d_model(vertices, faces, colors)
    if fig:
        st.subheader("3D Model Visualization"); st.plotly_chart(fig, use_container_width=True)

    create_download_section_with_groups(vertices, faces, colors, face_types)  