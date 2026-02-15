#!/usr/bin/env python3
"""
NeuroMechFly 3D Animation
Renders the full 3D fly model with articulated legs during recorded kinematic replay
Based on NeuroMechFly geometry and kinematics
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
import imageio

# ============================================================================
# NEUROMECHFLY GEOMETRY DEFINITION
# ============================================================================

# Segment lengths (mm) - from NeuroMechFly template
LEG_SEGMENTS = {
    "RF": {"Coxa": 0.400, "Femur": 0.706, "Tibia": 0.519, "Tarsus": 0.663},
    "RM": {"Coxa": 0.182, "Femur": 0.783, "Tibia": 0.573, "Tarsus": 0.704},
    "RH": {"Coxa": 0.396, "Femur": 0.756, "Tibia": 0.570, "Tarsus": 0.663},
    "LF": {"Coxa": 0.400, "Femur": 0.706, "Tibia": 0.519, "Tarsus": 0.663},
    "LM": {"Coxa": 0.182, "Femur": 0.783, "Tibia": 0.573, "Tarsus": 0.704},
    "LH": {"Coxa": 0.396, "Femur": 0.756, "Tibia": 0.570, "Tarsus": 0.663},
}

# Leg base positions relative to thorax center (mm)
LEG_BASE_POSITIONS = {
    "RF": np.array([0.35, -0.27, 0.0]),
    "RM": np.array([0.0, -0.125, 0.0]),
    "RH": np.array([-0.35, -0.27, 0.0]),
    "LF": np.array([0.35, 0.27, 0.0]),
    "LM": np.array([0.0, 0.125, 0.0]),
    "LH": np.array([-0.35, 0.27, 0.0]),
}

# Thorax dimensions (mm)
THORAX_SIZE = np.array([0.003, 0.00267, 0.00267])

# Colors for visualization
LEG_COLORS = {
    "RF": "#E74C3C",  # Red
    "RM": "#F39C12",  # Orange
    "RH": "#F1C40F",  # Yellow
    "LF": "#3498DB",  # Blue
    "LM": "#1ABC9C",  # Cyan
    "LH": "#9B59B6",  # Purple
}

print("=" * 70)
print("NeuroMechFly 3D Animation Generator")
print("=" * 70)

# ============================================================================
# LOAD KINEMATIC DATA
# ============================================================================
print("\n[1/5] Loading kinematic data...")

data_path = Path("./data/inverse_kinematics/leg_joint_angles.pkl")
with open(data_path, "rb") as f:
    raw_data = pickle.load(f)

# Format data
def format_seqikpy_data(data, corresp_dict={"ThC": "Coxa", "CTr": "Femur", "FTi": "Tibia", "TiTa": "Tarsus1"}):
    data_gym = {}
    for joint, values in data.items():
        if joint == "meta" or joint == "swing_stance_time":
            data_gym[joint] = values
        else:
            leg = joint[6:8]
            joint_name = joint[9:]
            seg, dof = joint_name.split("_")
            if dof == "pitch":
                newjoint = f"joint_{leg}{corresp_dict[seg]}"
            else:
                newjoint = f"joint_{leg}{corresp_dict[seg]}_{dof}"
            data_gym[newjoint] = values
    return data_gym

data = format_seqikpy_data(raw_data)
print(f"  Loaded {len(data)-2} joint trajectories")

# Get timestep info
meta_timestep = data["meta"]["timestep"]
run_time = len(data["joint_RFCoxa_yaw"]) * meta_timestep
print(f"  Duration: {run_time:.2f} seconds")

# ============================================================================
# FORWARD KINEMATICS - COMPUTE LEG POSITIONS
# ============================================================================
print("\n[2/5] Computing forward kinematics for all frames...")

def forward_kinematics_leg(base_pos, angles_dict, leg_name):
    """
    Compute 3D positions of leg segments using forward kinematics
    
    angles_dict contains:
    - joint_{leg}Coxa_yaw
    - joint_{leg}Coxa_roll  (or no roll for some)
    - joint_{leg}Femur
    - joint_{leg}Femur_roll (optional)
    - joint_{leg}Tibia
    - joint_{leg}Tarsus1
    """
    
    # Start from leg base
    positions = [base_pos.copy()]
    current_pos = base_pos.copy()
    current_frame = np.eye(3)  # Rotation matrix
    
    segments = ["Coxa", "Femur", "Tibia", "Tarsus"]
    
    for seg_idx, segment in enumerate(segments):
        # Get angle for this segment
        angle_key = f"joint_{leg_name}{segment}"
        
        if angle_key not in angles_dict:
            # Use last position
            positions.append(current_pos.copy())
            continue
        
        angle = angles_dict.get(angle_key, 0.0)
        
        # Rotation depends on segment type
        if segment == "Coxa":
            # Yaw rotation (around Z)
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        else:
            # Pitch rotation (around Y)
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        
        current_frame = current_frame @ R
        
        # Move forward by segment length
        seg_length = LEG_SEGMENTS[leg_name][segment]
        displacement = current_frame @ np.array([seg_length, 0, 0])
        current_pos = current_pos + displacement
        
        positions.append(current_pos.copy())
    
    return positions

# Precompute all leg positions for each frame
all_positions = {}
n_frames = len(data["joint_RFCoxa_yaw"])

for leg in ["RF", "RM", "RH", "LF", "LM", "LH"]:
    all_positions[leg] = []

for frame_idx in tqdm(range(0, n_frames, max(1, n_frames // 100)), desc="  Computing kinematics"):
    angles_frame = {}
    for joint_name in data.keys():
        if joint_name not in ["meta", "swing_stance_time"]:
            angles_frame[joint_name] = data[joint_name][frame_idx]
    
    for leg in ["RF", "RM", "RH", "LF", "LM", "LH"]:
        base_pos = LEG_BASE_POSITIONS[leg]
        leg_positions = forward_kinematics_leg(base_pos, angles_frame, leg)
        all_positions[leg].append(np.array(leg_positions))

# ============================================================================
# RENDER 3D ANIMATION
# ============================================================================
print("\n[3/5] Rendering 3D visualization...")

output_dir = Path("neuromechfly_3d")
output_dir.mkdir(exist_ok=True)
frames = []

# Create figure for high-quality rendering
fig = plt.figure(figsize=(12, 10), dpi=100)
ax = fig.add_subplot(111, projection="3d")

render_every = max(1, n_frames // 200)  # Render ~200 frames
target_shape = None

for frame_idx in tqdm(range(0, n_frames, render_every), desc="  Rendering frames"):
    ax.clear()
    
    # Draw thorax (body)
    thorax_x = [-THORAX_SIZE[0]/2, THORAX_SIZE[0]/2]
    thorax_y = [-THORAX_SIZE[1]/2, THORAX_SIZE[1]/2]
    thorax_z = [-THORAX_SIZE[2]/2, THORAX_SIZE[2]/2]
    
    # Thorax box outline
    for x in thorax_x:
        for y in thorax_y:
            ax.plot([x, x], [y, y], thorax_z, 'k-', linewidth=2, alpha=0.5)
    for y in thorax_y:
        for z in thorax_z:
            ax.plot(thorax_x, [y, y], [z, z], 'k-', linewidth=2, alpha=0.5)
    for x in thorax_x:
        for z in thorax_z:
            ax.plot([x, x], thorax_y, [z, z], 'k-', linewidth=2, alpha=0.5)
    
    # Draw all 6 legs
    for leg in ["RF", "RM", "RH", "LF", "LM", "LH"]:
        leg_pos = all_positions[leg][frame_idx // render_every] if frame_idx // render_every < len(all_positions[leg]) else all_positions[leg][-1]
        
        # Draw leg segments
        for segment_idx in range(len(leg_pos) - 1):
            p1 = leg_pos[segment_idx]
            p2 = leg_pos[segment_idx + 1]
            
            # Draw segment as line
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=LEG_COLORS[leg],
                linewidth=3,
                alpha=0.8
            )
            
            # Draw joint as sphere
            ax.scatter(*p1, color=LEG_COLORS[leg], s=20, alpha=0.6)
        
        # Draw foot (end effector)
        ax.scatter(*leg_pos[-1], color=LEG_COLORS[leg], s=40, marker='o', alpha=1.0, edgecolors='black', linewidths=1)
    
    # Set labels and limits
    ax.set_xlabel("X (mm)", fontsize=10)
    ax.set_ylabel("Y (mm)", fontsize=10)
    ax.set_zlabel("Z (mm)", fontsize=10)
    
    # Dynamic limits based on leg positions
    all_coords = []
    for leg in all_positions:
        leg_pos = all_positions[leg][frame_idx // render_every] if frame_idx // render_every < len(all_positions[leg]) else all_positions[leg][-1]
        all_coords.extend(leg_pos)
    all_coords = np.array(all_coords)
    
    margin = 0.5
    ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
    ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
    ax.set_zlim(min(all_coords[:, 2].min() - margin, -1), all_coords[:, 2].max() + margin)
    
    # Rotating camera view
    azim = 45 + frame_idx * 0.2  # Rotate around
    elev = 20
    ax.view_init(elev=elev, azim=azim)
    
    ax.set_title(f"NeuroMechFly 3D Kinematic Replay - Frame {frame_idx}/{n_frames}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Save frame to PNG buffer
    fig.tight_layout()
    frame_path = output_dir / f"frame_{frame_idx:06d}.png"
    fig.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
    
    # Read PNG and normalize size
    import imageio.v2 as imageio
    frame_image = imageio.imread(frame_path)
    
    # Set target shape from first frame
    if target_shape is None:
        # Round to nearest multiple of 16 for video codec compatibility
        h, w = frame_image.shape[:2]
        target_shape = (h - h % 16, w - w % 16)
    
    # Resize frame to target shape
    from PIL import Image
    img = Image.fromarray(frame_image)
    img_resized = img.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
    frame_image = np.array(img_resized)
    
    frames.append(frame_image)

plt.close(fig)

# ============================================================================
# CREATE VIDEO
# ============================================================================
print("\n[4/5] Creating MP4 video...")

fps = 15
output_video = Path("neuromechfly_3d") / "neuromechfly_3d_animation.mp4"

# Write video
writer = imageio.get_writer(output_video, fps=fps, codec='libx264')
for frame in frames:
    writer.append_data(frame)
writer.close()

print(f"  Video saved: {output_video}")
print(f"  Duration: {len(frames) / fps:.2f} seconds")
print(f"  File size: {output_video.stat().st_size / 1e6:.2f} MB")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("3D ANIMATION COMPLETE!")
print("=" * 70)
print(f"\nGenerated files:")
print(f"  • {output_video} - Full 3D animation with rotating view")
print(f"  • Individual frames in {output_dir}/")
print(f"\nVisualization details:")
print(f"  • Thorax: Dark gray box at center")
print(f"  • Legs: 6 colored articulated legs (RF, RM, RH, LF, LM, LH)")
print(f"  • Joints: Shown as spheres at segment connections")
print(f"  • Feet: Larger markers at leg tips")
print(f"  • Camera: Rotating view for 3D understanding")
print(f"\nFPS: {fps}")
print(f"Duration: {len(frames)/fps:.2f} seconds")
print("=" * 70 + "\n")
