#!/usr/bin/env python3
"""
NeuroMechFly Kinematic Replay - Complete Simulation
Replays experimentally recorded fly locomotion using FlyGym physics simulation
Based on the official NeuroMechFly tutorial notebooks
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange
import urllib.request
from hashlib import md5

# ============================================================================
# FLYGYM COMPATIBILITY LAYER
# ============================================================================

def get_all_leg_dofs():
    """Get all leg DoFs - compatible with different FlyGym versions"""
    try:
        from flygym.preprogrammed import all_leg_dofs
        return all_leg_dofs
    except (ImportError, ModuleNotFoundError, AttributeError):
        # Manual definition for compatibility
        all_leg_dofs = []
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]:
            all_leg_dofs.extend([
                f"joint_{leg}Coxa",
                f"joint_{leg}Coxa_roll",
                f"joint_{leg}Coxa_yaw",
                f"joint_{leg}Femur",
                f"joint_{leg}Femur_roll",
                f"joint_{leg}Tibia",
                f"joint_{leg}Tarsus1",
            ])
        return all_leg_dofs

def get_all_tarsi_links():
    """Get all tarsi links - compatible with different FlyGym versions"""
    try:
        from flygym.preprogrammed import all_tarsi_links
        return all_tarsi_links
    except (ImportError, ModuleNotFoundError, AttributeError):
        # Manual definition for compatibility
        return [
            f"{leg}_Tarsus{i}" 
            for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
            for i in range(1, 6)
        ]

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Simulation configuration parameters"""
    # Paths
    DATA_DIR = Path("./data/inverse_kinematics")
    OUTPUT_DIR = Path("./outputs/kinematic_replay")
    
    # Data download
    LEG_JOINT_ANGLES_FILE = "leg_joint_angles.pkl"
    LEG_JOINT_ANGLES_MD5 = "420d8380b2fcb9ca310f7936a11effd4"
    LEG_JOINT_ANGLES_URL = "https://github.com/NeLy-EPFL/neuromechfly-workshop/raw/refs/heads/main/data/inverse_kinematics/leg_joint_angles.pkl"
    
    # Simulation parameters
    TIMESTEP = 1e-4  # 0.1 ms
    PLAY_SPEED = 0.05  # Video playback speed
    TARSUS_OFFSET = -np.pi / 5  # Tippy-toe walking angle
    
    # Video rendering
    DRAW_CONTACTS = True  # Show contact forces as arrows
    ENABLE_VISION = True  # Enable vision simulation
    RENDER_RAW_VISION = True  # Render raw camera images

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def download_data():
    """Download kinematic data if not already present"""
    print("=" * 70)
    print("STEP 1: Downloading kinematic data")
    print("=" * 70)
    
    data_path = Config.DATA_DIR / Config.LEG_JOINT_ANGLES_FILE
    
    # Check if file exists and is valid
    download_needed = not data_path.is_file()
    
    if not download_needed:
        with open(data_path, "rb") as f:
            data_file = f.read()
            checksum = md5(data_file).hexdigest()
            if checksum != Config.LEG_JOINT_ANGLES_MD5:
                print(f"  ⚠️  Checksum mismatch! Expected {Config.LEG_JOINT_ANGLES_MD5}")
                print(f"      Got {checksum}")
                download_needed = True
    
    if download_needed:
        print("  📥 Downloading from GitHub...")
        Config.DATA_DIR.mkdir(exist_ok=True, parents=True)
        urllib.request.urlretrieve(Config.LEG_JOINT_ANGLES_URL, data_path)
        print(f"  ✓ Downloaded to {data_path}")
    else:
        print(f"  ✓ Using cached data from {data_path}")
    
    return data_path


def format_seqikpy_data(data, corresp_dict=None):
    """
    Convert seqikpy format to FlyGym format
    
    Args:
        data: Dictionary with joint angles from seqikpy
        corresp_dict: Mapping of segment names
    
    Returns:
        Dictionary with FlyGym-formatted joint names
    """
    if corresp_dict is None:
        corresp_dict = {
            "ThC": "Coxa",
            "CTr": "Femur",
            "FTi": "Tibia",
            "TiTa": "Tarsus1"
        }
    
    data_gym = {}
    for joint, values in data.items():
        if joint in ["meta", "swing_stance_time"]:
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


def load_and_prepare_data(data_path):
    """Load and interpolate kinematic data"""
    print("\n" + "=" * 70)
    print("STEP 2: Loading and preparing kinematic data")
    print("=" * 70)
    
    # Load pickled data
    with open(data_path, "rb") as f:
        seq_ikdata = pickle.load(f)
    
    data = format_seqikpy_data(seq_ikdata)
    print(f"  ✓ Loaded kinematic data")
    print(f"    Metadata keys: {list(data['meta'].keys())}")
    
    # Get actuated joints (using compatibility helper)
    actuated_joints = get_all_leg_dofs()
    
    # Calculate time parameters
    run_time = len(data["joint_RFCoxa_yaw"]) * data["meta"]["timestep"]
    target_num_steps = int(run_time / Config.TIMESTEP)
    
    print(f"    Recording duration: {run_time:.3f} seconds")
    print(f"    Original timestep: {data['meta']['timestep']*1000:.3f} ms")
    print(f"    Simulation timestep: {Config.TIMESTEP*1000:.3f} ms")
    print(f"    Total simulation steps: {target_num_steps}")
    
    # Interpolate data to match simulation timestep
    data_block = np.zeros((len(actuated_joints), target_num_steps))
    input_t = np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
    output_t = np.arange(target_num_steps) * Config.TIMESTEP
    
    for i, joint in enumerate(actuated_joints):
        data_block[i, :] = np.interp(output_t, input_t, data[joint])
    
    # Apply tarsus offset (tippy-toe walking)
    for i, joint in enumerate(actuated_joints):
        if "Tarsus" in joint:
            data_block[i, :] = Config.TARSUS_OFFSET
    
    print(f"  ✓ Interpolated to {target_num_steps} steps ({len(actuated_joints)} joints)")
    
    return data_block, actuated_joints, output_t, run_time


def visualize_joint_angles(data_block, actuated_joints, time_array):
    """Create visualization of joint angle trajectories"""
    print("\n" + "=" * 70)
    print("STEP 3: Visualizing joint angles")
    print("=" * 70)
    
    Config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    fig, axs = plt.subplots(
        3, 2, 
        figsize=(10, 8), 
        sharex=True, 
        sharey=True, 
        tight_layout=True
    )
    
    legs = [
        f"{side} {pos} leg"
        for pos in ["front", "middle", "hind"]
        for side in ["Left", "Right"]
    ]
    
    for i, leg in enumerate(legs):
        ax = axs.flatten()[i]
        leg_code = f"{leg.split()[0][0]}{leg.split()[1][0]}".upper()
        
        for j, dof in enumerate(actuated_joints):
            if dof.split("_")[1][:2] != leg_code:
                continue
            ax.plot(
                time_array, 
                np.rad2deg(data_block[j, :]), 
                label=dof[8:], 
                alpha=0.7
            )
        
        ax.set_ylim(-180, 180)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Angle (deg)", fontsize=9)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_title(leg, fontweight='bold')
        ax.grid(alpha=0.3)
        
        if leg == "Right front leg":
            ax.legend(loc='upper right', fontsize=7, ncol=2)
    
    fig.suptitle(
        "Joint Angle Trajectories for All Legs", 
        fontsize=12, 
        fontweight='bold'
    )
    
    output_path = Config.OUTPUT_DIR / "01_joint_angles.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to {output_path}")


def setup_simulation(actuated_joints):
    """Initialize FlyGym simulation"""
    print("\n" + "=" * 70)
    print("STEP 4: Setting up FlyGym simulation")
    print("=" * 70)
    
    # Try different import patterns for different FlyGym versions
    Fly = None
    Camera = None
    SingleFlySimulation = None
    
    # Pattern 1: Modern API (flygym 1.x+)
    try:
        from flygym import Fly, Camera, SingleFlySimulation
        print("  ✓ Using modern FlyGym API (flygym.Fly)")
    except ImportError:
        pass
    
    # Pattern 2: Older API with submodules
    if Fly is None:
        try:
            from flygym.fly import Fly
            from flygym.camera import Camera
            from flygym.simulation import SingleFlySimulation
            print("  ✓ Using FlyGym API with submodules")
        except ImportError:
            pass
    
    # Pattern 3: Core module
    if Fly is None:
        try:
            from flygym.core import Fly, Camera, SingleFlySimulation
            print("  ✓ Using FlyGym core API")
        except ImportError:
            pass
    
    # Pattern 4: Arena module (some versions)
    if Fly is None:
        try:
            from flygym.arena import Fly
            from flygym.camera import Camera
            from flygym.simulation import SingleFlySimulation
            print("  ✓ Using FlyGym arena API")
        except ImportError:
            pass
    
    if Fly is None or Camera is None or SingleFlySimulation is None:
        print("\n  ❌ ERROR: Cannot find FlyGym classes!")
        print("\n  Your FlyGym installation structure is not recognized.")
        print("  Please run the diagnostic tool:")
        print("    python diagnose_flygym.py")
        print("\n  Then share the output so we can create a compatible version.")
        raise ImportError("FlyGym API not compatible. Run diagnose_flygym.py for details.")
    
    # Get all tarsi links (using compatibility helper)
    all_tarsi_links = get_all_tarsi_links()
    
    # Monitor both actuated and passive tarsal joints
    tarsal_joints = [
        f"joint_{tarsus_seg}"
        for tarsus_seg in all_tarsi_links
        if "Tarsus1" not in tarsus_seg
    ]
    monitored_joints = actuated_joints + tarsal_joints
    
    print(f"  Creating Fly model...")
    print(f"    Actuated joints: {len(actuated_joints)}")
    print(f"    Monitored joints: {len(monitored_joints)} (including passive tarsal joints)")
    
    # Create fly with vision and position control
    fly = Fly(
        init_pose="stretch",
        actuated_joints=actuated_joints,
        control="position",  # Position control for kinematic replay
        monitored_joints=monitored_joints,
        enable_vision=Config.ENABLE_VISION,
        render_raw_vision=Config.RENDER_RAW_VISION,
    )
    
    print(f"  ✓ Fly model created")
    
    # Setup camera with contact force visualization
    print(f"  Creating Camera...")
    
    # Try different Camera initialization patterns
    cam = None
    
    # Pattern 1: Modern API (camera receives fly object)
    try:
        cam = Camera(
            fly=fly,
            camera_id="Animat/camera_left",
            play_speed=Config.PLAY_SPEED,
            draw_contacts=Config.DRAW_CONTACTS,
            play_speed_text=True,
        )
        print(f"    ✓ Camera created (modern API with fly parameter)")
    except TypeError:
        pass
    
    # Pattern 2: Older API (camera receives just camera_id and other params)
    if cam is None:
        try:
            cam = Camera(
                camera_id="Animat/camera_left",
                play_speed=Config.PLAY_SPEED,
                draw_contacts=Config.DRAW_CONTACTS,
                play_speed_text=True,
            )
            print(f"    ✓ Camera created (older API without fly parameter)")
        except TypeError:
            pass
    
    # Pattern 3: Minimal API (only camera_id)
    if cam is None:
        try:
            cam = Camera(
                camera_id="Animat/camera_left",
            )
            print(f"    ✓ Camera created (minimal API)")
        except:
            pass
    
    # Pattern 4: No parameters
    if cam is None:
        try:
            cam = Camera()
            print(f"    ✓ Camera created (no parameters)")
        except:
            pass
    
    if cam is None:
        print(f"    ⚠️  Could not create Camera, continuing without video rendering...")
        cam = None
    
    if cam is not None:
        print(f"    Camera: {getattr(cam, 'camera_id', 'unknown')}")
        print(f"    Draw contacts: {Config.DRAW_CONTACTS}")
        print(f"    Play speed: {Config.PLAY_SPEED}x")
    
    # Create simulation
    print(f"  Creating Simulation...")
    
    # Try different SingleFlySimulation patterns
    sim = None
    
    # Pattern 1: With cameras list
    if cam is not None:
        try:
            sim = SingleFlySimulation(
                fly=fly,
                cameras=[cam],
            )
            print(f"    ✓ Simulation created with camera")
        except TypeError:
            pass
    
    # Pattern 2: Without cameras (if camera failed)
    if sim is None:
        try:
            sim = SingleFlySimulation(
                fly=fly,
            )
            print(f"    ✓ Simulation created without camera")
            cam = None  # Set cam to None since simulation doesn't support it
        except TypeError:
            pass
    
    # Pattern 3: Just fly object, no named parameter
    if sim is None:
        try:
            sim = SingleFlySimulation(fly)
            print(f"    ✓ Simulation created (positional fly argument)")
            cam = None
        except:
            pass
    
    if sim is None:
        raise RuntimeError("Could not create SingleFlySimulation with any known API pattern")
    
    print(f"  ✓ Simulation ready")
    
    return sim, fly, cam, monitored_joints


def run_simulation(sim, fly, data_block, target_num_steps):
    """Execute kinematic replay simulation"""
    print("\n" + "=" * 70)
    print("STEP 5: Running kinematic replay simulation")
    print("=" * 70)
    
    obs, info = sim.reset()
    
    obs_list = []
    raw_vision_list = []
    vision_list = []
    
    print(f"  Starting simulation loop ({target_num_steps} steps)...")
    print(f"  This may take several minutes...")
    
    for i in trange(target_num_steps, desc="  Progress"):
        # Use recorded joint angles as target positions
        obs, reward, terminated, truncated, info = sim.step(
            {"joints": data_block[:, i]}
        )
        
        # Store vision data separately to reduce RAM usage (if available)
        vision = obs.pop("vision", None)
        obs_list.append(obs.copy())
        
        # Only store vision if enabled and available
        if vision is not None and fly.render_raw_vision:
            try:
                if fly._vision_update_mask[-1]:
                    raw_vision_list.append(info.get("raw_vision"))
                    vision_list.append(vision)
            except (AttributeError, KeyError, IndexError):
                # Vision might not be available in all FlyGym versions
                pass
        
        # Render frame (if render method exists)
        try:
            sim.render()
        except (AttributeError, TypeError):
            # Render might not be available or might have different signature
            pass
    
    print(f"  ✓ Simulation complete!")
    print(f"    Observations recorded: {len(obs_list)}")
    if vision_list:
        print(f"    Vision frames: {len(vision_list)}")
    
    return obs_list, vision_list, raw_vision_list


def save_video(cam):
    """Save rendered video"""
    print("\n" + "=" * 70)
    print("STEP 6: Saving video")
    print("=" * 70)
    
    if cam is None:
        print("  ⚠️  No camera available, skipping video generation...")
        return None
    
    try:
        camera_id = getattr(cam, 'camera_id', 'camera')
        video_name = f"kinematic_replay_{camera_id.split('/')[-1]}"
    except:
        video_name = "kinematic_replay_video"
    
    if hasattr(cam, 'draw_contacts') and cam.draw_contacts:
        video_name += "_contacts"
    video_name += ".mp4"
    
    output_path = Config.OUTPUT_DIR / video_name
    
    try:
        print(f"  Encoding video...")
        
        # Try different save_video API patterns
        try:
            cam.save_video(output_path, stabilization_time=0)
        except TypeError:
            try:
                cam.save_video(output_path)
            except:
                cam.save_video(str(output_path))
        
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Video saved to {output_path}")
            print(f"    File size: {file_size_mb:.2f} MB")
        else:
            print(f"  ⚠️  Video file was not created")
            
    except Exception as e:
        print(f"  ⚠️  Error saving video: {e}")
        print(f"  Continuing with analysis...")
    
    return output_path


def analyze_contact_forces(obs_list, monitored_joints):
    """Analyze and plot ground reaction forces"""
    print("\n" + "=" * 70)
    print("STEP 7: Analyzing contact forces")
    print("=" * 70)
    
    try:
        # Get all tarsi links (using compatibility helper)
        all_tarsi_links = get_all_tarsi_links()
        
        # Check if contact forces are available
        if "contact_forces" not in obs_list[0]:
            print("  ⚠️  Contact forces not available in observations, skipping...")
            return None
        
        legs = [side + pos for pos in "FMH" for side in "LR"]
        
        # Map tarsal segments to legs
        leg_tarsal_seg_contact_id = {
            leg: [i for i, tarsal_seg in enumerate(all_tarsi_links) if leg in tarsal_seg]
            for leg in legs
        }
        
        # Calculate contact forces per leg
        leg_contacts = np.linalg.norm(
            [
                np.sum(
                    [
                        [obs["contact_forces"][i] for obs in obs_list]
                        for i in leg_tarsal_seg_contact_id[leg]
                    ],
                    axis=0,
                )
                for leg in legs
            ],
            axis=-1,
        )
        
        time = np.arange(len(leg_contacts[0])) * Config.TIMESTEP
        
        # Plot ground reaction forces
        fig, axs = plt.subplots(
            len(legs), 1, 
            figsize=(10, 10), 
            tight_layout=True, 
            sharey=True
        )
        
        colors = plt.get_cmap("tab10", len(legs))
        
        for i, leg in enumerate(legs):
            ax = axs[i]
            ax.plot(time, leg_contacts[i], label=leg, color=colors(i), linewidth=2)
            ax.set_ylabel("Force [µN]", fontsize=9)
            ax.fill_between(time, leg_contacts[i], alpha=0.2, color=colors(i))
            ax.grid(alpha=0.3)
            max_force = leg_contacts.max()
            if max_force > 0:
                ax.set_ylim(0, max_force * 1.1)
            ax.text(0.02, 0.95, leg, transform=ax.transAxes, 
                    fontsize=10, fontweight='bold', va='top')
        
        axs[-1].set_xlabel("Time (s)", fontsize=10)
        fig.suptitle(
            "Ground Reaction Forces - Contact Forces at Leg Tips", 
            fontsize=12, 
            fontweight='bold'
        )
        
        output_path = Config.OUTPUT_DIR / "02_ground_reaction_forces.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}")
        
        return leg_contacts
        
    except Exception as e:
        print(f"  ⚠️  Error analyzing contact forces: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_joint_torques(obs_list, monitored_joints):
    """Analyze and plot joint torques for one leg"""
    print("\n" + "=" * 70)
    print("STEP 8: Analyzing joint torques")
    print("=" * 70)
    
    try:
        # Check if joints data is available
        if "joints" not in obs_list[0]:
            print("  ⚠️  Joint data not available in observations, skipping...")
            return
        
        legs = [side + pos for pos in "FMH" for side in "LR"]
        focus_leg = "LH"
        
        # Get indices for joints of the focus leg
        leg_joints_to_id = {
            leg: [i for i, joint in enumerate(monitored_joints) if leg in joint] 
            for leg in legs
        }
        
        # Extract torque values (index 2 = torque)
        # Convert to µN·mm (multiply by 1e9)
        leg_torques = np.array([
            obs["joints"][2, leg_joints_to_id[focus_leg]] 
            for obs in obs_list
        ]) * 1e9
        
        leg_joints = [joint for joint in monitored_joints if focus_leg in joint]
        
        time = np.arange(len(leg_torques)) * Config.TIMESTEP
        
        # Create subplots
        fig, axs = plt.subplots(
            len(leg_joints), 1,
            figsize=(12, 10),
            sharex=True,
            tight_layout=True
        )
        
        if len(leg_joints) == 1:
            axs = [axs]
        
        colors = plt.get_cmap("viridis", len(leg_joints))
        
        for i, (ax, joint) in enumerate(zip(axs, leg_joints)):
            label = joint.replace(f"joint_{focus_leg}", "")
            if "_" not in label:
                label = label + "_pitch"
            
            # Passive tarsal joints use dashed lines
            linestyle = "dashed" if ("Tarsus" in label and "Tarsus1" not in label) else "solid"
            
            ax.plot(
                time, 
                leg_torques[:, i], 
                color=colors(i), 
                linestyle=linestyle, 
                linewidth=2, 
                label=label
            )
            ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        
        # Add legend for active/passive joints
        handles = [
            plt.Line2D([0], [0], color="black", linestyle="solid", linewidth=2),
            plt.Line2D([0], [0], color="black", linestyle="dashed", linewidth=2)
        ]
        labels = ["Active joint", "Passive joint"]
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))
        
        axs[-1].set_xlabel("Time (s)", fontsize=10)
        fig.text(0.04, 0.5, "Torque [µN·mm]", va="center", rotation="vertical", fontsize=11)
        fig.suptitle(f"Joint Torques for {focus_leg} Leg", fontsize=12, fontweight='bold')
        
        output_path = Config.OUTPUT_DIR / "03_joint_torques.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}")
        
    except Exception as e:
        print(f"  ⚠️  Error analyzing joint torques: {e}")
        import traceback
        traceback.print_exc()


def analyze_fly_trajectory(obs_list):
    """Plot fly trajectory in 2D"""
    print("\n" + "=" * 70)
    print("STEP 9: Analyzing fly trajectory")
    print("=" * 70)
    
    try:
        # Check if fly position data is available
        if "fly" not in obs_list[0]:
            print("  ⚠️  Fly position data not available in observations, skipping...")
            return
        
        # Extract fly position (x, y) over time
        fly_positions = np.array([obs["fly"][0, :2] for obs in obs_list])
        
        # Calculate statistics
        total_distance = np.sum(np.linalg.norm(np.diff(fly_positions, axis=0), axis=1))
        displacement = np.linalg.norm(fly_positions[-1] - fly_positions[0])
        
        print(f"  Total distance traveled: {total_distance:.3f} mm")
        print(f"  Net displacement: {displacement:.3f} mm")
        print(f"  Start position: ({fly_positions[0, 0]:.2f}, {fly_positions[0, 1]:.2f}) mm")
        print(f"  End position: ({fly_positions[-1, 0]:.2f}, {fly_positions[-1, 1]:.2f}) mm")
        
        # Plot trajectory
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot path
        ax.plot(
            fly_positions[:, 0], 
            fly_positions[:, 1], 
            'b-', 
            linewidth=2, 
            alpha=0.6,
            label="Trajectory"
        )
        
        # Mark start and end
        ax.scatter(
            fly_positions[0, 0], 
            fly_positions[0, 1], 
            c='g', 
            s=150, 
            marker='o', 
            label="Start", 
            zorder=5,
            edgecolors='black',
            linewidths=2
        )
        ax.scatter(
            fly_positions[-1, 0], 
            fly_positions[-1, 1], 
            c='r', 
            s=150, 
            marker='X', 
            label="End", 
            zorder=5,
            edgecolors='black',
            linewidths=2
        )
        
        # Add distance annotation
        ax.text(
            0.05, 0.95, 
            f"Distance: {total_distance:.2f} mm\nDisplacement: {displacement:.2f} mm",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax.set_xlabel("X position (mm)", fontsize=11)
        ax.set_ylabel("Y position (mm)", fontsize=11)
        ax.set_title("Fly Locomotion Trajectory", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')
        
        output_path = Config.OUTPUT_DIR / "04_fly_trajectory.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}")
        
    except Exception as e:
        print(f"  ⚠️  Error analyzing fly trajectory: {e}")
        import traceback
        traceback.print_exc()


def visualize_vision(fly, vision_list, raw_vision_list):
    """Create video of fly's visual input"""
    print("\n" + "=" * 70)
    print("STEP 10: Visualizing fly vision")
    print("=" * 70)
    
    if not vision_list:
        print("  ⚠️  No vision data recorded, skipping...")
        return
    
    try:
        from flygym.vision.visualize import visualize_visual_input
        
        output_path = Config.OUTPUT_DIR / "05_retina_images.mp4"
        
        print(f"  Creating vision visualization...")
        
        # Check if retina exists
        if not hasattr(fly, 'retina') or fly.retina is None:
            print("  ⚠️  Fly retina not available, skipping vision visualization...")
            return
        
        visualize_visual_input(
            fly.retina,
            output_path,
            vision_list,
            raw_vision_list,
            np.ones(len(raw_vision_list), dtype=bool),
            playback_speed=Config.PLAY_SPEED,
        )
        plt.close("all")
        
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Vision video saved to {output_path}")
            print(f"    File size: {file_size_mb:.2f} MB")
        else:
            print("  ⚠️  Vision video was not created")
        
    except ImportError as e:
        print(f"  ⚠️  Vision visualization not available: {e}")
    except Exception as e:
        print(f"  ⚠️  Error creating vision video: {e}")
        import traceback
        print("  Full error:")
        traceback.print_exc()


def print_summary(run_time):
    """Print final summary"""
    print("\n" + "=" * 70)
    print("KINEMATIC REPLAY COMPLETE!")
    print("=" * 70)
    
    print(f"\nOutput directory: {Config.OUTPUT_DIR.absolute()}")
    print(f"\nGenerated files:")
    
    files = [
        ("01_joint_angles.png", "Joint angle trajectories for all legs"),
        ("02_ground_reaction_forces.png", "Contact forces during walking"),
        ("03_joint_torques.png", "Torques at each joint (LH leg)"),
        ("04_fly_trajectory.png", "Walking trajectory in XY plane"),
        ("kinematic_replay_camera_left_contacts.mp4", "3D video with contact forces"),
        ("05_retina_images.mp4", "Fly's visual input (optional)"),
    ]
    
    for filename, description in files:
        filepath = Config.OUTPUT_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename:45s} - {description} ({size_mb:.2f} MB)")
    
    print(f"\nSimulation statistics:")
    print(f"  Duration: {run_time:.3f} seconds")
    print(f"  Timestep: {Config.TIMESTEP * 1000:.3f} ms")
    print(f"  Physics engine: MuJoCo (via FlyGym)")
    
    print("\n" + "=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("NeuroMechFly Kinematic Replay Simulation")
    print("Using FlyGym physics-based simulation")
    print("=" * 70)
    
    try:
        # Check FlyGym installation
        try:
            import flygym
            # Try to get version, but don't fail if not available
            try:
                version = flygym.__version__
                print(f"\n✓ FlyGym version {version} detected")
            except AttributeError:
                print("\n✓ FlyGym detected (version info not available)")
        except ImportError:
            print("\n❌ ERROR: FlyGym is not installed!")
            print("\nPlease install FlyGym:")
            print('  pip install "flygym[examples]"')
            print("\nFor more information, visit:")
            print("  https://neuromechfly.org/installation.html")
            return
        
        # Step 1: Download data
        data_path = download_data()
        
        # Step 2: Load and prepare data
        data_block, actuated_joints, time_array, run_time = load_and_prepare_data(data_path)
        
        # Step 3: Visualize joint angles
        visualize_joint_angles(data_block, actuated_joints, time_array)
        
        # Step 4: Setup simulation
        sim, fly, cam, monitored_joints = setup_simulation(actuated_joints)
        
        # Step 5: Run simulation
        obs_list, vision_list, raw_vision_list = run_simulation(
            sim, fly, data_block, data_block.shape[1]
        )
        
        # Step 6: Save video
        save_video(cam)
        
        # Step 7: Analyze contact forces
        analyze_contact_forces(obs_list, monitored_joints)
        
        # Step 8: Analyze joint torques
        analyze_joint_torques(obs_list, monitored_joints)
        
        # Step 9: Analyze trajectory
        analyze_fly_trajectory(obs_list)
        
        # Step 10: Visualize vision (optional)
        if Config.ENABLE_VISION and vision_list:
            visualize_vision(fly, vision_list, raw_vision_list)
        
        # Print summary
        print_summary(run_time)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())