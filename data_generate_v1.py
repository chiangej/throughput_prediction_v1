from sionna.rt import load_scene
from sionna.rt import PlanarArray
from sionna.rt import (
    Transmitter,
    Receiver
)
from sionna.rt import PathSolver
import numpy as np
import os
from sionna.rt import Camera
import gc

def rotate_y(vec, angle):
    """Rotation convention already verified for this imported scene."""
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return R @ np.asarray(vec, dtype=float)
  
def sample_safe_turn_slot(start_x, speed, rng, room_x_max, boundary_margin, total_duration, turn_slots, dt, earliest_s=0.7):
    """
    Random turn time, but constrained so the UE cannot hit x=ROOM_X_MAX
    before the U-turn starts.
    """
    max_forward_time = (
        room_x_max - boundary_margin - float(start_x)
    ) / speed

    # Need enough time to complete the U-turn within this episode.
    latest_by_episode = total_duration - turn_slots * dt - 0.2
    latest_s = min(max_forward_time, latest_by_episode)

    # If geometry is tight, fall back to the latest feasible slot.
    earliest_s = min(earliest_s, latest_s)
    if latest_s <= 0:
        raise ValueError("No valid forward walking interval inside the 4x4 room.")

    turn_time = float(rng.uniform(max(0.1, earliest_s), max(0.1, latest_s)))
    return int(round(turn_time / dt))
  
def angle_between(v1, v2):
    
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return np.pi

    cos_angle = np.dot(v1, v2) / (n1 * n2)

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    return np.arccos(cos_angle)
  
def point_to_segment_distance(point, start, end):
    
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    line = end - start

    line_length_sq = np.dot(line, line)

    if line_length_sq == 0:
        return np.inf, 0.0

    alpha = np.dot(
        point - start,
        line
    ) / line_length_sq

    projection = (
        start
        + alpha * line
    )

    distance = np.linalg.norm(
        point - projection
    )

    return distance, alpha
  
def wrap_angle(angle):
  return (angle + np.pi) % (2 * np.pi) - np.pi

def data_generate():
  
  scene = load_scene(
      "3D_scene_sionna_with_VR_2people.xml",
      merge_shapes=False
  )
  scene.frequency = 28e9
  
  human_body = scene.get("elm__13")
  headset = scene.get("headset_rx")
  human2_body = scene.get("human2_body")
  HUMAN2_BODY = human2_body.position.numpy().reshape(-1).astype(float)
  BODY = human_body.position.numpy().reshape(-1).astype(float)
  HEADSET = headset.position.numpy().reshape(-1).astype(float)
  HEADSET_OFFSET = HEADSET - BODY
  UE2_Y_OFFSET_FROM_UE1 = 0.70   # UE2 與 UE1 橫向距離 70 cm
  HUMAN2_BODY[1] = (BODY[1] + UE2_Y_OFFSET_FROM_UE1)
  RX_LOCAL_OFFSET = np.array([0.0, -0.06, 0.0])
  records = []
  previous_ue1_heading = None
  previous_ue2_heading = None
  
  scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
  )

  scene.rx_array = PlanarArray(
      num_rows=1,
      num_cols=1,
      vertical_spacing=0.5,
      horizontal_spacing=0.5,
      pattern="iso",
      polarization="V"
  )

  tx = Transmitter(
      name="tx",
      position=[0, 3, 1.5],
      display_radius=0.05
  )

  rx = Receiver(
      name="rx",
      position=[0, -3, 1.5],
      display_radius=0.05
  )

  scene.add(tx)
  scene.add(rx)
  tx.look_at(rx)

  solver = PathSolver()

  cam = Camera(
      position=[4.0, -6.0, 4.0],
      look_at=[1.25, 0.0, 1.2]
  )
  
  rx = scene.get("rx")
  # Simulation parameters
  dt = 0.1                     # 100 ms / slot
  total_duration = 8.0         # total episode length
  num_slots = int(total_duration / dt) + 1

  # First-version randomness: only speed + turn time
  RANDOM_SEED = None           # set an integer (e.g. 42) to reproduce an episode
  rng = np.random.default_rng(RANDOM_SEED)

  UE1_SPEED_RANGE = (0.7, 1.3)   # m/s
  UE2_SPEED_RANGE = (0.5, 1.1)   # m/s

  ue1_speed = float(rng.uniform(*UE1_SPEED_RANGE))
  ue2_speed = float(rng.uniform(*UE2_SPEED_RANGE))

  # 面對+x方向
  START_HEADING_DEG = 90.0
  ue1_heading = float(np.deg2rad(START_HEADING_DEG))
  ue2_heading = float(np.deg2rad(START_HEADING_DEG))

  # U-turn: 180 degrees over 12 slots = 1.2 s, -15 degrees / slot.
  TURN_SLOTS = 12
  TURN_STEP_DEG = -15.0
  
  # 4m x 4m大小的room
  ROOM_X_MIN = 0.0
  ROOM_X_MAX = 4.0
  ROOM_Y_MIN = -0.5
  ROOM_Y_MAX = 3.5
  BOUNDARY_MARGIN = 0.05
  
  assert ROOM_Y_MIN <= BODY[1] <= ROOM_Y_MAX, "UE1 starts outside room Y bounds"
  assert ROOM_Y_MIN <= HUMAN2_BODY[1] <= ROOM_Y_MAX, "UE2 starts outside room Y bounds" 
  
  ue1_turn_slot = sample_safe_turn_slot(BODY[0],ue1_speed,rng,ROOM_X_MAX,BOUNDARY_MARGIN,
total_duration,TURN_SLOTS,dt)
  ue2_turn_slot = sample_safe_turn_slot(HUMAN2_BODY[0], ue2_speed, rng, ROOM_X_MAX, BOUNDARY_MARGIN,
total_duration, TURN_SLOTS, dt)

  ue1_turn_end = ue1_turn_slot + TURN_SLOTS
  ue2_turn_end = ue2_turn_slot + TURN_SLOTS

  ue1_x = float(BODY[0])
  ue2_x = float(HUMAN2_BODY[0])
  
  for t in range(num_slots):
    current_time = t * dt

    # UE1 state
    if t < ue1_turn_slot:
        ue1_phase = "walking_forward"
        if t > 0:
            ue1_x += ue1_speed * dt

    elif t < ue1_turn_end:
        ue1_phase = "turning"
        ue1_heading = (
            ue1_heading + np.deg2rad(TURN_STEP_DEG)
        ) % (2 * np.pi)

    else:
        ue1_phase = "walking_back"
        ue1_x -= ue1_speed * dt

    # Numerical safety: never leave the 4x4 room.
    ue1_x = float(np.clip(
        ue1_x,
        ROOM_X_MIN + BOUNDARY_MARGIN,
        ROOM_X_MAX - BOUNDARY_MARGIN
    ))
    
    # UE2 state
    if t < ue2_turn_slot:
        ue2_phase = "walking_forward"
        if t > 0:
            ue2_x += ue2_speed * dt

    elif t < ue2_turn_end:
        ue2_phase = "turning"
        ue2_heading = (
            ue2_heading + np.deg2rad(TURN_STEP_DEG)
        ) % (2 * np.pi)

    else:
        ue2_phase = "walking_back"
        ue2_x -= ue2_speed * dt

    ue2_x = float(np.clip(
        ue2_x,
        ROOM_X_MIN + BOUNDARY_MARGIN,
        ROOM_X_MAX - BOUNDARY_MARGIN
    ))
    
    # UE1 geometry
    body_pos = np.array([
        ue1_x,
        float(BODY[1]),
        float(BODY[2])
    ])

    ue1_orientation = [float(ue1_heading), 0.0, 0.0]
    human_body.orientation = ue1_orientation
    headset.orientation = ue1_orientation
    human_body.position = body_pos.tolist()

    headset_pos = body_pos + HEADSET_OFFSET
    headset.position = headset_pos.tolist()

    # Rx follows the headset face direction.
    rx_offset_rotated = rotate_y(RX_LOCAL_OFFSET, ue1_heading)
    rx_pos = headset_pos + rx_offset_rotated
    rx.position = rx_pos.tolist()
    
    # UE2 geometry
    human2_pos = np.array([
        ue2_x,
        float(HUMAN2_BODY[1]),
        float(HUMAN2_BODY[2])
    ])

    ue2_orientation = [float(ue2_heading), 0.0, 0.0]
    human2_body.orientation = ue2_orientation
    human2_body.position = human2_pos.tolist()
    
    # Safety check: both UEs remain inside 4x4 movement area
    assert ROOM_X_MIN <= body_pos[0] <= ROOM_X_MAX
    assert ROOM_Y_MIN <= body_pos[1] <= ROOM_Y_MAX
    assert ROOM_X_MIN <= human2_pos[0] <= ROOM_X_MAX
    assert ROOM_Y_MIN <= human2_pos[1] <= ROOM_Y_MAX
    
    paths = solver(
        scene=scene,
        max_depth=3,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        synthetic_array=True,
        seed=1
    )
    
    # Convert Paths -> CSI immediately
    num_subcarriers = 64
    subcarrier_spacing = 30e3 
    
    frequencies = (
        np.arange(num_subcarriers)
        - num_subcarriers // 2
    ) * subcarrier_spacing
    
    csi = paths.cfr(
        frequencies=frequencies,
        normalize_delays=False,
        normalize=False,
        out_type="numpy"
    )

    csi = np.squeeze(csi)
    
    overall_phase = "turning" if (
        ue1_phase == "turning" or ue2_phase == "turning"
    ) else "walking"
    
    BLOCKAGE_RADIUS = 0.30
    UE1_FOV_DEG = 120.0
    UE1_HALF_FOV = np.deg2rad(UE1_FOV_DEG / 2)
    
    # --------------------------------------------------------
    # Use Rx XY as UE1 location
    ue1_xy = np.array([
        float(rx_pos[0]),
        float(rx_pos[1])
    ])
    
    ue2_xy = np.array([
        float(human2_pos[0]),
        float(human2_pos[1])
    ])
    
    tx_pos = (
        scene.get("tx")
        .position
        .numpy()
        .reshape(-1)
        .astype(float)
    )
    
    tx_xy = np.array([
        tx_pos[0],
        tx_pos[1]
    ])
    
    # UE1 viewing direction
    
    view_heading = (
        ue1_heading
        - np.pi / 2
    )
    
    ue1_forward = np.array([
        np.cos(view_heading),
        np.sin(view_heading)
    ])
    
    # Is Tx inside UE1's viewing direction?
    ue1_to_tx = (
        tx_xy
        - ue1_xy
    )
    
    tx_view_angle = angle_between(
        ue1_forward,
        ue1_to_tx
    )
    
    tx_in_ue1_view = (
        tx_view_angle
        <= UE1_HALF_FOV
    )
    
    # Is UE2 in front of UE1?
    ue1_to_ue2 = (
        ue2_xy
        - ue1_xy
    )
    
    ue2_in_ue1_front = (
        np.dot(
            ue1_forward,
            ue1_to_ue2
        )
        > 0
    )
    
    # Is UE2 geometrically between UE1 and Tx?
    
    ue2_los_distance, alpha = (
        point_to_segment_distance(
            point=ue2_xy,
            start=ue1_xy,
            end=tx_xy
        )
    )
    
    ue2_between_ue1_tx = (
        0.0 < alpha < 1.0
    )
    
    ue2_close_to_los = (
        ue2_los_distance
        < BLOCKAGE_RADIUS
    )
    
    inter_user_blockage = (
        tx_in_ue1_view
        and
        ue2_in_ue1_front
        and
        ue2_between_ue1_tx
        and
        ue2_close_to_los
    )
    
    if t == 0:
        ue1_linear_velocity = 0.0
        ue2_linear_velocity = 0.0
    else:
        ue1_linear_velocity = abs(
            ue1_x - prev_ue1_x
        ) / dt
        ue2_linear_velocity = abs(
            ue2_x - prev_ue2_x
        ) / dt
            
    ue1_dir_sin = np.sin(
        ue1_heading / 2
    )

    ue1_dir_cos = np.cos(
        ue1_heading / 2
    )

    ue2_dir_sin = np.sin(
        ue2_heading / 2
    )

    ue2_dir_cos = np.cos(
        ue2_heading / 2
    )
    
    if previous_ue1_heading is None:
        ue1_angular_velocity = 0.0
    else:
        delta_heading = wrap_angle(
            ue1_heading - previous_ue1_heading
        )

        ue1_angular_velocity = (
            delta_heading / dt
        )

    if previous_ue2_heading is None:
        ue2_angular_velocity = 0.0
    else:
        delta_heading = wrap_angle(
            ue2_heading - previous_ue2_heading
        )

        ue2_angular_velocity = (
            delta_heading / dt
        )
        
    previous_ue1_heading = ue1_heading
    previous_ue2_heading = ue2_heading
    
    # throughput
    # Throughput parameters
    P_tx_dBm = 20.0  
    noise_dBm = -90.0   
    subcarrier_spacing = 30e3

    P_tx = 10 ** ((P_tx_dBm - 30) / 10)
    noise_power = 10 ** ((noise_dBm - 30) / 10)
    
    channel_gain = np.abs(csi) ** 2
    snr = P_tx * channel_gain / noise_power
    throughput_bps = np.sum(
      subcarrier_spacing
      * np.log2(1 + snr)
    )

    throughput_mbps = (
      throughput_bps / 1e6
    )
    
    records.append({
        "slot": t,
        "time_s": current_time,

        # UE1 / headset / Rx
        "x": float(headset_pos[0]),
        "y": float(headset_pos[1]),
        "z": float(headset_pos[2]),
        "heading_rad": float(ue1_heading),
        "heading_deg": float(np.rad2deg(ue1_heading)),
        "ue1_speed_mps": float(ue1_speed),
        "ue1_turn_slot": int(ue1_turn_slot),
        "ue1_phase": ue1_phase,
        "rx_x": float(rx_pos[0]),
        "rx_y": float(rx_pos[1]),
        "rx_z": float(rx_pos[2]),
        
        "ue1_dir_sin": float(ue1_dir_sin),
        "ue1_dir_cos": float(ue1_dir_cos),
        "ue1_linear_velocity": float(ue1_linear_velocity),
        "ue1_angular_velocity": float(ue1_angular_velocity),

        # UE2
        "human2_x": float(human2_pos[0]),
        "human2_y": float(human2_pos[1]),
        "human2_z": float(human2_pos[2]),
        "human2_heading_rad": float(ue2_heading),
        "human2_heading_deg": float(np.rad2deg(ue2_heading)),
        "human2_speed_mps": float(ue2_speed),
        "human2_turn_slot": int(ue2_turn_slot),
        "human2_phase": ue2_phase,

        "ue2_dir_sin": float(ue2_dir_sin),
        "ue2_dir_cos": float(ue2_dir_cos),

        "ue2_linear_velocity": float(ue2_linear_velocity),
        "ue2_angular_velocity": float(ue2_angular_velocity),

        # Inter-user blockage

        "tx_in_ue1_view": bool(tx_in_ue1_view),
        "tx_view_angle_deg": float(np.rad2deg(tx_view_angle)),
        "ue2_in_ue1_front": bool(ue2_in_ue1_front),
        "ue2_between_ue1_tx": bool(ue2_between_ue1_tx),
        "ue2_los_distance": float(ue2_los_distance),
        "inter_user_blockage": int(inter_user_blockage),
        "phase": overall_phase,
        "csi": csi,
        "throughput_mbps": float(throughput_mbps)
        })
    del paths
    
    
    print(
            f"slot={t:02d} time={current_time:.1f}s | "
            f"UE1 x={ue1_x:.2f} h={np.rad2deg(ue1_heading):.0f}° {ue1_phase} | "
            f"UE2 x={ue2_x:.2f} h={np.rad2deg(ue2_heading):.0f}° {ue2_phase}",
            flush=True
        )
    
  return records


def add_throughput_window(records):

    WINDOW_SIZE = 10   # 10 slots × 0.1 s = 1 second

    throughput_raw = np.array([
        r["throughput_mbps"]
        for r in records
    ], dtype=np.float32)

    throughput_window = np.zeros_like(
        throughput_raw
    )

    for i in range(len(throughput_raw)):

        start = max(
            0,
            i - WINDOW_SIZE + 1
        )

        throughput_window[i] = np.mean(
            throughput_raw[start:i+1]
        )

    # 放回每一個 record
    for i, r in enumerate(records):
        r["throughput_window_mbps"] = (
            throughput_window[i]
        )

    return records
          
# records -> numpy dataset
def training_data_save(records, filename):
    slots = np.array([r["slot"] for r in records], dtype=np.int32)
    time_s = np.array([r["time_s"] for r in records], dtype=np.float32)   
    
    position = np.array([[r["x"], r["y"], r["z"]] for r in records], dtype=np.float32)

    rx_position = np.array([[r["rx_x"], r["rx_y"], r["rx_z"]] for r in records], dtype=np.float32)

    heading_rad = np.array([r["heading_rad"] for r in records], dtype=np.float32)
    heading_deg = np.array([r["heading_deg"] for r in records], dtype=np.float32)

    human2_position = np.array([
        [r["human2_x"], r["human2_y"], r["human2_z"]]
        for r in records
    ], dtype=np.float32)

    human2_heading_rad = np.array([
        r["human2_heading_rad"] for r in records
    ], dtype=np.float32)

    human2_heading_deg = np.array([
        r["human2_heading_deg"] for r in records
    ], dtype=np.float32)
    
    csi = np.stack([r["csi"] for r in records])
    csi_real = np.real(csi).astype(np.float32)
    csi_imag = np.imag(csi).astype(np.float32)

    throughput_mbps = np.array([
        r["throughput_mbps"] for r in records
    ], dtype=np.float32)

    throughput_window_mbps = np.array([
    r["throughput_window_mbps"]
    for r in records
    ], dtype=np.float32)

    phase = np.array([r["phase"] for r in records])
    ue1_phase = np.array([r["ue1_phase"] for r in records])
    human2_phase = np.array([r["human2_phase"] for r in records])
    
    ue1_speed_mps = np.float32(records[0]["ue1_speed_mps"])
    human2_speed_mps = np.float32(records[0]["human2_speed_mps"])
    ue1_turn_slot = np.int32(records[0]["ue1_turn_slot"])
    human2_turn_slot = np.int32(records[0]["human2_turn_slot"])
    
    inter_user_blockage = np.array([r["inter_user_blockage"] for r in records], dtype=np.int8)

    tx_in_ue1_view = np.array([
        r["tx_in_ue1_view"]
        for r in records
    ], dtype=np.int8)

    tx_view_angle_deg = np.array([
        r["tx_view_angle_deg"]
        for r in records
    ], dtype=np.float32)

    ue2_in_ue1_front = np.array([
        r["ue2_in_ue1_front"]
        for r in records
    ], dtype=np.int8)

    ue2_between_ue1_tx = np.array([
        r["ue2_between_ue1_tx"]
        for r in records
    ], dtype=np.int8)

    ue2_los_distance = np.array([
        r["ue2_los_distance"]
        for r in records
    ], dtype=np.float32)
    
    ue1_dir_sin = np.array([r["ue1_dir_sin"] for r in records], dtype=np.float32)
    ue1_dir_cos = np.array([r["ue1_dir_cos"] for r in records], dtype=np.float32)
    ue1_linear_velocity = np.array([r["ue1_linear_velocity"] for r in records], dtype=np.float32)
    ue1_angular_velocity = np.array([r["ue1_angular_velocity"] for r in records], dtype=np.float32)
    ue2_dir_sin = np.array([r["ue2_dir_sin"] for r in records], dtype=np.float32)
    ue2_dir_cos = np.array([r["ue2_dir_cos"] for r in records], dtype=np.float32)
    ue2_linear_velocity = np.array([r["ue2_linear_velocity"] for r in records], dtype=np.float32)
    ue2_angular_velocity = np.array([r["ue2_angular_velocity"] for r in records], dtype=np.float32)

    np.savez(
    filename,
    slot=slots,
    time_s=time_s,
    position=position,
    rx_position=rx_position,
    heading_rad=heading_rad,
    heading_deg=heading_deg,
    human2_position=human2_position,
    human2_heading_rad=human2_heading_rad,
    human2_heading_deg=human2_heading_deg,
    csi_real=csi_real,
    csi_imag=csi_imag,
    throughput_mbps=throughput_mbps,
    throughput_window_mbps=throughput_window_mbps,
    phase=phase,
    ue1_phase=ue1_phase,
    human2_phase=human2_phase,
    ue1_speed_mps=ue1_speed_mps,
    human2_speed_mps=human2_speed_mps,
    ue1_turn_slot=ue1_turn_slot,
    human2_turn_slot=human2_turn_slot,
    
    inter_user_blockage=inter_user_blockage,
    tx_in_ue1_view=tx_in_ue1_view,
    tx_view_angle_deg=tx_view_angle_deg,
    ue2_in_ue1_front=ue2_in_ue1_front,
    ue2_between_ue1_tx=ue2_between_ue1_tx,
    ue2_los_distance=ue2_los_distance,
    
    ue1_dir_sin=ue1_dir_sin,
    ue1_dir_cos=ue1_dir_cos,
    ue1_linear_velocity=ue1_linear_velocity,
    ue1_angular_velocity=ue1_angular_velocity,
    ue2_dir_sin=ue2_dir_sin,    
    ue2_dir_cos=ue2_dir_cos,
    ue2_linear_velocity=ue2_linear_velocity,
    ue2_angular_velocity=ue2_angular_velocity
)
    print(f"saved: {filename}")
    

def main():
  
  output_dir = "dataset_v1"

  os.makedirs(
      output_dir,
      exist_ok=True
  )
  
  for i in range(500):
    print(f"\n========== Episode {i:03d} ==========")
    records = data_generate()
    add_throughput_window(records)
    filename = os.path.join(
            output_dir,
            f"episode_{i:04d}.npz"
        )

    training_data_save(
        records,
        filename
    )
    del records
    gc.collect()

if __name__ == "__main__":
    main()
