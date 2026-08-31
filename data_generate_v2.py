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


# ============================================================
# Geometry helpers
# ============================================================

def rotate_y(vec, angle):
    """
    Rotation convention already verified for this imported scene.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    return R @ np.asarray(
        vec,
        dtype=float
    )


def sample_next_turn_slot(
    current_slot,
    current_x,
    direction,
    speed,
    rng,
    room_x_min,
    room_x_max,
    boundary_margin,
    dt,
    min_walk_time=1.0,
    preferred_max_walk_time=4.0
):
    """
    Randomly select next turn time.

    Turn timing is random, but constrained so the UE
    turns before reaching the room boundary.

    direction:
        +1 -> toward +X
        -1 -> toward -X
    """

    # --------------------------------------------------------
    # Available distance in current walking direction
    # --------------------------------------------------------

    if direction > 0:

        available_distance = (
            room_x_max
            - boundary_margin
            - current_x
        )

    else:

        available_distance = (
            current_x
            - room_x_min
            - boundary_margin
        )


    # --------------------------------------------------------
    # Leave extra safety distance from wall
    # --------------------------------------------------------

    EXTRA_MARGIN = 0.25

    safe_distance = max(
        0.0,
        available_distance
        - EXTRA_MARGIN
    )

    max_safe_time = (
        safe_distance
        / speed
    )


    # --------------------------------------------------------
    # Random walking duration upper bound
    # --------------------------------------------------------

    max_walk_time = min(
        preferred_max_walk_time,
        max_safe_time
    )


    # --------------------------------------------------------
    # Random walk time
    # --------------------------------------------------------

    if max_walk_time <= 0.0:

        walk_time = dt

    elif max_walk_time <= min_walk_time:

        walk_time = rng.uniform(
            dt,
            max_walk_time
        )

    else:

        walk_time = rng.uniform(
            min_walk_time,
            max_walk_time
        )


    walk_slots = max(
        1,
        int(
            round(
                walk_time / dt
            )
        )
    )


    return (
        current_slot
        + walk_slots
    )


def angle_between(v1, v2):

    v1 = np.asarray(
        v1,
        dtype=float
    )

    v2 = np.asarray(
        v2,
        dtype=float
    )

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return np.pi

    cos_angle = np.dot(
        v1,
        v2
    ) / (
        n1 * n2
    )

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    return np.arccos(
        cos_angle
    )


def point_to_segment_distance(
    point,
    start,
    end
):

    point = np.asarray(
        point,
        dtype=float
    )

    start = np.asarray(
        start,
        dtype=float
    )

    end = np.asarray(
        end,
        dtype=float
    )

    line = end - start

    line_length_sq = np.dot(
        line,
        line
    )

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

    return (
        angle + np.pi
    ) % (
        2 * np.pi
    ) - np.pi


# ============================================================
# Main episode generation
# ============================================================

def data_generate():

    scene = load_scene(
        "3D_scene_sionna_with_VR_2people.xml",
        merge_shapes=False
    )

    # Paper-style carrier frequency
    scene.frequency = 28e9


    # ========================================================
    # Scene objects
    # ========================================================

    human_body = scene.get(
        "elm__13"
    )

    headset = scene.get(
        "headset_rx"
    )

    human2_body = scene.get(
        "human2_body"
    )


    HUMAN2_BODY = (
        human2_body
        .position
        .numpy()
        .reshape(-1)
        .astype(float)
    )

    BODY = (
        human_body
        .position
        .numpy()
        .reshape(-1)
        .astype(float)
    )

    HEADSET = (
        headset
        .position
        .numpy()
        .reshape(-1)
        .astype(float)
    )


    HEADSET_OFFSET = (
        HEADSET - BODY
    )


    # UE2 stays 70 cm away in Y
    UE2_Y_OFFSET_FROM_UE1 = 0.70

    HUMAN2_BODY[1] = (
        BODY[1]
        + UE2_Y_OFFSET_FROM_UE1
    )


    RX_LOCAL_OFFSET = np.array([
        0.0,
        -0.06,
        0.0
    ])


    records = []

    previous_ue1_heading = None
    previous_ue2_heading = None


    # ========================================================
    # Antenna arrays
    # ========================================================

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
        position=[
            0,
            3,
            1.5
        ],
        display_radius=0.05
    )

    rx = Receiver(
        name="rx",
        position=[
            0,
            -3,
            1.5
        ],
        display_radius=0.05
    )

    scene.add(tx)
    scene.add(rx)

    tx.look_at(rx)


    solver = PathSolver()


    cam = Camera(
        position=[
            4.0,
            -6.0,
            4.0
        ],
        look_at=[
            1.25,
            0.0,
            1.2
        ]
    )


    rx = scene.get(
        "rx"
    )


    # ========================================================
    # Simulation parameters
    # ========================================================

    dt = 0.1

    total_duration = 30.0

    num_slots = int(
        total_duration / dt
    ) + 1


    RANDOM_SEED = None

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    UE1_SPEED_RANGE = (
        0.7,
        1.3
    )

    UE2_SPEED_RANGE = (
        0.5,
        1.1
    )


    ue1_speed = float(
        rng.uniform(
            *UE1_SPEED_RANGE
        )
    )

    ue2_speed = float(
        rng.uniform(
            *UE2_SPEED_RANGE
        )
    )


    # ========================================================
    # Heading
    # ========================================================

    START_HEADING_DEG = 90.0

    ue1_heading = float(
        np.deg2rad(
            START_HEADING_DEG
        )
    )

    ue2_heading = float(
        np.deg2rad(
            START_HEADING_DEG
        )
    )


    # 180° turn
    TURN_SLOTS = 12
    TURN_STEP_DEG = -15.0


    # ========================================================
    # Room
    # ========================================================

    ROOM_X_MIN = 0.0
    ROOM_X_MAX = 4.0

    ROOM_Y_MIN = -0.5
    ROOM_Y_MAX = 3.5

    BOUNDARY_MARGIN = 0.05


    assert (
        ROOM_Y_MIN
        <= BODY[1]
        <= ROOM_Y_MAX
    )

    assert (
        ROOM_Y_MIN
        <= HUMAN2_BODY[1]
        <= ROOM_Y_MAX
    )


    # ========================================================
    # Initial movement state
    # ========================================================

    ue1_x = float(
        BODY[0]
    )

    ue2_x = float(
        HUMAN2_BODY[0]
    )


    # +1 -> +X
    # -1 -> -X
    ue1_direction = +1
    ue2_direction = +1


    ue1_is_turning = False
    ue2_is_turning = False


    ue1_turn_counter = 0
    ue2_turn_counter = 0


    # ========================================================
    # First random turn
    # ========================================================

    ue1_next_turn_slot = sample_next_turn_slot(
        current_slot=0,
        current_x=ue1_x,
        direction=ue1_direction,
        speed=ue1_speed,
        rng=rng,
        room_x_min=ROOM_X_MIN,
        room_x_max=ROOM_X_MAX,
        boundary_margin=BOUNDARY_MARGIN,
        dt=dt
    )


    ue2_next_turn_slot = sample_next_turn_slot(
        current_slot=0,
        current_x=ue2_x,
        direction=ue2_direction,
        speed=ue2_speed,
        rng=rng,
        room_x_min=ROOM_X_MIN,
        room_x_max=ROOM_X_MAX,
        boundary_margin=BOUNDARY_MARGIN,
        dt=dt
    )


    # ========================================================
    # Precompute CFR frequencies
    # ========================================================

    num_subcarriers = 64
    subcarrier_spacing = 30e3

    frequencies = (
        np.arange(
            num_subcarriers
        )
        - num_subcarriers // 2
    ) * subcarrier_spacing


    # ========================================================
    # Constant blockage config
    # ========================================================

    BLOCKAGE_RADIUS = 0.30

    UE1_FOV_DEG = 120.0

    UE1_HALF_FOV = np.deg2rad(
        UE1_FOV_DEG / 2
    )


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


    # ========================================================
    # Main time loop
    # ========================================================

    for t in range(
        num_slots
    ):

        current_time = (
            t * dt
        )


        # ----------------------------------------------------
        # Previous positions for actual velocity
        # ----------------------------------------------------

        prev_ue1_x = ue1_x
        prev_ue2_x = ue2_x


        # ====================================================
        # UE1 movement
        # ====================================================

        if ue1_is_turning:

            ue1_phase = "turning"

            ue1_heading = (
                ue1_heading
                + np.deg2rad(
                    TURN_STEP_DEG
                )
            ) % (
                2 * np.pi
            )

            ue1_turn_counter += 1


            if (
                ue1_turn_counter
                >= TURN_SLOTS
            ):

                ue1_is_turning = False

                ue1_turn_counter = 0

                ue1_direction *= -1


                ue1_next_turn_slot = (
                    sample_next_turn_slot(
                        current_slot=t + 1,
                        current_x=ue1_x,
                        direction=ue1_direction,
                        speed=ue1_speed,
                        rng=rng,
                        room_x_min=ROOM_X_MIN,
                        room_x_max=ROOM_X_MAX,
                        boundary_margin=BOUNDARY_MARGIN,
                        dt=dt
                    )
                )


        else:

            if ue1_direction > 0:

                ue1_phase = (
                    "walking_forward"
                )

            else:

                ue1_phase = (
                    "walking_back"
                )


            # ------------------------------------------------
            # Random turn begins
            # ------------------------------------------------

            if (
                t
                >= ue1_next_turn_slot
            ):

                ue1_is_turning = True

                ue1_turn_counter = 1

                ue1_phase = "turning"


                ue1_heading = (
                    ue1_heading
                    + np.deg2rad(
                        TURN_STEP_DEG
                    )
                ) % (
                    2 * np.pi
                )


            else:

                ue1_x += (
                    ue1_direction
                    * ue1_speed
                    * dt
                )


        # ====================================================
        # UE2 movement
        # ====================================================

        if ue2_is_turning:

            ue2_phase = "turning"

            ue2_heading = (
                ue2_heading
                + np.deg2rad(
                    TURN_STEP_DEG
                )
            ) % (
                2 * np.pi
            )

            ue2_turn_counter += 1


            if (
                ue2_turn_counter
                >= TURN_SLOTS
            ):

                ue2_is_turning = False

                ue2_turn_counter = 0

                ue2_direction *= -1


                ue2_next_turn_slot = (
                    sample_next_turn_slot(
                        current_slot=t + 1,
                        current_x=ue2_x,
                        direction=ue2_direction,
                        speed=ue2_speed,
                        rng=rng,
                        room_x_min=ROOM_X_MIN,
                        room_x_max=ROOM_X_MAX,
                        boundary_margin=BOUNDARY_MARGIN,
                        dt=dt
                    )
                )


        else:

            if ue2_direction > 0:

                ue2_phase = (
                    "walking_forward"
                )

            else:

                ue2_phase = (
                    "walking_back"
                )


            if (
                t
                >= ue2_next_turn_slot
            ):

                ue2_is_turning = True

                ue2_turn_counter = 1

                ue2_phase = "turning"


                ue2_heading = (
                    ue2_heading
                    + np.deg2rad(
                        TURN_STEP_DEG
                    )
                ) % (
                    2 * np.pi
                )


            else:

                ue2_x += (
                    ue2_direction
                    * ue2_speed
                    * dt
                )


        # ====================================================
        # Safety: no clipping
        # ====================================================

        if not (
            ROOM_X_MIN
            + BOUNDARY_MARGIN
            <= ue1_x
            <= ROOM_X_MAX
            - BOUNDARY_MARGIN
        ):

            raise RuntimeError(
                f"UE1 left room at "
                f"slot={t}, "
                f"x={ue1_x:.3f}"
            )


        if not (
            ROOM_X_MIN
            + BOUNDARY_MARGIN
            <= ue2_x
            <= ROOM_X_MAX
            - BOUNDARY_MARGIN
        ):

            raise RuntimeError(
                f"UE2 left room at "
                f"slot={t}, "
                f"x={ue2_x:.3f}"
            )


        # ====================================================
        # UE1 geometry
        # ====================================================

        body_pos = np.array([
            ue1_x,
            float(
                BODY[1]
            ),
            float(
                BODY[2]
            )
        ])


        ue1_orientation = [
            float(
                ue1_heading
            ),
            0.0,
            0.0
        ]


        human_body.orientation = (
            ue1_orientation
        )

        headset.orientation = (
            ue1_orientation
        )

        human_body.position = (
            body_pos.tolist()
        )


        headset_pos = (
            body_pos
            + HEADSET_OFFSET
        )

        headset.position = (
            headset_pos.tolist()
        )


        rx_offset_rotated = rotate_y(
            RX_LOCAL_OFFSET,
            ue1_heading
        )

        rx_pos = (
            headset_pos
            + rx_offset_rotated
        )

        rx.position = (
            rx_pos.tolist()
        )


        # ====================================================
        # UE2 geometry
        # ====================================================

        human2_pos = np.array([
            ue2_x,
            float(
                HUMAN2_BODY[1]
            ),
            float(
                HUMAN2_BODY[2]
            )
        ])


        ue2_orientation = [
            float(
                ue2_heading
            ),
            0.0,
            0.0
        ]


        human2_body.orientation = (
            ue2_orientation
        )

        human2_body.position = (
            human2_pos.tolist()
        )


        # ====================================================
        # Ray tracing
        # ====================================================

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


        # ====================================================
        # CSI
        # ====================================================

        csi = paths.cfr(
            frequencies=frequencies,
            normalize_delays=False,
            normalize=False,
            out_type="numpy"
        )

        csi = np.squeeze(
            csi
        )


        # ====================================================
        # Phase
        # ====================================================

        overall_phase = (
            "turning"
            if (
                ue1_phase == "turning"
                or
                ue2_phase == "turning"
            )
            else
            "walking"
        )


        # ====================================================
        # Inter-user blockage
        # ====================================================

        ue1_xy = np.array([
            float(
                rx_pos[0]
            ),
            float(
                rx_pos[1]
            )
        ])


        ue2_xy = np.array([
            float(
                human2_pos[0]
            ),
            float(
                human2_pos[1]
            )
        ])


        view_heading = (
            ue1_heading
            - np.pi / 2
        )


        ue1_forward = np.array([
            np.cos(
                view_heading
            ),
            np.sin(
                view_heading
            )
        ])


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


        (
            ue2_los_distance,
            alpha
        ) = (
            point_to_segment_distance(
                point=ue2_xy,
                start=ue1_xy,
                end=tx_xy
            )
        )


        ue2_between_ue1_tx = (
            0.0
            < alpha
            < 1.0
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


        # ====================================================
        # Linear velocity
        # ====================================================

        if t == 0:

            ue1_linear_velocity = 0.0
            ue2_linear_velocity = 0.0

        else:

            ue1_linear_velocity = (
                abs(
                    ue1_x
                    - prev_ue1_x
                )
                / dt
            )

            ue2_linear_velocity = (
                abs(
                    ue2_x
                    - prev_ue2_x
                )
                / dt
            )


        # ====================================================
        # Direction encoding
        # ====================================================

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


        # ====================================================
        # Angular velocity
        # ====================================================

        if previous_ue1_heading is None:

            ue1_angular_velocity = 0.0

        else:

            delta_heading = wrap_angle(
                ue1_heading
                - previous_ue1_heading
            )

            ue1_angular_velocity = (
                delta_heading
                / dt
            )


        if previous_ue2_heading is None:

            ue2_angular_velocity = 0.0

        else:

            delta_heading = wrap_angle(
                ue2_heading
                - previous_ue2_heading
            )

            ue2_angular_velocity = (
                delta_heading
                / dt
            )


        previous_ue1_heading = (
            ue1_heading
        )

        previous_ue2_heading = (
            ue2_heading
        )


        # ====================================================
        # Throughput
        # ====================================================

        P_tx_dBm = 20.0
        noise_dBm = -90.0


        P_tx = 10 ** (
            (
                P_tx_dBm
                - 30
            )
            / 10
        )


        noise_power = 10 ** (
            (
                noise_dBm
                - 30
            )
            / 10
        )


        channel_gain = (
            np.abs(csi) ** 2
        )


        snr = (
            P_tx
            * channel_gain
            / noise_power
        )


        throughput_bps = np.sum(
            subcarrier_spacing
            * np.log2(
                1 + snr
            )
        )


        throughput_mbps = (
            throughput_bps
            / 1e6
        )


        # ====================================================
        # Save numerical record
        # ====================================================

        records.append({

            "slot":
                t,

            "time_s":
                current_time,


            # UE1
            "x":
                float(
                    headset_pos[0]
                ),

            "y":
                float(
                    headset_pos[1]
                ),

            "z":
                float(
                    headset_pos[2]
                ),

            "heading_rad":
                float(
                    ue1_heading
                ),

            "heading_deg":
                float(
                    np.rad2deg(
                        ue1_heading
                    )
                ),

            "ue1_speed_mps":
                float(
                    ue1_speed
                ),

            "ue1_phase":
                ue1_phase,


            "rx_x":
                float(
                    rx_pos[0]
                ),

            "rx_y":
                float(
                    rx_pos[1]
                ),

            "rx_z":
                float(
                    rx_pos[2]
                ),


            "ue1_dir_sin":
                float(
                    ue1_dir_sin
                ),

            "ue1_dir_cos":
                float(
                    ue1_dir_cos
                ),

            "ue1_linear_velocity":
                float(
                    ue1_linear_velocity
                ),

            "ue1_angular_velocity":
                float(
                    ue1_angular_velocity
                ),


            # UE2
            "human2_x":
                float(
                    human2_pos[0]
                ),

            "human2_y":
                float(
                    human2_pos[1]
                ),

            "human2_z":
                float(
                    human2_pos[2]
                ),

            "human2_heading_rad":
                float(
                    ue2_heading
                ),

            "human2_heading_deg":
                float(
                    np.rad2deg(
                        ue2_heading
                    )
                ),

            "human2_speed_mps":
                float(
                    ue2_speed
                ),

            "human2_phase":
                ue2_phase,


            "ue2_dir_sin":
                float(
                    ue2_dir_sin
                ),

            "ue2_dir_cos":
                float(
                    ue2_dir_cos
                ),

            "ue2_linear_velocity":
                float(
                    ue2_linear_velocity
                ),

            "ue2_angular_velocity":
                float(
                    ue2_angular_velocity
                ),


            # Blockage
            "tx_in_ue1_view":
                bool(
                    tx_in_ue1_view
                ),

            "tx_view_angle_deg":
                float(
                    np.rad2deg(
                        tx_view_angle
                    )
                ),

            "ue2_in_ue1_front":
                bool(
                    ue2_in_ue1_front
                ),

            "ue2_between_ue1_tx":
                bool(
                    ue2_between_ue1_tx
                ),

            "ue2_los_distance":
                float(
                    ue2_los_distance
                ),

            "inter_user_blockage":
                int(
                    inter_user_blockage
                ),


            "phase":
                overall_phase,

            "csi":
                csi,

            "throughput_mbps":
                float(
                    throughput_mbps
                )
        })


        del paths


        print(
            f"slot={t:03d} "
            f"time={current_time:.1f}s | "
            f"UE1 x={ue1_x:.2f} "
            f"h={np.rad2deg(ue1_heading):.0f}° "
            f"{ue1_phase} | "
            f"UE2 x={ue2_x:.2f} "
            f"h={np.rad2deg(ue2_heading):.0f}° "
            f"{ue2_phase}",
            flush=True
        )


    return records


# ============================================================
# 1-second averaged throughput
# ============================================================

def add_throughput_window(records):

    WINDOW_SIZE = 10

    throughput_raw = np.array(
        [
            r["throughput_mbps"]
            for r in records
        ],
        dtype=np.float32
    )


    throughput_window = np.zeros_like(
        throughput_raw
    )


    for i in range(
        len(
            throughput_raw
        )
    ):

        start = max(
            0,
            i
            - WINDOW_SIZE
            + 1
        )


        throughput_window[i] = np.mean(
            throughput_raw[
                start:
                i + 1
            ]
        )


    for i, r in enumerate(
        records
    ):

        r[
            "throughput_window_mbps"
        ] = float(
            throughput_window[i]
        )


    return records


# ============================================================
# Save episode
# ============================================================

def training_data_save(
    records,
    filename
):

    slots = np.array(
        [
            r["slot"]
            for r in records
        ],
        dtype=np.int32
    )


    time_s = np.array(
        [
            r["time_s"]
            for r in records
        ],
        dtype=np.float32
    )


    position = np.array(
        [
            [
                r["x"],
                r["y"],
                r["z"]
            ]
            for r in records
        ],
        dtype=np.float32
    )


    rx_position = np.array(
        [
            [
                r["rx_x"],
                r["rx_y"],
                r["rx_z"]
            ]
            for r in records
        ],
        dtype=np.float32
    )


    heading_rad = np.array(
        [
            r["heading_rad"]
            for r in records
        ],
        dtype=np.float32
    )


    heading_deg = np.array(
        [
            r["heading_deg"]
            for r in records
        ],
        dtype=np.float32
    )


    human2_position = np.array(
        [
            [
                r["human2_x"],
                r["human2_y"],
                r["human2_z"]
            ]
            for r in records
        ],
        dtype=np.float32
    )


    human2_heading_rad = np.array(
        [
            r["human2_heading_rad"]
            for r in records
        ],
        dtype=np.float32
    )


    human2_heading_deg = np.array(
        [
            r["human2_heading_deg"]
            for r in records
        ],
        dtype=np.float32
    )


    csi = np.stack(
        [
            r["csi"]
            for r in records
        ]
    )


    csi_real = np.real(
        csi
    ).astype(
        np.float32
    )


    csi_imag = np.imag(
        csi
    ).astype(
        np.float32
    )


    throughput_mbps = np.array(
        [
            r["throughput_mbps"]
            for r in records
        ],
        dtype=np.float32
    )


    throughput_window_mbps = np.array(
        [
            r[
                "throughput_window_mbps"
            ]
            for r in records
        ],
        dtype=np.float32
    )


    phase = np.array(
        [
            r["phase"]
            for r in records
        ]
    )


    ue1_phase = np.array(
        [
            r["ue1_phase"]
            for r in records
        ]
    )


    human2_phase = np.array(
        [
            r["human2_phase"]
            for r in records
        ]
    )


    ue1_speed_mps = np.float32(
        records[0][
            "ue1_speed_mps"
        ]
    )


    human2_speed_mps = np.float32(
        records[0][
            "human2_speed_mps"
        ]
    )


    inter_user_blockage = np.array(
        [
            r["inter_user_blockage"]
            for r in records
        ],
        dtype=np.int8
    )


    tx_in_ue1_view = np.array(
        [
            r["tx_in_ue1_view"]
            for r in records
        ],
        dtype=np.int8
    )


    tx_view_angle_deg = np.array(
        [
            r["tx_view_angle_deg"]
            for r in records
        ],
        dtype=np.float32
    )


    ue2_in_ue1_front = np.array(
        [
            r["ue2_in_ue1_front"]
            for r in records
        ],
        dtype=np.int8
    )


    ue2_between_ue1_tx = np.array(
        [
            r["ue2_between_ue1_tx"]
            for r in records
        ],
        dtype=np.int8
    )


    ue2_los_distance = np.array(
        [
            r["ue2_los_distance"]
            for r in records
        ],
        dtype=np.float32
    )


    ue1_dir_sin = np.array(
        [
            r["ue1_dir_sin"]
            for r in records
        ],
        dtype=np.float32
    )


    ue1_dir_cos = np.array(
        [
            r["ue1_dir_cos"]
            for r in records
        ],
        dtype=np.float32
    )


    ue1_linear_velocity = np.array(
        [
            r["ue1_linear_velocity"]
            for r in records
        ],
        dtype=np.float32
    )


    ue1_angular_velocity = np.array(
        [
            r["ue1_angular_velocity"]
            for r in records
        ],
        dtype=np.float32
    )


    ue2_dir_sin = np.array(
        [
            r["ue2_dir_sin"]
            for r in records
        ],
        dtype=np.float32
    )


    ue2_dir_cos = np.array(
        [
            r["ue2_dir_cos"]
            for r in records
        ],
        dtype=np.float32
    )


    ue2_linear_velocity = np.array(
        [
            r["ue2_linear_velocity"]
            for r in records
        ],
        dtype=np.float32
    )


    ue2_angular_velocity = np.array(
        [
            r["ue2_angular_velocity"]
            for r in records
        ],
        dtype=np.float32
    )


    np.savez(
        filename,

        slot=slots,
        time_s=time_s,

        position=position,
        rx_position=rx_position,

        heading_rad=heading_rad,
        heading_deg=heading_deg,

        human2_position=
            human2_position,

        human2_heading_rad=
            human2_heading_rad,

        human2_heading_deg=
            human2_heading_deg,

        csi_real=csi_real,
        csi_imag=csi_imag,

        throughput_mbps=
            throughput_mbps,

        throughput_window_mbps=
            throughput_window_mbps,

        phase=phase,

        ue1_phase=ue1_phase,

        human2_phase=
            human2_phase,

        ue1_speed_mps=
            ue1_speed_mps,

        human2_speed_mps=
            human2_speed_mps,

        inter_user_blockage=
            inter_user_blockage,

        tx_in_ue1_view=
            tx_in_ue1_view,

        tx_view_angle_deg=
            tx_view_angle_deg,

        ue2_in_ue1_front=
            ue2_in_ue1_front,

        ue2_between_ue1_tx=
            ue2_between_ue1_tx,

        ue2_los_distance=
            ue2_los_distance,

        ue1_dir_sin=
            ue1_dir_sin,

        ue1_dir_cos=
            ue1_dir_cos,

        ue1_linear_velocity=
            ue1_linear_velocity,

        ue1_angular_velocity=
            ue1_angular_velocity,

        ue2_dir_sin=
            ue2_dir_sin,

        ue2_dir_cos=
            ue2_dir_cos,

        ue2_linear_velocity=
            ue2_linear_velocity,

        ue2_angular_velocity=
            ue2_angular_velocity
    )


    print(
        f"saved: {filename}"
    )


# ============================================================
# Main
# ============================================================

def main():

    output_dir = (
        "dataset_v2"
    )


    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for i in range(469, 500):

        print(
            f"\n========== "
            f"Episode {i:03d} "
            f"=========="
        )


        records = (
            data_generate()
        )


        records = (
            add_throughput_window(
                records
            )
        )


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
