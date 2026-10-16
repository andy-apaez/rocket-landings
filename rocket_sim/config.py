"""Configuration dataclasses. Each class maps to a section of notes.md."""

from dataclasses import dataclass, field
import math

G0 = 9.81  # standard gravity for the fuel burn formula


@dataclass
class RocketSpec:
    """Fixed per rocket."""
    dry_mass: float = 2000.0          # kg
    fuel_capacity: float = 1500.0     # kg
    length: float = 12.0              # m
    radius: float = 0.9               # m
    t_max: float = 60_000.0           # N
    min_throttle: float = 0.3         # fraction; below this the engine is off
    isp: float = 280.0                # s
    max_gimbal: float = 0.12          # rad
    gimbal_rate: float = 0.6          # rad/s
    throttle_rate: float = 1.5        # per s
    cd: float = 0.8
    ref_area: float | None = None     # m², defaults to π·radius²
    leg_radius: float = 2.5           # m from center axis
    leg_drop: float = 1.0             # m below the base
    rcs_roll_torque: float = 3000.0   # N·m (0 disables roll control)
    fuel_com_shift: float = 0.5       # 0 = fuel at geometric center, 1 = fuel at the base

    def __post_init__(self):
        if self.ref_area is None:
            self.ref_area = math.pi * self.radius ** 2


@dataclass
class Body:
    """Environment / celestial body."""
    name: str = "earth"
    gravity: float = 9.81             # m/s²
    air_density: float = 1.225        # kg/m³ (0 for the Moon)
    isp_scale: float = 1.0            # Isp multiplier (e.g. >1 for vacuum)
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)  # mean wind, m/s
    wind_gust: float = 0.0            # std-dev of random per-episode wind, m/s


EARTH = Body("earth", 9.81, 1.225, 1.0)
MARS = Body("mars", 3.71, 0.020, 1.08)
MOON = Body("moon", 1.62, 0.0, 1.1)
BODIES = {b.name: b for b in (EARTH, MARS, MOON)}


@dataclass
class TerrainConfig:
    grid: int = 64                    # heightmap resolution (grid x grid)
    world_size: float = 400.0         # m, square world centered at the origin
    amplitude: float = 6.0            # m, peak-ish height of the bumps
    octaves: int = 4
    pad_radius: float = 5.0           # m
    pad_range: float = 60.0           # pad center is drawn uniformly in ±pad_range
    flatten_radius: float = 12.0      # m, terrain is flattened around the pad
    randomize: bool = True            # new terrain each reset (else flat)


@dataclass
class SimConfig:
    dt: float = 0.02                  # s
    substeps: int = 5                 # physics steps per network decision
    max_time: float = 90.0            # s, episode time limit (low-g bodies descend slowly)
    world_height: float = 600.0       # m, ceiling for out-of-bounds
    num_envs: int = 64                # batch size N
    # Reset: randomization ranges
    start_offset: float = 80.0        # horizontal offset from pad, ± m
    start_altitude: tuple[float, float] = (120.0, 250.0)
    start_hvel: float = 10.0          # ± m/s horizontal
    start_vvel: tuple[float, float] = (-30.0, -5.0)
    start_tilt: float = 0.25          # rad, max initial tilt
    start_angvel: float = 0.05        # rad/s, ±
    start_fuel: tuple[float, float] = (0.6, 1.0)  # fraction of capacity
    bodies: tuple[str, ...] = ("earth",)          # sampled uniformly per episode
    terminate_on_fuel_empty: bool = False
    # Weather: set by orbit.site_conditions; None keeps the body's own wind
    wind: tuple[float, float, float] | None = None    # mean wind, m/s
    wind_gust: float | None = None                    # std-dev of per-episode wind, m/s


@dataclass
class LandingCriteria:
    max_vspeed: float = 2.0           # m/s, along surface normal
    max_hspeed: float = 1.0           # m/s, tangential to surface
    max_tilt: float = 0.17            # rad from vertical (~10°)
    max_angvel: float = 0.3           # rad/s
    max_pad_dist: float = 5.0         # m
    max_slope: float = 0.15           # rad
    max_leg_height_diff: float = 0.5  # m
    flight_tilt_limit: float = 1.2    # rad, crash if exceeded mid-flight


@dataclass
class RewardWeights:
    # Step: multiplied by the decision interval in seconds
    distance: float = 0.02
    speed: float = 0.02
    tilt: float = 0.5
    angvel: float = 0.5
    fuel: float = 1.0                 # per fraction of capacity burned
    time: float = 0.3                 # per second, discourages hovering
    # Shaping: potential-based, so progress toward the pad is positive
    shaping: float = 30.0
    # Terminal
    landing_bonus: float = 100.0
    crash_penalty: float = 100.0
    fuel_bonus: float = 20.0          # × remaining fuel fraction on landing
    slope_penalty: float = 50.0       # × slope (rad) on landing
    impact_penalty: float = 1.0       # × impact speed on crash
    timeout_penalty: float = 50.0     # running out of time counts as a (terminal) failure


@dataclass
class CatchConfig:
    """Catch tower geometry (used by both tasks, so the viewer draws it where the physics has it)
    and the criteria for the "catch" task."""
    # Tower
    tower_distance: float = 17.0      # m, pad center to tower center
    tower_direction: tuple[float, float] = (-0.7071, -0.7071)  # unit xz vector, pad -> tower
    tower_width: float = 3.2          # m, square lattice
    tower_height: float = 28.0        # m
    collide: bool = True              # tower is solid in the catch task
    # Catch
    base_clearance: float = 4.0       # m, booster base above the pad when held in the arms
    catch_below_top: float = 1.9      # m, catch point below the top of the booster
    catch_radius: float = 1.5         # m, horizontal slack between the arms
    height_tolerance: float = 1.0     # m, vertical slack around the catch height
    max_vspeed: float = 1.0           # m/s
    max_hspeed: float = 1.0           # m/s
    max_tilt: float = 0.1             # rad (~6°)
    max_angvel: float = 0.2           # rad/s
    hold_time: float = 0.5            # s the booster must stay in the zone while the arms close


@dataclass
class EnvConfig:
    task: str = "land"                # "land" on legs, or "catch" with the tower arms
    rocket: RocketSpec = field(default_factory=RocketSpec)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    criteria: LandingCriteria = field(default_factory=LandingCriteria)
    reward: RewardWeights = field(default_factory=RewardWeights)
    catch: CatchConfig = field(default_factory=CatchConfig)

    def __post_init__(self):
        if self.task not in ("land", "catch"):
            raise ValueError(f"task must be 'land' or 'catch', not {self.task!r}")
