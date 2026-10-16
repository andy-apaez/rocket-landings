=================== Rocket spec (fixed per rocket) ===================
Dry mass (kg)
Fuel capacity (kg)
Length (m)
Radius (m)
Max thrust t_max (N)
Minimum throttle (fraction, e.g. 0.4)
Specific impulse isp (s)
Max gimbal angle (rad)
Gimbal rate limit (rad/s)
Throttle rate limit (per s)
Drag coefficient cd
Reference area (m², usually π·radius²)
Leg radius (m, distance from center axis)
Leg drop (m, how far below the base)
RCS roll torque (N·m), if using roll control
Fuel center-of-mass shift factor

=================== Derived each step (from spec + fuel) ===================

Total mass
Inertia tensor (side and roll moments)
Center of mass offset
Engine lever arm (center of mass to engine)
Thrust-to-weight ratio (for tuning checks)

=================== Rocket state (changes every step) ===================

Position (x, y, z)
Velocity (vx, vy, vz)
Orientation quaternion (w, x, y, z)
Angular velocity (ωx, ωy, ωz), body frame
Fuel mass remaining
Current throttle
Current gimbal angles (δx, δz)

=================== Network outputs (controls) ===================

Target throttle [0, 1]
Target gimbal (δx, δz)
RCS roll command [-1, 1], optional

=================== Environment / celestial body ===================

Local gravity (m/s²)
Air density (kg/m³, 0 for the Moon)
Standard gravity g₀ = 9.81 for the fuel burn formula (never changes)
Engine Isp adjustment for vacuum, optional
Wind, optional

=================== Terrain ===================

Heightmap grid
World size (m)
Ground height at a point (lookup)
Surface normal / slope at a point
Landing pad location and radius, if using a fixed pad

=================== Simulation settings ===================

Timestep dt (0.01–0.05 s)
Sub-steps per network decision
Max episode length (time limit)
World bounds
Batch size N (parallel rockets)
Reset randomization ranges: starting position, velocity, tilt, fuel, terrain

=================== Landing success criteria ===================

Max vertical speed (relative to surface normal)
Max horizontal speed
Max tilt from vertical
Max angular velocity
Max distance from pad
Max surface slope
Max height difference between legs

=================== Crash / termination conditions ===================

Any leg or body touches ground outside success criteria
Tilt exceeds limit mid-flight
Out of bounds
Time limit reached
Fuel empty, optional

=================== Reward terms ===================

Per step: distance to pad, speed, tilt, angular velocity, fuel use (all small penalties)
Terminal: landing bonus, crash penalty, fuel-remaining bonus, slope penalty

=================== Network observation (inputs, all normalized) ===================

Position relative to pad
Velocity
Orientation (quaternion or up-vector)
Angular velocity
Fuel fraction
Current throttle and gimbal
Altitude above ground directly below
Terrain info: local slope, raycast heights, or heightmap patch
Gravity and air density, if training across multiple bodies
