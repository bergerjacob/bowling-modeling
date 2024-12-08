import math
import matplotlib.pyplot as plt

# =========================================
# Constants and Parameters
# =========================================
g = 9.81
w_b = 0.0254  # Width of one board in meters

lane_length = 18.29  # 60 ft in meters
L_oil = 12.19         # Oil pattern length in meters

# Oil pattern data
oil_pattern_data = [
    {'direction': 'Forward', 'left_board': 3,  'right_board': 3,  'loads': 1},
    {'direction': 'Forward', 'left_board': 7,  'right_board': 7,  'loads': 1},
    {'direction': 'Forward', 'left_board': 8,  'right_board': 8,  'loads': 2},
    {'direction': 'Forward', 'left_board': 10, 'right_board': 10, 'loads': 3},
    {'direction': 'Forward', 'left_board': 11, 'right_board': 11, 'loads': 3},
    {'direction': 'Reverse', 'left_board': 6,  'right_board': 6,  'loads': 1},
    {'direction': 'Reverse', 'left_board': 10, 'right_board': 10, 'loads': 3},
    {'direction': 'Reverse', 'left_board': 11, 'right_board': 11, 'loads': 3},
    {'direction': 'Reverse', 'left_board': 13, 'right_board': 13, 'loads': 3},
]

mu_min = 0.02
mu_max = 0.08

K = 0.2
alpha_deg = 5.0  # Axis tilt in degrees - desired direction
alpha = math.radians(alpha_deg)

initial_ball_speed = 6.0   # m/s
phi0 = 0.0                 # Initially straight down the lane
starting_board = 5
y0 = starting_board * w_b

dt = 0.01


# =========================================
# Functions
# =========================================

def compute_oil_distribution(oil_data):
    load_map = {}
    for entry in oil_data:
        b_left = entry['left_board']
        b_right = entry['right_board']
        loads = entry['loads']
        for b in range(b_left, b_right + 1):
            load_map[b] = load_map.get(b, 0) + loads
    if load_map:
        max_loads = max(load_map.values())
        for b in load_map:
            load_map[b] = load_map[b] / max_loads
    return load_map

oil_map = compute_oil_distribution(oil_pattern_data)

def mu_of_position(x, y):
    b = int(math.floor(y/w_b))
    # Clamp boards to [1,39]
    if b < 1:
        b = 1
    if b > 39:
        b = 39
    o_b = oil_map.get(b, 0.0)
    if x <= L_oil:
        return mu_min + (mu_max - mu_min) * (1 - o_b)
    else:
        return mu_max

def equations_of_motion(t, state, friction_func):
    # State: [v, phi, x, y]
    v, phi, x, y = state
    mu = friction_func(x, y)
    
    dv_dt = -mu * g
    dphi_dt = K * (alpha - phi)
    dx_dt = v * math.cos(phi)
    dy_dt = v * math.sin(phi)
    
    return [dv_dt, dphi_dt, dx_dt, dy_dt]

def runge_kutta_4_step(t, state, dt, friction_func):
    k1 = equations_of_motion(t, state, friction_func)
    k1_state = [s + (dt/2)*k for s, k in zip(state, k1)]
    
    k2 = equations_of_motion(t + dt/2, k1_state, friction_func)
    k2_state = [s + (dt/2)*k for s, k in zip(state, k2)]
    
    k3 = equations_of_motion(t + dt/2, k2_state, friction_func)
    k3_state = [s + dt*k for s, k in zip(state, k3)]
    
    k4 = equations_of_motion(t + dt, k3_state, friction_func)
    
    new_state = [
        s + (dt/6)*(k1_i + 2*k2_i + 2*k3_i + k4_i)
        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4)
    ]
    return new_state

def simulate(friction_func):
    state = [initial_ball_speed, phi0, 0.0, y0]
    t = 0.0
    trajectory = [(t, state[2], state[3])]
    
    while state[2] < lane_length and state[0] > 0.1:
        state = runge_kutta_4_step(t, state, dt, friction_func)
        t += dt
        trajectory.append((t, state[2], state[3]))
    
    return state, trajectory

def constant_friction(x, y):
    # No oil pattern, just minimal friction
    return mu_min

def main():
    # Run simulation with oil pattern
    state_oil, trajectory_oil = simulate(mu_of_position)
    v_oil, phi_oil, x_oil, y_oil = state_oil
    final_angle_deg_oil = math.degrees(phi_oil)
    
    print("With Oil Pattern:")
    print(f"Final downlane position: {x_oil:.2f} m")
    print(f"Final lateral position: {y_oil:.2f} m")
    print(f"Final heading angle: {final_angle_deg_oil:.3f} deg")
    print(f"Final velocity: {v_oil:.2f} m/s")
    
    # Run simulation without oil pattern (constant friction)
    state_const, trajectory_const = simulate(constant_friction)
    v_const, phi_const, x_const, y_const = state_const
    final_angle_deg_const = math.degrees(phi_const)
    
    print("\nWithout Oil Pattern (Constant Friction):")
    print(f"Final downlane position: {x_const:.2f} m")
    print(f"Final lateral position: {y_const:.2f} m")
    print(f"Final heading angle: {final_angle_deg_const:.3f} deg")
    print(f"Final velocity: {v_const:.2f} m/s")

    # Print some friction values to verify oil pattern influence
    # For example, friction at start (x=0) and near the pins (x=18.29, board 20)
    print("\nFriction Checks:")
    print(f"Friction at start (x=0, board=20): {mu_of_position(0, 20*w_b):.3f}")
    print(f"Friction at pins (x={lane_length}, board=20): {mu_of_position(lane_length, 20*w_b):.3f}")
    
    # Plot the results
    xs_oil = [p[1] for p in trajectory_oil]
    ys_oil = [p[2] for p in trajectory_oil]
    board_positions_oil = [y / w_b for y in ys_oil]

    xs_const = [p[1] for p in trajectory_const]
    ys_const = [p[2] for p in trajectory_const]
    board_positions_const = [y / w_b for y in ys_const]

    plt.plot(board_positions_oil, xs_oil, label='Trajectory with Oil Pattern')
    plt.plot(board_positions_const, xs_const, label='Trajectory without Oil Pattern', linestyle='--')

    plt.scatter([20], [lane_length], s=100, color='red', label='Headpin')

    plt.xlabel('Board Number')
    plt.ylabel('Downlane Distance (m)')
    plt.title('Hooking Model Comparison with/without Oil Pattern')

    # Flip the x-axis so board #1 is on the right and board #39 on the left
    plt.xlim(39, 0)
    plt.ylim(0, 20)

    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
