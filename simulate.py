import math
import matplotlib.pyplot as plt

# =========================================
# Constants and Parameters
# =========================================
g = 9.81
w_b = 0.0254    # Width of one board in meters

lane_length = 18.29  # 60 ft in meters
L_oil = 12.19         # Oil pattern length in meters

oil_pattern_data = [
    {'direction': 'Forward', 'left_board': 3,  'right_board': 3,  'loads': 1},
    {'direction': 'Forward', 'left_board': 7,  'right_board': 7,  'loads': 1},
    {'direction': 'Forward', 'left_board': 8,  'right_board': 8,  'loads': 2},
    {'direction': 'Forward', 'left_board': 10, 'right_board': 10, 'loads': 3},
    {'direction': 'Forward', 'left_board': 11, 'right_board': 11, 'loads': 3},
    {'direction': 'Reverse', 'left_board': 6,  'right_board': 6, 'loads': 1},
    {'direction': 'Reverse', 'left_board': 10, 'right_board': 10, 'loads': 3},
    {'direction': 'Reverse', 'left_board': 11, 'right_board': 11, 'loads': 3},
    {'direction': 'Reverse', 'left_board': 13, 'right_board': 13, 'loads': 3},
]

mu_min = 0.005
mu_max = 0.02

K = 0.005
initial_ball_speed = 9.0   # m/s
phi0 = math.radians(-1.5)                 
starting_board = 15
y0 = starting_board * w_b

dt = 0.01

# Ball parameters
m = 6.8          
diameter = 0.2159
R = diameter / 2
I = (2/5)*m*(R**2)  # Moment of inertia for a solid sphere

def compute_oil_distribution(oil_data):
    load_map = {}
    for entry in oil_data:
        for b in range(entry['left_board'], entry['right_board'] + 1):
            load_map[b] = load_map.get(b, 0) + entry['loads']
    if load_map:
        max_loads = max(load_map.values())
        for b in load_map:
            # Normalize by maximum loads
            load_map[b] = load_map[b] / max_loads
    else:
        load_map = {}
    return load_map

oil_map = compute_oil_distribution(oil_pattern_data)

def oil_fraction(x, y):
    # Compute board number b(t)
    b = int(math.floor(y / w_b))
    # Clamp board number between 1 and 39
    b = min(max(b, 1), 39)
    o_b = oil_map.get(b, 0.0)
    
    # o_x(x) = 1 if x <= L_oil else 0
    if x <= L_oil:
        o_x = 1.0
    else:
        o_x = 0.0
        
    # o(x,b) = o_b(b)*o_x(x)
    return o_b * o_x

def mu_of_position(x, y):
    o_val = oil_fraction(x, y)
    if x <= L_oil:
        return mu_min + (mu_max - mu_min)*(1 - o_val)
    else:
        return mu_max

def equations_of_motion(t, state):
    # State: [v, omega, phi, x, y]
    v, omega, phi, x, y = state
    mu = mu_of_position(x, y)
    
    dv_dt = -mu * g
    domega_dt = -(mu*m*g*R)/I
    # Modified line for correct hooking direction:
    dphi_dt = K * (omega - (v/R))
    dx_dt = v * math.cos(phi)
    dy_dt = v * math.sin(phi)
    
    return [dv_dt, domega_dt, dphi_dt, dx_dt, dy_dt]

def runge_kutta_4_step(t, state, dt):
    k1 = equations_of_motion(t, state)
    k1_state = [s + (dt/2)*kk for s, kk in zip(state, k1)]
    
    k2 = equations_of_motion(t + dt/2, k1_state)
    k2_state = [s + (dt/2)*kk for s, kk in zip(state, k2)]
    
    k3 = equations_of_motion(t + dt/2, k2_state)
    k3_state = [s + dt*kk for s, kk in zip(state, k3)]
    
    k4 = equations_of_motion(t + dt, k3_state)
    
    new_state = [
        s + (dt/6)*(k1_i + 2*k2_i + 2*k3_i + k4_i)
        for s, k1_i, k2_i, k3_i, k4_i in zip(state, k1, k2, k3, k4)
    ]
    return new_state

def simulate(initial_rev_rate):
    omega0 = -(initial_rev_rate * 2 * math.pi) / 60.0
    state = [initial_ball_speed, omega0, phi0, 0.0, y0]
    t = 0.0
    trajectory = [(t, *state)]  # (t, v, omega, phi, x, y)

    print("***state: ", state)
    
    print(f"--- Simulation for Rev Rate {initial_rev_rate} RPM ---")
    initial_mu = mu_of_position(state[3], state[4])
    print(f"Initial Friction (mu): {initial_mu:.4f}")
    print(f"Initial Oil Fraction: {oil_fraction(state[3], state[4]):.4f}")
    print(f"Start Board: {int(state[4] // w_b)}\n")
    
    while state[3] < lane_length and state[0] > 0.1:
        if round(t, 2) % 0.5 == 0:  # Print at regular intervals (0.5s)
            mu = mu_of_position(state[3], state[4])
            print(f"t={t:.2f}s: x={state[3]:.3f}, y={state[4]:.3f}, v={state[0]:.3f}, "
                  f"omega={state[1]:.3f}, phi={math.degrees(state[2]):.3f} deg, mu={mu:.4f}")
        
        state = runge_kutta_4_step(t, state, dt)
        t += dt
        trajectory.append((t, *state))
    
    final_mu = mu_of_position(state[3], state[4])
    print(f"\nFinal Friction (mu): {final_mu:.4f}")
    print(f"Final Oil Fraction: {oil_fraction(state[3], state[4]):.4f}")
    print(f"End Board: {int(state[4] // w_b)}")
    print(f"Final State: x={state[3]:.2f} m, y={state[4]:.2f} m, v={state[0]:.2f} m/s, "
          f"omega={state[1]:.2f} rad/s, phi={math.degrees(state[2]):.2f} deg\n")
    
    # Print state at key positions
    key_positions = [6.0, 12.0, 18.0]
    for pos in key_positions:
        closest = min(trajectory, key=lambda row: abs(row[3] - pos))
        mu_here = mu_of_position(closest[3], closest[4])
        print(f"At x~{pos:.1f}m: mu={mu_here:.4f}, v={closest[1]:.3f}, omega={closest[2]:.3f}, "
              f"phi={math.degrees(closest[3]):.3f} deg")
    
    return state, trajectory

def main():
    rev_rates = [100, 300, 500, 700]
    plt.figure(figsize=(6, 8))

    for rr in rev_rates:
        state, trajectory = simulate(rr)
        v_final, omega_final, phi_final, x_final, y_final = state
        final_angle_deg = math.degrees(phi_final)

        xs = [p[4] for p in trajectory]
        ys = [p[5] for p in trajectory]
        boards = [y / w_b for y in ys]

        plt.plot(boards, xs, label=f'{rr} RPM')

    plt.scatter([20], [lane_length], s=100, color='red', label='Headpin')
    plt.xlabel('Board Number')
    plt.ylabel('Downlane Distance (m)')
    plt.title('Trajectories with Updated Friction Logic')
    plt.xlim(39, 0)
    plt.ylim(0, 20)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

