# """
# Traffic Light Reinforcement Learning Controller
# Complete implementation with training, evaluation, and visualization
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from collections import deque, defaultdict
# import time

# # ========================================
# # 1. TRAFFIC ENVIRONMENT
# # ========================================

# class TrafficLightEnv:
#     """Custom traffic light environment"""
    
#     def __init__(self, max_steps=1000):
#         self.max_steps = max_steps
#         self.reset()
    
#     def reset(self):
#         """Reset environment to initial state"""
#         self.queues = {'N': deque(), 'S': deque(), 'E': deque(), 'W': deque()}
#         self.current_phase = 0  # 0: NS green, 1: EW green
#         self.time_in_phase = 0
#         self.time_step = 0
#         self.total_waiting_time = 0
#         self.min_green_time = 5
#         self.max_green_time = 60
#         return self._get_state()
    
#     def _get_state(self):
#         """Get current state representation"""
#         return np.array([
#             len(self.queues['N']),
#             len(self.queues['S']),
#             len(self.queues['E']),
#             len(self.queues['W']),
#             self.current_phase,
#             min(self.time_in_phase, 60)
#         ], dtype=np.float32)
    
#     def _generate_vehicles(self):
#         """Simulate vehicle arrivals (Poisson process)"""
#         arrival_rate = 0.25  # vehicles per second per direction
#         for direction in ['N', 'S', 'E', 'W']:
#             if np.random.random() < arrival_rate:
#                 self.queues[direction].append(self.time_step)
    
#     def _process_traffic(self):
#         """Process vehicles through green lights"""
#         if self.current_phase == 0:
#             active_dirs = ['N', 'S']
#         else:
#             active_dirs = ['E', 'W']
        
#         for direction in active_dirs:
#             if len(self.queues[direction]) > 0:
#                 if np.random.random() < 0.6:  # 60% chance to pass
#                     arrival_time = self.queues[direction].popleft()
#                     wait_time = self.time_step - arrival_time
#                     self.total_waiting_time += wait_time
    
#     def step(self, action):
#         """Execute one time step"""
#         self.time_step += 1
#         self.time_in_phase += 1
        
#         # Generate new vehicles
#         self._generate_vehicles()
        
#         # Handle action
#         switched = False
#         if action == 1 and self.time_in_phase >= self.min_green_time:
#             self.current_phase = 1 - self.current_phase
#             self.time_in_phase = 0
#             switched = True
        
#         # Process traffic
#         self._process_traffic()
        
#         # Calculate reward
#         total_waiting = sum(len(q) for q in self.queues.values())
#         reward = -total_waiting
        
#         if switched:
#             reward -= 3  # penalty for switching
#         if self.time_in_phase > self.max_green_time:
#             reward -= 5  # penalty for holding too long
        
#         # Check if done
#         done = self.time_step >= self.max_steps
        
#         return self._get_state(), reward, done, {'total_wait': self.total_waiting_time}


# # ========================================
# # 2. Q-LEARNING AGENT
# # ========================================

# class QLearningAgent:
#     """Tabular Q-Learning agent"""
    
#     def __init__(self, n_actions=2):
#         self.q_table = defaultdict(lambda: np.zeros(n_actions))
#         self.n_actions = n_actions
        
#         # Hyperparameters
#         self.alpha = 0.1          # learning rate
#         self.gamma = 0.95         # discount factor
#         self.epsilon = 1.0        # exploration rate
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.995
    
#     def _discretize_state(self, state):
#         """Convert continuous state to discrete bins"""
#         bins = [5, 10, 20, 50]
#         discrete = []
        
#         # Discretize queue lengths
#         for i in range(4):
#             bin_idx = np.digitize(state[i], bins)
#             discrete.append(bin_idx)
        
#         # Add phase and time
#         discrete.append(int(state[4]))
#         discrete.append(min(int(state[5] // 10), 6))
        
#         return tuple(discrete)
    
#     def get_action(self, state, training=True):
#         """Select action using epsilon-greedy policy"""
#         state_key = self._discretize_state(state)
        
#         if training and np.random.random() < self.epsilon:
#             return np.random.randint(self.n_actions)
        
#         return np.argmax(self.q_table[state_key])
    
#     def update(self, state, action, reward, next_state, done):
#         """Update Q-values"""
#         state_key = self._discretize_state(state)
#         next_state_key = self._discretize_state(next_state)
        
#         # Q-learning update
#         current_q = self.q_table[state_key][action]
#         max_next_q = np.max(self.q_table[next_state_key])
#         new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
#         self.q_table[state_key][action] = new_q
        
#         # Decay epsilon
#         if done:
#             self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# # ========================================
# # 3. FIXED-TIME BASELINE
# # ========================================

# class FixedTimeController:
#     """Traditional fixed-time traffic light controller"""
    
#     def __init__(self, green_time=25):
#         self.green_time = green_time
    
#     def get_action(self, state):
#         """Simple fixed-time logic"""
#         time_in_phase = state[5]
#         if time_in_phase >= self.green_time:
#             return 1  # switch
#         return 0  # keep current


# # ========================================
# # 4. TRAINING FUNCTION
# # ========================================

# def train_agent(episodes=200, verbose=True):
#     """Train the Q-learning agent"""
#     env = TrafficLightEnv(max_steps=1000)
#     agent = QLearningAgent()
    
#     rewards_history = []
#     wait_times = []
    
#     print("🚀 Starting Training...")
#     print("-" * 60)
    
#     for episode in range(episodes):
#         state = env.reset()
#         episode_reward = 0
#         done = False
        
#         while not done:
#             action = agent.get_action(state, training=True)
#             next_state, reward, done, info = env.step(action)
#             agent.update(state, action, reward, next_state, done)
            
#             state = next_state
#             episode_reward += reward
        
#         rewards_history.append(episode_reward)
#         wait_times.append(info['total_wait'])
        
#         if verbose and (episode + 1) % 20 == 0:
#             avg_reward = np.mean(rewards_history[-20:])
#             avg_wait = np.mean(wait_times[-20:])
#             print(f"Episode {episode+1:3d} | "
#                   f"Avg Reward: {avg_reward:7.1f} | "
#                   f"Avg Wait: {avg_wait:7.1f}s | "
#                   f"Epsilon: {agent.epsilon:.3f}")
    
#     print("-" * 60)
#     print("✅ Training Complete!")
    
#     return agent, rewards_history, wait_times


# # ========================================
# # 5. EVALUATION FUNCTION
# # ========================================

# def evaluate_controller(controller, episodes=50, controller_name="Controller"):
#     """Evaluate a controller's performance"""
#     env = TrafficLightEnv(max_steps=1000)
#     wait_times = []
    
#     for _ in range(episodes):
#         state = env.reset()
#         done = False
        
#         while not done:
#             if isinstance(controller, QLearningAgent):
#                 action = controller.get_action(state, training=False)
#             else:
#                 action = controller.get_action(state)
            
#             state, _, done, info = env.step(action)
        
#         wait_times.append(info['total_wait'])
    
#     avg_wait = np.mean(wait_times)
#     std_wait = np.std(wait_times)
    
#     print(f"{controller_name:20s} | Avg Wait: {avg_wait:7.1f}s ± {std_wait:6.1f}s")
    
#     return avg_wait, std_wait, wait_times


# # ========================================
# # 6. VISUALIZATION
# # ========================================

# def plot_training_progress(rewards, wait_times):
#     """Plot training progress"""
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
#     # Plot rewards
#     ax1.plot(rewards, alpha=0.3, color='blue')
#     window = 20
#     smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
#     ax1.plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=2)
#     ax1.set_xlabel('Episode')
#     ax1.set_ylabel('Total Reward')
#     ax1.set_title('Training Rewards (20-episode moving average)')
#     ax1.grid(alpha=0.3)
    
#     # Plot wait times
#     ax2.plot(wait_times, alpha=0.3, color='red')
#     smoothed = np.convolve(wait_times, np.ones(window)/window, mode='valid')
#     ax2.plot(range(window-1, len(wait_times)), smoothed, color='red', linewidth=2)
#     ax2.set_xlabel('Episode')
#     ax2.set_ylabel('Total Waiting Time (s)')
#     ax2.set_title('Waiting Times (20-episode moving average)')
#     ax2.grid(alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
#     print("\n📊 Saved training progress to 'training_progress.png'")
#     plt.show()


# def plot_comparison(results):
#     """Plot comparison between controllers"""
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     controllers = list(results.keys())
#     means = [results[c][0] for c in controllers]
#     stds = [results[c][1] for c in controllers]
    
#     colors = ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
#     bars = ax.bar(controllers, means, yerr=stds, capsize=8, 
#                    color=colors[:len(controllers)], alpha=0.8, edgecolor='black')
    
#     ax.set_ylabel('Average Total Waiting Time (seconds)', fontsize=12)
#     ax.set_title('Traffic Controller Performance Comparison', fontsize=14, fontweight='bold')
#     ax.grid(axis='y', alpha=0.3)
    
#     # Add value labels on bars
#     for bar, mean in zip(bars, means):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{mean:.0f}s', ha='center', va='bottom', fontweight='bold')
    
#     plt.xticks(rotation=15, ha='right')
#     plt.tight_layout()
#     plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
#     print("📊 Saved comparison to 'comparison.png'")
#     plt.show()


# # ========================================
# # 7. MAIN EXECUTION
# # ========================================

# def main():
#     """Main function to run the complete project"""
    
#     print("\n" + "="*60)
#     print("🚦 TRAFFIC LIGHT REINFORCEMENT LEARNING CONTROLLER")
#     print("="*60 + "\n")
    
#     # Step 1: Train the RL agent
#     print("📚 PHASE 1: Training RL Agent")
#     agent, rewards, wait_times = train_agent(episodes=200, verbose=True)
    
#     # Step 2: Plot training progress
#     print("\n📈 PHASE 2: Plotting Training Progress")
#     plot_training_progress(rewards, wait_times)
    
#     # Step 3: Evaluate all controllers
#     print("\n🔍 PHASE 3: Evaluating Controllers")
#     print("-" * 60)
    
#     results = {}
    
#     # Evaluate RL agent
#     results['RL Agent'] = evaluate_controller(agent, episodes=50, controller_name="RL Agent")
    
#     # Evaluate fixed-time controllers
#     for green_time in [20, 25, 30, 40]:
#         fixed = FixedTimeController(green_time=green_time)
#         results[f'Fixed-{green_time}s'] = evaluate_controller(
#             fixed, episodes=50, controller_name=f"Fixed-{green_time}s"
#         )
    
#     print("-" * 60)
    
#     # Step 4: Calculate improvement
#     rl_avg = results['RL Agent'][0]
#     best_fixed_avg = results['Fixed-25s'][0]
#     improvement = ((best_fixed_avg - rl_avg) / best_fixed_avg) * 100
    
#     print(f"\n🎯 RESULTS:")
#     print(f"   RL Agent:       {rl_avg:.1f}s")
#     print(f"   Best Fixed:     {best_fixed_avg:.1f}s (25s cycle)")
#     print(f"   Improvement:    {improvement:.1f}% reduction in wait time")
    
#     # Step 5: Plot comparison
#     print("\n📊 PHASE 4: Plotting Comparison")
#     plot_comparison(results)
    
#     print("\n✅ All Done! Check the generated PNG files for visualizations.")
#     print("="*60 + "\n")


# if __name__ == "__main__":
#     main()

"""
Traffic Light RL Controller - Web Application
Flask backend with real-time visualization
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
from collections import deque, defaultdict
import json
import threading
import time

app = Flask(__name__)
CORS(app)

# Global variables for training state
training_state = {
    'is_training': False,
    'episode': 0,
    'total_episodes': 0,
    'history': [],
    'agent': None,
    'metrics': {
        'rl_wait': 0,
        'fixed_wait': 0,
        'improvement': 0
    }
}

# ========================================
# ENVIRONMENT
# ========================================

class TrafficLightEnv:
    def __init__(self, max_steps=1000):
        self.max_steps = max_steps
        self.reset()
    
    def reset(self):
        self.queues = {'N': deque(), 'S': deque(), 'E': deque(), 'W': deque()}
        self.current_phase = 0
        self.time_in_phase = 0
        self.time_step = 0
        self.total_waiting_time = 0
        self.min_green_time = 5
        self.max_green_time = 60
        return self._get_state()
    
    def _get_state(self):
        return np.array([
            len(self.queues['N']),
            len(self.queues['S']),
            len(self.queues['E']),
            len(self.queues['W']),
            self.current_phase,
            min(self.time_in_phase, 60)
        ], dtype=np.float32)
    
    def _generate_vehicles(self):
        arrival_rate = 0.25
        for direction in ['N', 'S', 'E', 'W']:
            if np.random.random() < arrival_rate:
                self.queues[direction].append(self.time_step)
    
    def _process_traffic(self):
        if self.current_phase == 0:
            active_dirs = ['N', 'S']
        else:
            active_dirs = ['E', 'W']
        
        for direction in active_dirs:
            if len(self.queues[direction]) > 0:
                if np.random.random() < 0.6:
                    arrival_time = self.queues[direction].popleft()
                    wait_time = self.time_step - arrival_time
                    self.total_waiting_time += wait_time
    
    def step(self, action):
        self.time_step += 1
        self.time_in_phase += 1
        
        self._generate_vehicles()
        
        switched = False
        if action == 1 and self.time_in_phase >= self.min_green_time:
            self.current_phase = 1 - self.current_phase
            self.time_in_phase = 0
            switched = True
        
        self._process_traffic()
        
        total_waiting = sum(len(q) for q in self.queues.values())
        reward = -total_waiting
        
        if switched:
            reward -= 3
        if self.time_in_phase > self.max_green_time:
            reward -= 5
        
        done = self.time_step >= self.max_steps
        
        return self._get_state(), reward, done, {'total_wait': self.total_waiting_time}

# ========================================
# Q-LEARNING AGENT
# ========================================

class QLearningAgent:
    def __init__(self, n_actions=2):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.n_actions = n_actions
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def _discretize_state(self, state):
        bins = [5, 10, 20, 50]
        discrete = []
        for i in range(4):
            bin_idx = np.digitize(state[i], bins)
            discrete.append(bin_idx)
        discrete.append(int(state[4]))
        discrete.append(min(int(state[5] // 10), 6))
        return tuple(discrete)
    
    def get_action(self, state, training=True):
        state_key = self._discretize_state(state)
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state_key][action] = new_q
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ========================================
# TRAINING FUNCTIONS
# ========================================

def train_agent_background(episodes=200):
    """Train agent in background thread"""
    global training_state
    
    training_state['is_training'] = True
    training_state['total_episodes'] = episodes
    training_state['history'] = []
    
    env = TrafficLightEnv(max_steps=1000)
    agent = QLearningAgent()
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
        
        training_state['episode'] = episode + 1
        training_state['history'].append({
            'episode': episode + 1,
            'reward': float(episode_reward),
            'wait_time': float(info['total_wait']),
            'epsilon': float(agent.epsilon)
        })
        
        time.sleep(0.01)  # Small delay for smoother updates
    
    # Evaluate
    rl_waits = []
    for _ in range(20):
        env_eval = TrafficLightEnv(max_steps=1000)
        state = env_eval.reset()
        done = False
        while not done:
            action = agent.get_action(state, training=False)
            state, _, done, info = env_eval.step(action)
        rl_waits.append(info['total_wait'])
    
    fixed_waits = []
    for _ in range(20):
        env_eval = TrafficLightEnv(max_steps=1000)
        state = env_eval.reset()
        done = False
        time_in_phase = 0
        while not done:
            time_in_phase += 1
            action = 1 if time_in_phase >= 25 else 0
            if action == 1:
                time_in_phase = 0
            state, _, done, info = env_eval.step(action)
        fixed_waits.append(info['total_wait'])
    
    rl_avg = np.mean(rl_waits)
    fixed_avg = np.mean(fixed_waits)
    improvement = ((fixed_avg - rl_avg) / fixed_avg) * 100
    
    training_state['metrics'] = {
        'rl_wait': float(rl_avg),
        'fixed_wait': float(fixed_avg),
        'improvement': float(improvement)
    }
    training_state['agent'] = agent
    training_state['is_training'] = False

# ========================================
# FLASK ROUTES
# ========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_training', methods=['POST'])
def start_training():
    data = request.json
    episodes = data.get('episodes', 200)
    
    if training_state['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    thread = threading.Thread(target=train_agent_background, args=(episodes,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'episodes': episodes})

@app.route('/api/training_status')
def training_status():
    return jsonify({
        'is_training': training_state['is_training'],
        'episode': training_state['episode'],
        'total_episodes': training_state['total_episodes'],
        'metrics': training_state['metrics']
    })

@app.route('/api/training_history')
def training_history():
    return jsonify({'history': training_state['history']})

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Run a single simulation episode"""
    env = TrafficLightEnv(max_steps=100)
    state = env.reset()
    
    states_log = []
    done = False
    
    use_rl = request.json.get('use_rl', True)
    time_in_phase = 0
    
    while not done:
        if use_rl and training_state['agent']:
            action = training_state['agent'].get_action(state, training=False)
        else:
            time_in_phase += 1
            action = 1 if time_in_phase >= 25 else 0
            if action == 1:
                time_in_phase = 0
        
        states_log.append({
            'queues': {
                'N': len(env.queues['N']),
                'S': len(env.queues['S']),
                'E': len(env.queues['E']),
                'W': len(env.queues['W'])
            },
            'phase': int(env.current_phase),
            'action': int(action)
        })
        
        state, _, done, _ = env.step(action)
    
    return jsonify({'states': states_log})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚦 TRAFFIC LIGHT RL CONTROLLER - WEB APP")
    print("="*60)
    print("\n🌐 Starting server at http://localhost:5000")
    print("📱 Open your browser and visit: http://localhost:5000")
    print("\n✨ Features:")
    print("   - Real-time training visualization")
    print("   - Interactive traffic simulation")
    print("   - Performance comparison charts")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, port=5000, use_reloader=False)