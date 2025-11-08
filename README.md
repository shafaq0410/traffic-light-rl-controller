\# 🚦 Traffic Light Reinforcement Learning Controller



A self-learning traffic light system that uses Q-Learning to optimize traffic flow and reduce waiting times by 20-40% compared to traditional fixed-time controllers.

<img width="1867" height="563" alt="image" src="https://github.com/user-attachments/assets/d2e405a2-5d1e-4301-8002-a8995233105d" />



<img width="1867" height="720" alt="image" src="https://github.com/user-attachments/assets/9d81809f-3f88-492f-807d-78a091e07195" />



<img width="1784" height="746" alt="image" src="https://github.com/user-attachments/assets/8552b2bb-9c4c-4a89-89e7-b466e80cf02c" />



\## 🎯 Features



\- \*\*Q-Learning Agent\*\*: Tabular reinforcement learning for adaptive traffic control

\- \*\*Web Interface\*\*: Real-time training visualization and interactive simulation

\- \*\*Performance Comparison\*\*: Benchmarks against fixed-time traffic controllers

\- \*\*Live Visualization\*\*: Animated traffic intersection with queue monitoring

\- \*\*Training Dashboard\*\*: Real-time metrics and progress tracking



\## 🚀 Demo



\### Training Progress

The agent learns optimal traffic light timing through trial and error:

\- \*\*200 episodes\*\* of training

\- \*\*Q-Learning algorithm\*\* with epsilon-greedy exploration

\- \*\*Real-time metrics\*\* showing improvement over baseline



\### Results

\- ✅ \*\*20-40% reduction\*\* in average waiting time

\- ✅ \*\*Adaptive control\*\* responds to traffic patterns

\- ✅ \*\*Stable performance\*\* across various traffic conditions



\## 📦 Installation



\### Prerequisites

\- Python 3.8 or higher

\- pip package manager



\### Quick Start



1\. \*\*Clone the repository\*\*

```bash

git clone https://github.com/shafaq0410/traffic-light-rl-controller.git

cd traffic-light-rl-controller

```



2\. \*\*Install dependencies\*\*

```bash

pip install -r requirements.txt

```



3\. \*\*Run the web application\*\*

```bash

python app.py

```



4\. \*\*Open in browser\*\*

```

http://localhost:5000

```



\## 📁 Project Structure



```

traffic-light-rl-controller/

│

├── app.py                  # Flask backend server

├── templates/

│   └── index.html          # Web frontend interface

├── traffic\_light\_rl.py     # Standalone CLI version

├── requirements.txt        # Python dependencies

├── .gitignore             # Git ignore rules

└── README.md              # This file

```



\## 🎮 Usage



\### Web Application (Recommended)



1\. \*\*Start Training\*\*

&nbsp;  - Click "Start Training" button

&nbsp;  - Watch real-time progress (200 episodes, ~30-60 seconds)

&nbsp;  - View metrics updating live



2\. \*\*Run Simulation\*\*

&nbsp;  - After training completes, click "Run Simulation"

&nbsp;  - Watch the animated traffic intersection

&nbsp;  - See queue lengths and light changes in real-time



3\. \*\*Analyze Results\*\*

&nbsp;  - Compare RL agent vs fixed-time controllers

&nbsp;  - View training progress chart

&nbsp;  - Check improvement percentage



\### Standalone CLI Version



```bash

python traffic\_light\_rl.py

```



Generates:

\- `training\_progress.png` - Training curves

\- `comparison.png` - Performance comparison chart



\## 🧠 How It Works



\### Environment

\- \*\*State Space\*\*: Queue lengths (N, S, E, W), current phase, time in phase

\- \*\*Action Space\*\*: Keep current phase (0) or switch phase (1)

\- \*\*Reward\*\*: Negative sum of waiting vehicles with penalties for frequent switching



\### Agent

\- \*\*Algorithm\*\*: Q-Learning (Tabular)

\- \*\*Policy\*\*: Epsilon-greedy exploration

\- \*\*Learning Rate\*\*: 0.1

\- \*\*Discount Factor\*\*: 0.95

\- \*\*State Discretization\*\*: Queue lengths binned into categories



\### Training

\- 200 episodes of 1000 timesteps each

\- Epsilon decay from 1.0 to 0.01

\- Evaluation over 20 test episodes



\## 📊 Technical Details



\### State Representation

```python

\[queue\_N, queue\_S, queue\_E, queue\_W, current\_phase, time\_in\_phase]

```



\### Reward Function

```python

reward = -total\_waiting\_vehicles - switch\_penalty - overtime\_penalty

```



\### Q-Learning Update

```python

Q(s,a) ← Q(s,a) + α\[r + γ·max(Q(s',a')) - Q(s,a)]

```



\## 🔧 Configuration



\### Modify Training Parameters



In `app.py` or `traffic\_light\_rl.py`:



```python

\# Environment parameters

max\_steps = 1000           # Simulation length

arrival\_rate = 0.25        # Vehicle arrival probability

min\_green\_time = 5         # Minimum green light duration

max\_green\_time = 60        # Maximum green light duration



\# Agent parameters

alpha = 0.1                # Learning rate

gamma = 0.95               # Discount factor

epsilon\_decay = 0.995      # Exploration decay rate

```



\## 📈 Performance Metrics



| Controller | Avg Wait Time | Improvement |

|-----------|---------------|-------------|

| RL Agent | ~5,200s | Baseline |

| Fixed-20s | ~8,400s | +61% slower |

| Fixed-25s | ~7,900s | +52% slower |

| Fixed-30s | ~8,200s | +58% slower |

| Fixed-40s | ~9,100s | +75% slower |



\*Results from 1000-step simulations, averaged over 20 episodes\*



\## 🛠️ Technologies Used



\- \*\*Python 3.8+\*\*: Core programming language

\- \*\*NumPy\*\*: Numerical computations

\- \*\*Matplotlib\*\*: Data visualization (CLI version)

\- \*\*Flask\*\*: Web server backend

\- \*\*Chart.js\*\*: Interactive charts (web version)

\- \*\*HTML/CSS/JavaScript\*\*: Frontend interface



\## 🤝 Contributing



Contributions are welcome! Here are some ideas:



\- \[ ] Add rush hour traffic patterns

\- \[ ] Implement emergency vehicle priority

\- \[ ] Extend to multiple intersection networks

\- \[ ] Add Deep Q-Network (DQN) variant

\- \[ ] Create pedestrian crossing logic

\- \[ ] Add weather/time-of-day variations



\### How to Contribute



1\. Fork the repository

2\. Create a feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## 📝 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## 👤 Author



\*\*Shafaq\*\*

\- GitHub: \[@shafaq0410](https://github.com/shafaq0410)



\## 🙏 Acknowledgments



\- Inspired by real-world traffic optimization problems

\- Q-Learning algorithm from Sutton \& Barto's RL book

\- Built as a reinforcement learning educational project



\## 📚 References



\- \[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Sutton \& Barto

\- \[OpenAI Gym Documentation](https://gymnasium.farama.org/)

\- \[Traffic Signal Control using Reinforcement Learning](https://arxiv.org/abs/1903.04527)



\## 🐛 Known Issues



\- Training time may vary based on system performance

\- Web interface requires modern browser (Chrome/Firefox/Edge recommended)

\- Simulation speed depends on JavaScript execution



\## 📞 Support



If you encounter any issues or have questions:

1\. Check the \[Issues](https://github.com/shafaq0410/traffic-light-rl-controller/issues) page

2\. Open a new issue with detailed description

3\. Include error messages and system information



---



⭐ \*\*If you found this project helpful, please give it a star!\*\* ⭐

