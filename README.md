# ⬡ Hex Neural Engine

A reinforcement learning environment and interactive web arena for the game of Hex. 

This project separates heavy GPU-accelerated model training from lightweight, CPU-optimized cloud inference. The underlying game engine is built in pure Python with zero UI dependencies, allowing for both million-game headless simulations and interactive human-vs-AI web deployment.

## 🚀 The Architecture
* **The Engine:** A purely mathematical, headless game logic module running O(N) pathfinding (Union-Find/DFS) to determine win states on a hexagonal lattice.
* **The Brain:** A Deep Reinforcement Learning policy network trained locally via self-play on GPU hardware.
* **The Arena:** A Streamlit-based graphical frontend that serves a frozen, optimized `ONNX` model for real-time inference against human players.

## 🛠️ Local Environment Setup (Mamba)

This project requires a robust Python environment to handle both deep learning and web rendering. We use `mamba` for fast dependency resolution.

1. **Clone the repository:**
```bash
git clone [https://github.com/yourusername/hex-zero.git](https://github.com/yourusername/hex-zero.git)
cd hex-zero
```

2. **Create the environment:**
```bash
mamba create -n hex-zero python=3.10
mamba activate hex-zero
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Interactive Arena:**
```bash
streamlit run app.py
```