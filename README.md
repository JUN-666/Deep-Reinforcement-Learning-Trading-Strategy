# Deep Reinforcement Learning Trading Strategy

This project implements a High-Frequency Trading (HFT) agent using the Proximal Policy Optimization (PPO) reinforcement learning algorithm. The agent is trained and evaluated in a simulated trading environment.

## Project Structure

```
├── hft_agent/
│   ├── env/
│   │   └── trading_env.py        # Defines the trading environment, including state/action spaces and reward calculation.
│   ├── models/
│   │   └── actor_critic.py       # Contains the neural network architecture for the PPO agent's policy and value functions.
│   ├── ppo/
│   │   └── ppo_agent.py          # Implements the PPO algorithm, managing agent learning and updates.
│   ├── utils/                    # Utility functions for data loading, evaluation metrics, logging, etc.
│   └── quantization/             # (If known) Tools for model quantization (e.g., ptq.py) for optimizing model size and inference speed.
├── scripts/
│   ├── train.py                  # Script for initializing and running the agent training process.
│   └── evaluate.py               # Script for loading a trained agent and evaluating its performance on test data.
├── trained_models/               # Directory to save and load trained model checkpoints.
├── main.py                       # Main script to run the agent (Note: currently empty, its intended use is to provide a central entry point for running different modes of the application, such as training, evaluation, or live trading if implemented in the future.)
├── config.py                     # Configuration file for parameters (Note: currently empty, intended to hold project-wide parameters. Currently, parameters like learning rate, batch size, etc., are likely managed within the respective scripts or agent/environment constructors.)
├── requirements.txt              # Lists project dependencies.
└── README.md                     # This file.
```

## Getting Started

### Prerequisites

- Python 3.x
- pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd deep-reinforcement-learning-trading-strategy
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the agent, run the `train.py` script:
```bash
python scripts/train.py
```
You might need to configure training parameters. These may be located in `scripts/train.py`, within the agent or environment constructors, or eventually in `config.py`.

### Evaluation

To evaluate a trained agent, run the `evaluate.py` script, specifying the path to the trained model:
```bash
python scripts/evaluate.py --model_path trained_models/ppo_actor_critic_hft.pth
```
Evaluation parameters might also need configuration, similar to training parameters.

### Configuration

The `config.py` file is intended for project-wide parameters. Although currently empty, it may be populated in the future to centralize configuration. For now, key parameters for training and evaluation can typically be found and modified within `scripts/train.py` and `scripts/evaluate.py`, or directly within the agent and environment constructor arguments.

**Examples of parameters you might configure:**

*   **In `scripts/train.py` (or PPO agent constructor):**
    *   `learning_rate`: The learning rate for the optimizer (e.g., Adam).
    *   `gamma`: Discount factor for future rewards.
    *   `ppo_epsilon`: Clipping parameter for PPO's objective function.
    *   `ppo_epochs`: Number of epochs to update the policy for each batch of experience.
    *   `batch_size` (or `rollout_length` / `n_steps`): Number of timesteps to collect before an update.
    *   `num_episodes` or `total_timesteps`: Duration of training.
*   **In `scripts/train.py` or `scripts/evaluate.py` (or environment constructor):**
    *   `data_file_path`: Path to the market data used for training or evaluation.
    *   `initial_capital`: Starting capital for the trading agent.
    *   `max_position`: Maximum number of shares the agent can hold.
    *   `transaction_cost_config`: Parameters defining transaction costs (e.g., fixed cost, percentage of trade value).
    *   `pnl_magnification_C`: A constant to scale the Profit and Loss (PnL) for reward calculation.

It is recommended to review these scripts and the constructors of `PPOAgent` and `TradingEnv` to understand all available configurable parameters. The `config.py` file remains a placeholder for future refactoring to centralize these settings.

## Core Components

### PPO Agent (`hft_agent/ppo/ppo_agent.py`)

-   Implements the Proximal Policy Optimization (PPO) algorithm, a policy gradient method known for its stability and performance.
-   The implementation likely involves collecting rollouts of experience from the environment, then updating the policy and value functions for multiple epochs using these rollouts.
-   It uses a clipping mechanism (defined by `ppo_epsilon`) in the objective function to prevent excessively large policy updates, which contributes to its stability.
-   Generalized Advantage Estimation (GAE) might be used for calculating advantages, providing a balance between bias and variance in the advantage estimates.
-   The agent utilizes an Actor-Critic model, defined in `hft_agent/models/actor_critic.py`. This model typically consists of two neural networks:
    *   **Actor Network**: Outputs the policy (i.e., probabilities of taking each action).
    *   **Critic Network**: Estimates the value function (i.e., expected return from a given state).

### Trading Environment (`hft_agent/env/trading_env.py`)

-   A custom OpenAI Gym-like environment simulating a high-frequency trading scenario. It manages market data, executes trades based on agent actions, calculates rewards, and provides observations to the agent.
-   **Observation Space**: The features provided to the agent at each timestep. This typically includes:
    *   Normalized current bid and ask prices.
    *   Current bid-ask spread.
    *   Agent's current position quantity (e.g., number of shares held, can be negative for short positions).
    *   Other market indicators or features derived from the price data might also be included.
-   **Action Space**: Defines the set of possible actions the agent can take. Common actions include:
    *   `Buy`: Execute a buy order at the current ask price.
    *   `Sell`: Execute a sell order at the current bid price.
    *   `Hold` / `No-Action`: Take no action in the current timestep.
    The specific size of buy/sell orders might be fixed or determined by the agent's policy.
-   **Reward Calculation**: Rewards are typically calculated based on the change in the agent's portfolio value (Profit and Loss - PnL).
    *   Transaction costs (e.g., a fixed fee per trade or a percentage of the trade value) are usually deducted, realistically penalizing frequent trading.
    *   The PnL might be magnified by a constant (`pnl_magnification_C`) to adjust the scale of the reward signal for more effective learning.
    *   Rewards might be normalized or shaped to guide learning (e.g., using Sharpe ratio or penalties for excessive risk).
-   **Key Configurable Parameters**:
    *   `max_position`: The maximum absolute number of shares the agent can hold (long or short).
    *   `transaction_cost_config`: A dictionary or object specifying the fixed and/or percentage-based transaction costs.
    *   `pnl_magnification_C`: A scalar used to amplify the raw PnL to make the reward signal more significant for the agent.
    *   `data_source`: Configuration for loading market data, including file paths and relevant columns.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.