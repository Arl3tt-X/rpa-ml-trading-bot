from models.reinforcement import ppo_agent, dqn_agent, backtesting
import shutil
import os

# Step 1: Train the model (choose PPO or DQN)
def train_model(strategy="ppo"):
    if strategy == "ppo":
        model, env = ppo_agent.train_model()
    else:
        model, env = dqn_agent.train_model()
    return model, env

# Step 2: Backtest the model
def run_backtest(model, env):
    metrics = backtesting.evaluate_model(model, env)
    return metrics

# Step 3: Evaluate metrics
def should_deploy(metrics, sharpe_threshold=1.0, profit_threshold=100):
    sharpe, profit = metrics.get("sharpe"), metrics.get("profit")
    if sharpe >= sharpe_threshold and profit >= profit_threshold:
        print(f"✅ Passed: Sharpe {sharpe:.2f}, Profit ${profit:.2f}")
        return True
    print(f"❌ Failed: Sharpe {sharpe:.2f}, Profit ${profit:.2f}")
    return False

# Step 4: Deploy model (copy to deployment folder)
def deploy_model(model_path, strategy):
    dest = f"./deployed_models/{strategy}_latest.zip"
    shutil.copy(model_path, dest)
    print(f"🚀 Model deployed to: {dest}")

# Main pipeline
if __name__ == "__main__":
    strategy = "ppo"  # or "dqn"
    model, env = train_model(strategy)
    metrics = run_backtest(model, env)
    if should_deploy(metrics):
        deploy_model(f"models/reinforcement/{strategy}_trader.zip", strategy)
