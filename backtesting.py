import numpy as np

def evaluate_model(model, env):
    obs = env.reset()
    account_values = [env.envs[0].balance]
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        account_values.append(env.envs[0].balance)
        if done:
            break

    returns = np.diff(account_values)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
    profit = account_values[-1] - account_values[0]

    return {"sharpe": sharpe, "profit": profit}
