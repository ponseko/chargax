import wandb
import jax

def wandb_logger(data, iteration):
    assert (
        "returned_episode_returns" in data
        and "returned_episode" in data
        and "timestep" in data
    ), "Missing keys in logging data. Is the environment wrapped with LogWrapper?"

    num_envs = data["timestep"].shape[-1]
    mask = data["returned_episode"]

    return_values = jax.tree.map(lambda x: x[mask], data["returned_episode_returns"])
    timesteps = data["timestep"][mask] * num_envs

    # Other per-step metrics from get_info (profit, occupancy, etc.)
    extra_keys = [
        k for k in data.keys()
        if k not in ("returned_episode_returns", "returned_episode", "timestep")
    ]

    for t in range(len(timesteps)):
        log_dict = {"episodic_return": return_values[t].item()}
        for k in extra_keys:
            v = data[k]
            try:
                log_dict[k] = float(jax.numpy.mean(v))
            except Exception:
                pass  # skip anything that doesn't reduce cleanly
        wandb.log(log_dict, step=int(timesteps[t]))

   