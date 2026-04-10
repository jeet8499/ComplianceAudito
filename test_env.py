from environment import ComplianceAuditorEnv
from models import TaskDifficulty, Action

env = ComplianceAuditorEnv(TaskDifficulty.EASY)
obs = env.reset()
print("Clauses:", [(c.index, c.text) for c in obs.clauses])

# Flag the correct violation (clause 1)
action = Action(action_type="flag", clause_index=1, is_violation=True)
obs, reward, done, info = env.step(action)
print(f"After flag: reward={reward:.3f}, info={info}")

# Finalize
action = Action(action_type="finalize")
obs, reward, done, info = env.step(action)
print(f"Final reward: {reward:.3f}, final_score={info.get('final_score')}")