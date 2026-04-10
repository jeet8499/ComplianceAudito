import os
import json
from openai import OpenAI
from environment import ComplianceAuditorEnv
from models import TaskDifficulty, Action

# Use Groq's endpoint and a current free model
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")   # active model
HF_TOKEN = os.environ.get("HF_TOKEN", "")   # not needed for Groq

# Read the Groq API key from the environment
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=os.environ.get("GROQ_API_KEY")   # <-- changed from OPENAI_API_KEY
)
def run_agent(env: ComplianceAuditorEnv, max_steps=20):
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    while not done and step < max_steps:
        prompt = f"""You are a compliance auditor. Review the following document clauses and flag any that violate {obs.regulation.value.upper()}.

Document: {obs.document_title}
Clauses:
{chr(10).join(f"{c.index}: {c.text}" for c in obs.clauses)}

Already reviewed clauses: {list(env.flags.keys())}
Remaining clauses: {obs.remaining_clauses}

Choose an action:
- To flag a clause as violation: {{"action_type": "flag", "clause_index": <int>, "is_violation": true/false}}
- To skip (not flag) a clause: {{"action_type": "skip", "clause_index": <int>}}
- To finish: {{"action_type": "finalize"}}

Respond with ONLY the JSON action."""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        try:
            action_dict = json.loads(response.choices[0].message.content)
            action = Action(**action_dict)
        except:
            action = Action(action_type="finalize")

        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        print(f"Step {step}: reward={reward:.3f}, done={done}")

    final_score = info.get("final_score", 0.0)
    return final_score, total_reward

def main():
    scores = {}
    for difficulty in [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]:
        env = ComplianceAuditorEnv(difficulty=difficulty)
        final_score, _ = run_agent(env)
        scores[difficulty.value] = final_score
        print(f"{difficulty.value}: final score = {final_score:.3f}")

    print("\n=== Baseline Scores ===")
    for d, s in scores.items():
        print(f"{d.upper()}: {s:.3f}")

if __name__ == "__main__":
    main()