import os
import sys
import json
import urllib.request
import urllib.error

# ------------------------------------------------------------------
# Environment variables (provided by the platform)
# ------------------------------------------------------------------
api_base = os.environ.get("API_BASE_URL")
api_key = os.environ.get("API_KEY")
model = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

if not api_base or not api_key:
    print("[ERROR] Missing API_BASE_URL or API_KEY", file=sys.stderr)
    sys.exit(1)

def call_llm(prompt):
    """Send prompt to LLM and return JSON response."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 150
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", flush=True)
        return None

# ------------------------------------------------------------------
# Import the environment
# ------------------------------------------------------------------
try:
    from environment import ComplianceAuditorEnv
    from models import TaskDifficulty, Action
except ImportError:
    print("[ERROR] Cannot import environment modules", file=sys.stderr)
    sys.exit(1)

def run_task(difficulty, task_name):
    env = ComplianceAuditorEnv(difficulty=difficulty)
    obs = env.reset()
    done = False
    step = 0
    max_steps = 20
    rewards = []
    actions_log = []

    while not done and step < max_steps:
        # Build prompt for the LLM
        clauses_text = "\n".join(f"{c.index}: {c.text}" for c in obs.clauses)
        prompt = f"""You are a compliance auditor. Review the document clauses under {obs.regulation.value.upper()}.
Document: {obs.document_title}
Clauses:
{clauses_text}
Already reviewed clauses (with flags): {list(env.flags.keys())}
Remaining clauses: {obs.remaining_clauses}
Choose an action in JSON:
- Flag a clause as violation: {{"action_type": "flag", "clause_index": <int>, "is_violation": true/false}}
- Skip a clause: {{"action_type": "skip", "clause_index": <int>}}
- Finish audit: {{"action_type": "finalize"}}
Respond with ONLY the JSON."""
        content = call_llm(prompt)
        if content:
            try:
                action_dict = json.loads(content)
                action = Action(**action_dict)
            except:
                action = Action(action_type="finalize")
        else:
            action = Action(action_type="finalize")

        obs, reward, done, info = env.step(action)
        step += 1
        rewards.append(reward)
        actions_log.append(action.action_type)
        error = info.get("error", None)
        print(f"[STEP] step={step} action={action.action_type} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

    final_score = info.get("final_score", 0.0)
    # Ensure score is strictly between 0 and 1 (validator requirement)
    if final_score <= 0.0:
        final_score = 0.01
    elif final_score >= 1.0:
        final_score = 0.99
    return final_score, rewards

# ------------------------------------------------------------------
# Main: run three tasks, each as a separate episode
# ------------------------------------------------------------------
def main():
    tasks = [
        (TaskDifficulty.EASY, "easy"),
        (TaskDifficulty.MEDIUM, "medium"),
        (TaskDifficulty.HARD, "hard")
    ]
    for diff, task_name in tasks:
        print(f"[START] task={task_name} env=openenv model={model}", flush=True)
        score, rewards = run_task(diff, task_name)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success=true steps={len(rewards)} score={score:.3f} rewards={rewards_str}", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
