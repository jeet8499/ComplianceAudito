# Compliance Auditor Environment

Real-world OpenEnv environment for auditing documents against GDPR/CCPA.

## Action Space
- `flag` (clause_index, is_violation)
- `skip` (clause_index)
- `finalize`

## Observation
- clauses, remaining_clauses, difficulty, regulation

## Tasks & Graders
| Difficulty | Description | Grader |
|------------|-------------|--------|
| Easy | One clear violation | Binary F1 |
| Medium | Two violations | Binary F1 |
| Hard | Partial violation + full | Weighted F1 |

## Baseline Scores (gpt-3.5-turbo)
- EASY: 1.000
- MEDIUM: 0.667
- HARD: 0.350

## Setup
```bash
export OPENAI_API_KEY=your_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-3.5-turbo
python inference.py