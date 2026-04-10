# Compliance Auditor Environment
---
title: Compliance Auditor Environment
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Compliance Auditor Environment

Real-world OpenEnv environment for auditing documents against GDPR/CCPA.

## Action Space
- `flag` (clause_index, is_violation) – mark a clause as violating or not
- `skip` (clause_index) – move on without flagging
- `finalize` – end the audit

## Observation Space
- `clauses`: list of (index, text)
- `remaining_clauses`: indices not yet reviewed
- `difficulty`, `regulation`, `step_count`, `max_steps`

## Tasks & Graders
| Difficulty | Description | Grader |
|------------|-------------|--------|
| Easy | One clear GDPR violation | Binary F1 |
| Medium | Two CCPA violations | Binary F1 |
| Hard | Partial (0.7) + full violation (1.0) | Weighted F1 |

## Baseline Scores (gpt-3.5-turbo)
- **EASY**: 1.000
- **MEDIUM**: 1.000
- **HARD**: 0.000

## Setup & Usage

### Local inference (requires OpenAI API key)
```bash
export OPENAI_API_KEY=your_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-3.5-turbo
python inference.py
