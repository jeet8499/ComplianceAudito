import uuid
from typing import Tuple, Dict, Any
from models import Observation, Action, TaskDifficulty, Regulation, Clause
from graders import TASK_GROUND_TRUTH, grade

class ComplianceAuditorEnv:
    def __init__(self, difficulty: TaskDifficulty = TaskDifficulty.EASY):
        self.difficulty = difficulty
        self.task_id = None
        self.regulation = None
        self.document_title = ""
        self.clauses = []
        self.max_steps = 20
        self.step_count = 0
        self.flags = {}            # clause_index -> is_violation
        self.explanations = {}     # clause_index -> explanation text (for 'explain' action)
        self._init_task()

    def _init_task(self):
        if self.difficulty == TaskDifficulty.EASY:
            self.task_id = "easy_1"
        elif self.difficulty == TaskDifficulty.MEDIUM:
            self.task_id = "medium_1"
        else:
            self.task_id = "hard_1"
        task_data = TASK_GROUND_TRUTH[self.task_id]
        self.regulation = Regulation(task_data["regulation"])
        self.document_title = f"Sample {self.regulation.value.upper()} Policy"
        self.clauses = [Clause(index=i, text=txt) for i, txt in enumerate(task_data["clause_texts"])]

    def reset(self) -> Observation:
        self._init_task()
        self.step_count = 0
        self.flags = {}
        self.explanations = {}
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = -0.01               # step penalty
        done = False
        info = {}

        # --- FLAG action (with optional confidence) ---
        if action.action_type == "flag":
            if action.clause_index is None or action.is_violation is None:
                reward -= 0.1
                info["error"] = "flag requires clause_index and is_violation"
            else:
                idx = action.clause_index
                if 0 <= idx < len(self.clauses):
                    if idx in self.flags:
                        reward -= 0.05
                        info["error"] = "clause already flagged"
                    else:
                        self.flags[idx] = action.is_violation
                        truth = TASK_GROUND_TRUTH[self.task_id]["violations"]
                        expected = truth.get(idx, False)
                        expected_bool = expected > 0.5 if isinstance(expected, float) else expected

                        # Base correctness reward
                        if action.is_violation == expected_bool:
                            base_reward = 0.3
                        else:
                            base_reward = -0.1

                        # Confidence bonus/penalty
                        confidence = getattr(action, 'confidence', 1.0)
                        if confidence is None:
                            confidence = 1.0
                        # Clamp confidence to [0,1]
                        confidence = max(0.0, min(1.0, confidence))

                        if action.is_violation == expected_bool:
                            # Correct flag: more reward if confident
                            reward += base_reward * confidence
                        else:
                            # Wrong flag: less penalty if low confidence
                            reward += base_reward * (1 - confidence)

                        info["confidence_used"] = confidence
                else:
                    reward -= 0.2
                    info["error"] = "invalid clause index"

        # --- SKIP action (no reward change) ---
        elif action.action_type == "skip":
            pass

        # --- EXPLAIN action (store explanation, no direct reward but adds transparency) ---
        elif action.action_type == "explain":
            if action.clause_index is not None and action.explanation:
                idx = action.clause_index
                if 0 <= idx < len(self.clauses):
                    self.explanations[idx] = action.explanation
                    info["explanation_stored"] = True
                    # Optional: give a tiny bonus for providing an explanation (encourages transparency)
                    reward += 0.02
                else:
                    reward -= 0.1
                    info["error"] = "invalid clause index for explanation"
            else:
                reward -= 0.1
                info["error"] = "explain requires clause_index and explanation text"

        # --- FINALIZE action (end episode, compute final score and bonuses) ---
        elif action.action_type == "finalize":
            done = True
            final_score = grade(self.task_id, self.flags)
            info["final_score"] = final_score
            reward += final_score

            # Bonus for reviewing all clauses (completeness)
            if len(self.flags) == len(self.clauses):
                reward += 0.2
                info["complete_review"] = True

            # Bonus for providing explanations on all flagged clauses (extra transparency)
            flagged_indices = set(self.flags.keys())
            explained_indices = set(self.explanations.keys())
            if flagged_indices and flagged_indices.issubset(explained_indices):
                reward += 0.1
                info["all_flagged_explained"] = True

        # --- Unknown action ---
        else:
            reward -= 0.2
            info["error"] = f"unknown action_type: {action.action_type}"

        # --- Max steps reached (force finalize) ---
        if self.step_count >= self.max_steps and not done:
            done = True
            final_score = grade(self.task_id, self.flags)
            info["final_score"] = final_score
            reward += final_score
            info["message"] = "Max steps reached"

        obs = self._get_obs()
        return obs, reward, done, info

    def _get_obs(self) -> Observation:
        reviewed = list(self.flags.keys())
        remaining = [c.index for c in self.clauses if c.index not in reviewed]
        return Observation(
            task_id=self.task_id,
            difficulty=self.difficulty,
            regulation=self.regulation,
            document_title=self.document_title,
            clauses=self.clauses,
            step_count=self.step_count,
            max_steps=self.max_steps,
            remaining_clauses=remaining
        )

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty.value,
            "regulation": self.regulation.value,
            "flags": self.flags,
            "explanations": self.explanations,
            "step_count": self.step_count
        }: self.step_count
        }
