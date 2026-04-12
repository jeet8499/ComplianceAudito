TASK_GROUND_TRUTH = {
    "easy_1": {
        "regulation": "gdpr",
        "violations": {1: True},
        "clause_texts": [
            "We collect your email address.",
            "We may share your data with third parties without your consent.",
            "You can request deletion within 30 days."
        ]
    },
    "medium_1": {
        "regulation": "ccpa",
        "violations": {0: True, 3: True},
        "clause_texts": [
            "We sell your personal information to advertisers.",
            "You can opt out by clicking a link.",
            "We retain data for 5 years.",
            "We do not respond to deletion requests."
        ]
    },
    "hard_1": {
        "regulation": "gdpr",
        "violations": {1: 0.7, 4: 1.0},
        "clause_texts": [
            "We process data based on legitimate interest.",
            "We may transfer data to countries without adequacy decision if you agree.",
            "You have the right to access.",
            "We charge a fee for access requests.",
            "We keep data indefinitely without purpose limitation."
        ]
    }
}

def grade(task_id: str, agent_flags: dict) -> float:
    truth = TASK_GROUND_TRUTH[task_id]["violations"]
    if task_id == "hard_1":
        tp = 0.0
        fp = 0
        fn = 0
        for idx, truth_val in truth.items():
            agent_val = agent_flags.get(idx, False)
            if agent_val and truth_val > 0:
                tp += min(agent_val, truth_val)
            elif agent_val and truth_val == 0:
                fp += 1
            elif not agent_val and truth_val > 0:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    else:
        tp = sum(1 for idx, val in agent_flags.items() if truth.get(idx, False) and val)
        fp = sum(1 for idx, val in agent_flags.items() if not truth.get(idx, False) and val)
        fn = sum(1 for idx in truth if idx not in agent_flags or not agent_flags[idx])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
