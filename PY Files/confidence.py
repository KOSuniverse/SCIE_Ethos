# app/confidence.py
def score_ravc(recency: float, alignment: float, variance: float, coverage: float):
    # All in [0,1]; tune as needed
    raw = 0.25*(recency + alignment + (1-variance) + coverage)
    badge = "High" if raw >= 0.75 else "Medium" if raw >= 0.55 else "Low"
    return {"score": round(raw, 3), "badge": badge}

def should_abstain(score: float, threshold: float = 0.52) -> bool:
    return score < threshold
