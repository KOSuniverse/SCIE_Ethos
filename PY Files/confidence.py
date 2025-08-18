# app/confidence.py

def score_ravc(recency: float, alignment: float, variance: float, coverage: float):
    # All in [0,1]; tune as needed
    raw = 0.25*(recency + alignment + (1-variance) + coverage)
    badge = "High" if raw >= 0.75 else "Medium" if raw >= 0.55 else "Low"
    return {"score": round(raw, 3), "badge": badge}

def should_abstain(score: float, threshold: float = 0.52) -> bool:
    return score < threshold

def get_service_level_zscore(service_level: float) -> float:
    """
    Convert service level percentage to z-score.
    Service levels: 90%, 95%, 97.5%, 99%
    """
    service_level_map = {
        0.90: 1.645,
        0.95: 1.960,
        0.975: 2.241,
        0.99: 2.576
    }
    return service_level_map.get(service_level, 1.960)  # Default to 95%

def score_confidence_enhanced(recency: float, alignment: float, variance: float, coverage: float, service_level: float = 0.95):
    """
    Enhanced confidence scoring using orchestrator rules formula.
    Formula: 0.35*R + 0.25*A + 0.25*V + 0.15*C
    """
    # Apply the formula from orchestrator_rules.yaml
    raw_score = 0.35 * recency + 0.25 * alignment + 0.25 * (1 - variance) + 0.15 * coverage
    
    # Determine confidence level based on buckets
    if raw_score >= 0.75:
        badge = "High"
        css_class = "confidence-high"
    elif raw_score >= 0.55:
        badge = "Medium"
        css_class = "confidence-medium"
    else:
        badge = "Low"
        css_class = "confidence-low"
    
    # Calculate z-score for service level
    z_score = get_service_level_zscore(service_level)
    
    return {
        "score": round(raw_score, 3),
        "badge": badge,
        "css_class": css_class,
        "service_level": service_level,
        "z_score": z_score,
        "abstain": should_abstain(raw_score)
    }

def get_confidence_badge(score: float, service_level: float = 0.95) -> str:
    """Generate enhanced confidence badge with service level context."""
    # Use enhanced confidence scoring
    confidence_data = score_confidence_enhanced(
        recency=0.8,  # Placeholder values - should come from actual analysis
        alignment=0.9,
        variance=0.2,
        coverage=0.8,
        service_level=service_level
    )
    
    css_class = confidence_data.get("css_class", "confidence-medium")
    badge_text = confidence_data.get("badge", "MED")
    
    return f'<span class="badge {css_class}">{badge_text} ({score:.2f})</span>'
