# app/confidence.py
from typing import Dict, Any, List, Tuple

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

def requires_model_escalation(confidence_score: float, threshold: float = 0.55) -> bool:
    """Phase 3C: Check if query requires escalation to stronger model."""
    return confidence_score < threshold

def get_escalation_model(current_model: str = "gpt-4o-mini") -> str:
    """Phase 3C: Return stronger model for escalation."""
    escalation_map = {
        "gpt-4o-mini": "gpt-4o",
        "gpt-3.5-turbo": "gpt-4o",
        "gpt-4": "gpt-4o"
    }
    return escalation_map.get(current_model, "gpt-4o")

def calculate_ravc_confidence(
    retrieval_score: float = 0.8,
    analysis_quality: float = 0.9, 
    variance_consistency: float = 0.8,
    citation_coverage: float = 0.7
) -> Dict[str, Any]:
    """
    Phase 3C: Calculate R/A/V/C confidence score with detailed breakdown.
    
    R = Retrieval quality (KB hits, data freshness)
    A = Analysis alignment (intent match, numeric validation)
    V = Variance/consistency (low contradiction, stable results)
    C = Citation coverage (source diversity, authority)
    """
    # Apply orchestrator rules formula: 0.35*R + 0.25*A + 0.25*V + 0.15*C
    raw_score = (0.35 * retrieval_score + 
                 0.25 * analysis_quality + 
                 0.25 * (1 - variance_consistency) +  # Lower variance = higher confidence
                 0.15 * citation_coverage)
    
    # Determine badge and escalation
    if raw_score >= 0.75:
        badge = "High"
        css_class = "confidence-high"
        escalate = False
    elif raw_score >= 0.55:
        badge = "Medium" 
        css_class = "confidence-medium"
        escalate = False
    else:
        badge = "Low"
        css_class = "confidence-low"
        escalate = True
    
    return {
        "score": round(raw_score, 3),
        "badge": badge,
        "css_class": css_class,
        "escalate": escalate,
        "breakdown": {
            "retrieval": retrieval_score,
            "analysis": analysis_quality,
            "variance": variance_consistency,
            "coverage": citation_coverage
        },
        "formula": "0.35*R + 0.25*A + 0.25*V + 0.15*C"
    }

def calculate_ravc_confidence(query: str, sources: List[Any], execution_calls: List[Dict]) -> Tuple[float, Dict[str, float]]:
    """Calculate RAVC confidence score for DP orchestrator"""
    try:
        # R - Retrieval strength
        retrieval_strength = min(len(sources) / 4.0, 1.0)  # Normalize to 4+ sources
        
        # A - Agreement (based on execution success)
        successful_calls = len([c for c in execution_calls if not c.get("error")])
        total_calls = max(len(execution_calls), 1)
        agreement = successful_calls / total_calls
        
        # V - Validations (based on result quality)
        validations = 0.8 if successful_calls > 0 else 0.3
        
        # C - Citation density
        citation_density = min(len(sources) / 10.0, 1.0)  # Normalize to 10 sources
        
        # Calculate final score
        score = 0.35 * retrieval_strength + 0.25 * agreement + 0.25 * validations + 0.15 * citation_density
        
        ravc_breakdown = {
            "retrieval_strength_R": retrieval_strength,
            "agreement_A": agreement,
            "validations_V": validations,
            "citation_density_C": citation_density
        }
        
        return score, ravc_breakdown
        
    except Exception as e:
        print(f"Warning: RAVC calculation failed: {e}")
        # Return safe defaults
        return 0.7, {
            "retrieval_strength_R": 0.7,
            "agreement_A": 0.7,
            "validations_V": 0.7,
            "citation_density_C": 0.7
        }

def get_confidence_badge(confidence_score: float, ravc_breakdown: Dict[str, float] = None) -> str:
    """Generate confidence badge HTML for display"""
    try:
        # Determine confidence level
        if confidence_score >= 0.75:
            level = "High"
            css_class = "confidence-high"
            color = "#28a745"  # Green
        elif confidence_score >= 0.55:
            level = "Medium"
            css_class = "confidence-medium" 
            color = "#ffc107"  # Yellow
        else:
            level = "Low"
            css_class = "confidence-low"
            color = "#dc3545"  # Red
        
        # Build badge HTML
        badge_html = f"""
        <div class="confidence-badge {css_class}" style="
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            background-color: {color};
            color: white;
            font-weight: bold;
            font-size: 0.9em;
            margin: 4px;
        ">
            {level}: {confidence_score:.2f}
        </div>
        """
        
        # Add RAVC breakdown if available
        if ravc_breakdown:
            breakdown_html = "<div style='font-size: 0.8em; margin-top: 4px;'>"
            breakdown_html += f"R: {ravc_breakdown.get('retrieval_strength_R', 0):.2f} | "
            breakdown_html += f"A: {ravc_breakdown.get('agreement_A', 0):.2f} | "
            breakdown_html += f"V: {ravc_breakdown.get('validations_V', 0):.2f} | "
            breakdown_html += f"C: {ravc_breakdown.get('citation_density_C', 0):.2f}"
            breakdown_html += "</div>"
            badge_html += breakdown_html
        
        return badge_html
        
    except Exception as e:
        # Fallback badge
        return f"""
        <div class="confidence-badge confidence-medium" style="
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            background-color: #6c757d;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
            margin: 4px;
        ">
            Confidence: {confidence_score:.2f}
        </div>
        """