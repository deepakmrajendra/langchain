from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field, StrictBool


class CriterionEvaluation(BaseModel):
    """Evaluation of a single extracted clinical criterion."""
    criterion_name: str = Field(..., description="The specific rule extracted from the policy")
    met: StrictBool = Field(..., description="True if the patient explicitly meets this specific criterion. False otherwise.")
    missing_data: StrictBool = Field(..., description="True if there is not enough information in the patient data to determine if this criterion is met (e.g., null values).")
    reasoning: str = Field(..., description="Brief explanation based on patient data")


class ClinicalEvaluation(BaseModel):
    """Complete evaluation of all required criteria."""
    criteria_evaluations: List[CriterionEvaluation] = Field(
        ..., 
        description="List of all individual criteria evaluated"
    )
    meets_all_criteria: StrictBool = Field(
        ...,
        description="True ONLY if every single criterion in the criteria_evaluations list is met"
    )


class State(TypedDict):
    """LangGraph state for the prior authorization workflow."""
    
    patient_data: dict
    policy_text: str
    request_type: Optional[str]  # e.g., "Initial Approval" or "Continued Therapy"
    evaluation_results: Optional[ClinicalEvaluation]
    next_step: Optional[str]
