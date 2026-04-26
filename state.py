from typing import TypedDict, Optional
from pydantic import BaseModel, Field, StrictBool, StrictStr


class ClinicalEvaluation(BaseModel):
    """Pydantic model representing Leqembi initial approval criteria evaluation."""
    
    matches_diagnosis: StrictBool = Field(
        ...,
        description="Whether the primary diagnosis code matches Leqembi criteria (G30.x for Alzheimer's)"
    )
    dementia_is_mild: StrictBool = Field(
        ...,
        description="Whether the dementia severity is mild"
    )
    amyloid_confirmed: StrictBool = Field(
        ...,
        description="Whether amyloid plaque has been confirmed"
    )
    has_recent_mri: StrictBool = Field(
        ...,
        description="Whether the patient has a recent MRI within the past year"
    )
    other_dementia_ruled_out: StrictBool = Field(
        ...,
        description="Whether other forms of dementia have been ruled out"
    )
    agrees_to_aria_monitoring: StrictBool = Field(
        ...,
        description="Whether the prescriber agrees to ARIA monitoring"
    )
    meets_all_criteria: StrictBool = Field(
        ...,
        description="Overall determination if all Leqembi approval criteria are met (True only if all above are true)"
    )


class State(TypedDict):
    """LangGraph state for the prior authorization workflow."""
    
    patient_data: dict
    policy_text: str
    evaluation_results: Optional[ClinicalEvaluation]
    next_step: Optional[str]
