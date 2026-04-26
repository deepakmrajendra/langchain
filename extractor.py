import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from state import State, ClinicalEvaluation


def evaluate_clinical_criteria(state: State) -> dict:
    """
    Evaluates clinical criteria against Leqembi policy using an LLM.
    
    Args:
        state: LangGraph State dictionary containing patient_data and policy_text
        
    Returns:
        Updated state dictionary with evaluation_results populated
    """
    
    # Initialize the chat model with structured output
    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        model=os.getenv("OPENROUTER_MODEL"),
        temperature=0
    ).with_structured_output(ClinicalEvaluation)
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical reviewer specializing in prior authorization determinations.

Current date: 26 April 2026

Your task is to evaluate patient data against the Leqembi policy criteria and determine if the patient meets all requirements for initial approval.

Carefully review each criterion:
- Diagnosis code must match G30.x (Alzheimer's disease)
- Dementia severity must be mild
- Amyloid plaque must be confirmed
- MRI must be within the past year (relative to current date: 26 April 2026)
- Other forms of dementia must be ruled out
- Prescriber must agree to ARIA monitoring

CRITICAL: You MUST return a complete evaluation containing exactly these boolean keys:
1. matches_diagnosis
2. dementia_is_mild
3. amyloid_confirmed
4. has_recent_mri
5. other_dementia_ruled_out
6. agrees_to_aria_monitoring
7. meets_all_criteria (True ONLY if all the above are True)

Do not omit any fields."""),
        ("human", """Policy Text:
{policy_text}

Patient Data:
{patient_data}

Please evaluate this patient against the policy criteria and return your assessment.""")
    ])
    
    # Create the chain
    chain = (
        {
            "policy_text": lambda x: x["policy_text"],
            "patient_data": lambda x: x["patient_data"]
        }
        | prompt
        | llm
    )
    
    # Invoke the chain
    evaluation = chain.invoke(state)
    
    # Return updated state with evaluation results
    return {"evaluation_results": evaluation}
