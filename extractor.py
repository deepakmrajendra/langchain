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

Your task is to evaluate patient data strictly against the provided Policy Text and determine if the patient meets the requirements for the specified Request Type.

CRITICAL INSTRUCTIONS:
1. Do NOT use prior knowledge.
2. Look at the provided 'Request Type'. Locate the specific section in the Policy Text that applies to this Request Type (e.g., Initial vs. Continued Therapy).
3. Read that specific section to identify the clinical criteria required for approval.
4. Evaluate the provided Patient Data against those extracted criteria.
5. For each criterion, set:
   - met: True ONLY if the patient data explicitly proves the criterion is met. False if explicitly proven otherwise.
   - missing_data: True if the patient data is silent, null, or lacks the specific information needed to prove the criterion. NEVER assume a criterion is met simply because the patient data does not mention a required condition.
6. You MUST map your findings exactly to the provided JSON schema.
7. Output ONLY valid JSON. Do not include markdown formatting or conversational text."""),
        ("human", """Request Type: {request_type}

Policy Text:
{policy_text}

Patient Data:
{patient_data}

Please evaluate this patient against the policy criteria and return your structured assessment.""")
    ])
    
    # Create the chain
    chain = (
        {
            "policy_text": lambda x: x["policy_text"],
            "patient_data": lambda x: x["patient_data"],
            "request_type": lambda x: x.get("request_type", "Initial Approval")
        }
        | prompt
        | llm
    )
    
    # Invoke the chain
    evaluation = chain.invoke(state)
    
    # Return updated state with evaluation results
    return {"evaluation_results": evaluation}
