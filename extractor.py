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

Your task is to evaluate patient data strictly against the provided Policy Text and determine if the patient meets the requirements for initial approval.

CRITICAL INSTRUCTIONS:
1. Do NOT use prior knowledge.
2. You must read the provided Policy Text to identify the specific clinical criteria required for approval.
3. Evaluate the provided Patient Data against those extracted criteria.
4. You MUST map your findings exactly to the provided JSON schema boolean fields (e.g., matches_diagnosis, dementia_is_mild).
5. Output ONLY valid JSON. Do not include markdown formatting or conversational text."""),
        ("human", """Policy Text:
{policy_text}

Patient Data:
{patient_data}

Please evaluate this patient against the policy criteria and return your structured assessment.""")
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
