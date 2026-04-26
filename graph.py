from langgraph.graph import StateGraph, START, END
from state import State
from extractor import evaluate_clinical_criteria


def draft_approval(state: State) -> dict:
    """Stub node for drafting an approval letter."""
    return {"next_step": "Approved"}


def draft_denial(state: State) -> dict:
    """Stub node for drafting a denial letter."""
    return {"next_step": "Denied"}


def generate_rfi(state: State) -> dict:
    """Stub node for generating a request for information."""
    return {"next_step": "RFI Generated"}


def route_review(state: State) -> str:
    """
    Conditional routing function based on evaluation results.
    
    Args:
        state: LangGraph State containing evaluation_results
        
    Returns:
        String indicating which node to route to: "approve", "deny", or "rfi"
    """
    evaluation = state["evaluation_results"]
    
    if evaluation.meets_all_criteria:
        return "approve"
    elif not evaluation.matches_diagnosis or not evaluation.dementia_is_mild:
        return "deny"
    else:
        return "rfi"


# Build the LangGraph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("evaluate_clinical_criteria", evaluate_clinical_criteria)
workflow.add_node("draft_approval", draft_approval)
workflow.add_node("draft_denial", draft_denial)
workflow.add_node("generate_rfi", generate_rfi)

# Set entry point
workflow.add_edge(START, "evaluate_clinical_criteria")

# Add conditional edges from evaluation node
workflow.add_conditional_edges(
    "evaluate_clinical_criteria",
    route_review,
    {
        "approve": "draft_approval",
        "deny": "draft_denial",
        "rfi": "generate_rfi"
    }
)

# Add edges from stub nodes to END
workflow.add_edge("draft_approval", END)
workflow.add_edge("draft_denial", END)
workflow.add_edge("generate_rfi", END)

# Compile the graph
app = workflow.compile()
