from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
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
    evaluation = state["evaluation_results"]

    if evaluation.meets_all_criteria:
        return "approve"

    # Check if any criterion has missing data (RFI) vs explicitly unmet (Denial)
    has_missing_data = any(criterion.missing_data for criterion in evaluation.criteria_evaluations)
    has_explicitly_unmet = any(not criterion.met and not criterion.missing_data for criterion in evaluation.criteria_evaluations)

    if has_missing_data:
        return "rfi"
    elif has_explicitly_unmet:
        return "deny"
    else:
        # Fallback - if it doesn't meet all but no explicit flags, request more info
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

# Add the Checkpointer
memory = MemorySaver()

# Compile the graph with the checkpointer and interrupt rule
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["evaluate_clinical_criteria"]
)
