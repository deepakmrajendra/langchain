from dotenv import load_dotenv
load_dotenv()

import json
from langchain_community.document_loaders import PyPDFLoader
from graph import app
from state import State


def main():
    # Load the policy text from PDF
    loader = PyPDFLoader("input/pa/TX.PHAR.115.pdf")
    pages = loader.load()
    policy_text = "\n".join([page.page_content for page in pages])
    
    # Load patient payloads
    with open("input/patient_payload.json", "r") as f:
        patients = json.load(f)
    
    # Process each patient through the graph
    for patient in patients:
        # Initialize state
        state: State = {
            "patient_data": patient,
            "policy_text": policy_text,
            "request_type": None,
            "evaluation_results": None,
            "next_step": None
        }
        
        # 1. Define a thread configuration. The checkpointer requires a thread_id 
        # to separate the state of different executions in memory.
        config = {"configurable": {"thread_id": patient["patient_id"]}}
   
        print(f"\nProcessing Patient ID: {patient['patient_id']}")
   
        # 2. Start execution. It will immediately pause due to 'interrupt_before'.
        app.invoke(state, config=config)
   
        # 3. Capture standard user input for HITL.
        user_choice = input(f"Is this request for Initial (I) or Continued (C) therapy? [I/C]: ").strip().upper()
        req_type = "Continued Therapy" if user_choice == 'C' else "Initial Approval"
   
        # 4. Inject the user input directly into the interrupted state graph.
        app.update_state(config, {"request_type": req_type})
   
        # 5. Resume execution by invoking with None. The checkpointer loads 
        # the saved state and proceeds past the interrupt.
        result = app.invoke(None, config=config)
   
        # Print results
        print(f"Next Step: {result['next_step']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
