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
            "evaluation_results": None,
            "next_step": None
        }
        
        result = app.invoke(state)
        
        # Print results
        print(f"Patient ID: {patient['patient_id']}")
        print(f"Next Step: {result['next_step']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
