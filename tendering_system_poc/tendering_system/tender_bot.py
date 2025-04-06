import uuid
import json

def issue_tender(title, description, budget, deadline):
    tender_id = str(uuid.uuid4())  # Generate a unique tender ID
    tender = {
        "id": tender_id,
        "title": title,
        "description": description,
        "budget": budget,
        "deadline": deadline,
        "bids": []  # List to store bids for the tender
    }

    # Load existing tenders from file
    try:
        with open('data/tenders.json', 'r') as f:
            tenders = json.load(f)
    except FileNotFoundError:
        tenders = []

    # Check if the tender with the same ID already exists
    if any(existing_tender["id"] == tender_id for existing_tender in tenders):
        raise ValueError("Tender with the same ID already exists.")

    # Add the new tender to the list
    tenders.append(tender)

    # Save back to file
    with open('data/tenders.json', 'w') as f:
        json.dump(tenders, f, indent=4)

    return tender
