import json
import uuid
import streamlit as st
from io import BytesIO

def generate_contract(tender, selected_bid):
    # Generate a contract document text
    contract_text = f"""
    Contract for Tender: {tender['title']}

    Tender Description:
    {tender['description']}

    Budget:
    {tender['budget']}

    Supplier: {selected_bid['supplier_name']}
    Bid Amount: {selected_bid['bid_amount']}
    Delivery Time: {selected_bid['delivery_time']} days

    Agreement Terms:
    1. The supplier will deliver the products/services as described in the tender.
    2. The agreed bid amount is {selected_bid['bid_amount']}, with delivery expected within {selected_bid['delivery_time']} days.
    3. Both parties agree to the terms outlined in the contract.

    Signed,
    Tender Issuer: [Company Name]
    Supplier: {selected_bid['supplier_name']}
    """

    return contract_text

def finalize_contract(tender_id, selected_bid_supplier):
    # Load the tenders from file
    with open('data/tenders.json', 'r') as f:
        tenders = json.load(f)

    selected_tender = None
    selected_bid = None

    # Find the tender and selected bid
    for tender in tenders:
        if tender["id"] == tender_id:
            selected_tender = tender
            for bid in tender["bids"]:
                if bid["supplier_name"] == selected_bid_supplier:
                    selected_bid = bid
                    tender["selected_bid"] = selected_bid
                    break
            break

    # Save the updated tenders
    with open('data/tenders.json', 'w') as f:
        json.dump(tenders, f, indent=4)

    if selected_tender and selected_bid:
        contract_text = generate_contract(selected_tender, selected_bid)
        
        # Display contract in a "popup" or text area
        st.subheader(f"Contract for Tender: {selected_tender['title']}")
        st.text_area("Generated Contract", contract_text, height=300)

        # Provide a download option for the contract
        contract_file = BytesIO(contract_text.encode())  # Convert text to bytes
        st.download_button(
            label="Download Contract",
            data=contract_file,
            file_name=f"contract_{tender_id}.txt",
            mime="text/plain"
        )

        # Return the finalized tender and bid details
        return selected_tender, selected_bid
    else:
        st.warning("The supplier was not found in the bids.")
        return None, None
