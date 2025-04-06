import json

def submit_bid(tender_id, supplier_name, bid_amount, delivery_time):
    bid = {
        "supplier_name": supplier_name,
        "bid_amount": bid_amount,
        "delivery_time": delivery_time
    }

    # Load tenders and find the one to bid on
    with open('data/tenders.json', 'r') as f:
        tenders = json.load(f)

    for tender in tenders:
        if tender["id"] == tender_id:
            # Check if a bid from the same supplier already exists
            existing_bid = next((b for b in tender["bids"] if b["supplier_name"] == supplier_name), None)
            if existing_bid:
                st.warning(f"Supplier {supplier_name} has already submitted a bid for this tender.")
                return existing_bid  # Return existing bid if supplier already submitted
            else:
                tender["bids"].append(bid)
                break

    # Save the updated tenders back to file
    with open('data/tenders.json', 'w') as f:
        json.dump(tenders, f, indent=4)

    return bid
