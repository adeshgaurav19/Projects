import json

def evaluate_bids(tender_id):
    with open('data/tenders.json', 'r') as f:
        tenders = json.load(f)

    best_bid_by_cost = None
    best_bid_by_time = None

    for tender in tenders:
        if tender["id"] == tender_id:
            bids = tender["bids"]
            if bids:
                # Sort bids by cost (lowest bid amount) and by delivery time (fastest)
                sorted_by_cost = sorted(bids, key=lambda x: x["bid_amount"])
                sorted_by_time = sorted(bids, key=lambda x: x["delivery_time"])

                best_bid_by_cost = sorted_by_cost[0]  # Best bid by cost
                best_bid_by_time = sorted_by_time[0]  # Best bid by delivery time

            break

    return best_bid_by_cost, best_bid_by_time
