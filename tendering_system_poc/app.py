import streamlit as st
import json
from tendering_system.tender_bot import issue_tender
from tendering_system.supplier_bot import submit_bid
from tendering_system.bid_evaluation import evaluate_bids
from tendering_system.contract_bot import finalize_contract

# Helper function to load data
def load_data():
    try:
        with open('data/tenders.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Helper function to save data
def save_data(tenders):
    with open('data/tenders.json', 'w') as f:
        json.dump(tenders, f, indent=4)

# Set up the app title and logo
st.title("Bid Sense")

# Display the logo
st.image("image.jpg", width=200)

# Dashboard Introduction
st.markdown("""
    ## Welcome to **Bid Sense** – The Smart Way to Handle Bids and Tenders
    **Bid Sense** helps you streamline the tendering process. You can issue tenders, submit bids, evaluate bids based on cost and time, and finalize contracts in just a few clicks.

    ### Features:
    - **Issue Tender**: Create new tenders for your projects.
    - **Submit Bid**: Suppliers can submit their bids to your tenders.
    - **Evaluate Bids**: Compare and evaluate bids based on cost and delivery time.
    - **Finalize Contract**: Select the winning supplier and finalize the contract.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose an action", ("Dashboard", "Issue Tender", "Submit Bid", "Evaluate Bids", "Finalize Contract"))

if page == "Dashboard":
    st.header("Welcome to the Tendering Dashboard")
    st.write("You can issue tenders, submit bids, evaluate bids, and finalize contracts using the sidebar options.")

elif page == "Issue Tender":
    st.header("Issue a New Tender")
    
    # Create columns for better UI
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Tender Title")
        budget = st.number_input("Budget", min_value=0, step=1000)
    
    with col2:
        description = st.text_area("Tender Description")
        deadline = st.date_input("Deadline")
    
    if st.button("Issue Tender"):
        try:
            tender = issue_tender(title, description, budget, str(deadline))
            tenders = load_data()
            tenders.append(tender)
            save_data(tenders)
            st.success("Tender issued successfully! [Simulated Email Sent]")
        except ValueError as e:
            st.error(f"Error: {e}")

elif page == "Submit Bid":
    st.header("Submit a Bid")
    tenders = load_data()
    tender_options = [tender["title"] for tender in tenders]
    
    tender_title = st.selectbox("Select Tender", tender_options)
    tender_id = next(tender["id"] for tender in tenders if tender["title"] == tender_title)
    
    # Create columns for better UI
    col1, col2 = st.columns(2)
    
    with col1:
        supplier_name = st.text_input("Your Supplier Name")
        bid_amount = st.number_input("Bid Amount", min_value=0, step=100)
    
    with col2:
        delivery_time = st.number_input("Delivery Time (days)", min_value=1, step=1)

    if st.button("Submit Bid"):
        bid = submit_bid(tender_id, supplier_name, bid_amount, delivery_time)
        tenders = load_data()
        for tender in tenders:
            if tender["id"] == tender_id:
                tender["bids"].append(bid)
        save_data(tenders)
        st.success("Bid submitted successfully! [Simulated Email Sent]")

elif page == "Evaluate Bids":
    st.header("Evaluate Bids for a Tender")
    tenders = load_data()
    tender_options = [tender["title"] for tender in tenders]
    
    tender_title = st.selectbox("Select Tender", tender_options)
    tender_id = next(tender["id"] for tender in tenders if tender["title"] == tender_title)

    if st.button("Evaluate Bids"):
        best_bid_by_cost, best_bid_by_time = evaluate_bids(tender_id)
        
        if best_bid_by_cost and best_bid_by_time:
            st.subheader("Best Bid by Cost")
            st.text_area("Bid Details (Best by Cost)", 
                f"Supplier: {best_bid_by_cost['supplier_name']}\n"
                f"Bid Amount: {best_bid_by_cost['bid_amount']}\n"
                f"Delivery Time: {best_bid_by_cost['delivery_time']} days",
                height=200)

            st.subheader("Best Bid by Delivery Time")
            st.text_area("Bid Details (Best by Delivery Time)", 
                f"Supplier: {best_bid_by_time['supplier_name']}\n"
                f"Bid Amount: {best_bid_by_time['bid_amount']}\n"
                f"Delivery Time: {best_bid_by_time['delivery_time']} days",
                height=200)
            
            st.success("Bids evaluated successfully! [Simulated Email Sent]")
        else:
            st.warning("No bids submitted for this tender yet.")

elif page == "Finalize Contract":
    st.header("Finalize Contract with a Supplier")
    tenders = load_data()
    tender_options = [tender["title"] for tender in tenders]
    
    tender_title = st.selectbox("Select Tender", tender_options)
    tender_id = next(tender["id"] for tender in tenders if tender["title"] == tender_title)
    supplier_name = st.text_input("Supplier Name for Contract")

    if st.button("Finalize Contract"):
        tender, selected_bid = finalize_contract(tender_id, supplier_name)
        if tender and selected_bid:
            st.success(f"Contract finalized with {supplier_name} for tender: {tender_title}.")
        else:
            st.warning("The supplier was not found in the bids.")
