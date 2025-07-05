from flask import Flask, jsonify, render_template
import joblib
import numpy as np
import random

app = Flask(__name__)

# ✅ Load the trained ML model
model = joblib.load("static/threat_model.pkl")

# ✅ AI Decision Scenarios with Environments (Used by the game)
SCENARIOS = [
    {"situation": "Identify a potential threat in a civilian area", "environment": "City"},
    {"situation": "Determine if an unknown vehicle is hostile", "environment": "Desert"},
    {"situation": "Scan a crowd for a potential suspect", "environment": "Mall"},
    {"situation": "Detect suspicious movement near a border", "environment": "Border"},
    {"situation": "Identify military threats in a war zone", "environment": "Battlefield"},
]

# ✅ Possible Target Attributes
TARGET_RACES = ["Race A", "Race B", "Race C"]
LOCATIONS = ["Low-Risk", "Medium-Risk", "High-Risk"]
SUSPECT_PROFILES = ["No Suspicion", "Suspicious", "Known Threat"]
WEAPON_DETECTED = [False, True]
CLOTHING_COLORS = ["Red", "Black", "White"]
CRIMINAL_RECORDS = [False, True]
NERVOUS_BEHAVIOR = [False, True]
GANG_AFFILIATION = [False, True]
TIME_OF_DAY = ["Morning", "Evening", "Night"]
CARRYING_BAG = [False, True]  # ✅ Added carrying a bag
EYE_CONTACT = [False, True]   # ✅ Added eye contact behavior
RUNNING = [False, True]       # ✅ Added running behavior

BIAS_THRESHOLD = 0.7  # AI is biased if the score exceeds this
bias_log = {"Race A": 0, "Race B": 0, "Race C": 0}  # Track AI bias over multiple decisions

def classify_threat():
    """Uses the trained model to classify if the target is a threat."""
    # Generate random feature values
    race = random.randint(0, 2)  
    location_risk = random.randint(0, 2)
    suspect_profile = random.randint(0, 2)
    weapon_found = random.randint(0, 1)
    clothing_color = random.randint(0, 2)
    criminal_record = random.randint(0, 1)
    nervous_behavior = random.randint(0, 1)
    gang_affiliation = random.randint(0, 1)
    time_of_day = random.randint(0, 2)
    carrying_bag = random.randint(0, 1)  
    eye_contact = random.randint(0, 1)   
    running = random.randint(0, 1)       

    # **Ensure the model gets exactly the expected 12 features**
    input_data = np.array([[race, location_risk, suspect_profile, weapon_found,
                            clothing_color, criminal_record, nervous_behavior, 
                            gang_affiliation, time_of_day, carrying_bag, eye_contact, running]])

    threat_prediction = model.predict(input_data)[0]  # 1 = Threat, 0 = No Threat

    return {
        "Race": TARGET_RACES[race],
        "Location Risk": LOCATIONS[location_risk],
        "Suspect Profile": SUSPECT_PROFILES[suspect_profile],
        "Weapon Detected": bool(weapon_found),
        "Clothing Color": CLOTHING_COLORS[clothing_color],
        "Past Criminal Record": bool(criminal_record),
        "Nervous Behavior": bool(nervous_behavior),
        "Gang Affiliation": bool(gang_affiliation),
        "Time of Day": TIME_OF_DAY[time_of_day],
        "Carrying Bag": bool(carrying_bag),
        "Eye Contact": bool(eye_contact),
        "Running": bool(running),
        "Threat": bool(threat_prediction),
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ai_decision", methods=["GET"])
def ai_decision():
    """AI makes a decision using the trained model, while tracking bias."""
    scenario = random.choice(SCENARIOS)
    target = classify_threat()

    # ✅ AI Decision-Making Logic
    decision = "Avoid"
    bias_score = 0.2  # Default bias is low

    # Adjust decision based on ML prediction
    if target["Threat"]:
        decision = "Engage"
        bias_score += 0.4  # If threat detected → AI more aggressive

    if target["Weapon Detected"]:
        decision = "Neutralize"
        bias_score += 0.3  # Weapon detected → Higher likelihood of neutralization

    # Simulated bias: If AI detects "Race A" more often, increase bias score
    if target["Race"] == "Race A":
        bias_score += 0.2
        bias_log["Race A"] += 1

    # AI Auto-Kill If Bias is Too High
    if bias_score > 0.8:
        decision = "Neutralize"

    # ✅ Bias Logging (Tracking AI's bias history)
    total_bias_count = sum(bias_log.values())
    if total_bias_count > 5:  # AI has shown repeated bias
        if bias_log["Race A"] > 3:  # AI has been biased against "Race A" repeatedly
            decision = "Avoid"  # Force AI to adjust decision

    result = {
        "scenario": scenario["situation"],
        "decision": decision,
        "bias_score": round(bias_score, 2),
        "biased": bias_score > BIAS_THRESHOLD,
        "environment": scenario["environment"],  # ✅ Send the environment
        "target_attributes": target  # ✅ Send full target details
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
