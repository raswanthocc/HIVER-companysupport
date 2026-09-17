import json
import os
import re

INTENTS_20 = [
    "order_delivery_status", "delayed_delivery", "cancellation", "refund_request",
    "refund_status", "payment_billing_issue", "unauthorized_fraudulent_transaction",
    "account_access_login", "account_problem", "password_reset", "product_service_issue",
    "product_information", "pricing_charges", "subscription_plan", "technical_issue",
    "verification_identity", "complaint_poor_experience", "general_information",
    "request_for_human_support", "other_unclear"
]

def heuristic_classify(text):
    t = text.lower()
    
    # 1. Fraud / Security compromise
    if any(w in t for w in ["unauthorized", "fraud", "hacked", "stolen", "scam", "suspicious", "compromised"]):
        return "unauthorized_fraudulent_transaction"
        
    # 2. Refund inquiries
    if any(w in t for w in ["refund status", "where is my refund", "refund update", "status of my refund"]):
        return "refund_status"
    if any(w in t for w in ["refund", "money back", "reimburse", "reimbursement", "return my money", "want my money"]):
        return "refund_request"
        
    # 3. Passwords & Reset
    if any(w in t for w in ["password", "passcode", "reset", "forgot", "senha", "iforgot", "change password", "recover password"]):
        return "password_reset"
        
    # 4. Identity & 2FA
    if any(w in t for w in ["2fa", "two-factor", "two factor", "security code", "verification code", "trusted number", "authenticator", "identity check"]):
        return "verification_identity"
        
    # 5. Account Access & Sign In
    if any(w in t for w in ["login", "log in", "sign in", "can't access", "locked out", "activation lock", "icloud lock", "touch id", "face id", "Account ID locked", "security questions"]):
        return "account_access_login"
        
    # 6. General Account
    if any(w in t for w in ["Account ID", "icloud account", "Account ID", "account profile", "my account"]):
        return "account_problem"
        
    # 7. Subscriptions
    if any(w in t for w in ["subscription", "plan", "renewal", "renew", "cancel subscription", "monthly charge", "Company music sub", "itunes match", "storage plan"]):
        return "subscription_plan"
        
    # 8. Billing & Payments
    if any(w in t for w in ["bill", "billing", "payment", "card", "charged", "overcharged", "double charge", "invoice", "receipt", "deducted", "itunes purchase", "in-app purchase", "bought", "credit card", "bank", "dispute charge"]):
        return "payment_billing_issue"
        
    # 9. Human Support Request
    if any(w in t for w in ["human", "agent", "representative", "real person", "speak to someone", "talk to a person", "customer service rep", "live agent", "transfer me"]):
        return "request_for_human_support"
        
    # 10. Cancellation
    if any(w in t for w in ["cancel my order", "order cancellation", "cancel order", "cancellation"]):
        return "cancellation"
        
    # 11. Delivery Delays
    if any(w in t for w in ["delay", "late delivery", "hasn't arrived", "not arrived", "taking so long", "delivery late"]):
        return "delayed_delivery"
        
    # 12. Order / Delivery Tracking
    if any(w in t for w in ["order", "delivery", "track", "tracking", "shipping", "shipped", "package", "parcel", "dispatch", "courier", "ups", "fedex"]):
        return "order_delivery_status"
        
    # 13. Pricing / Cost
    if any(w in t for w in ["how much does", "price", "cost", "repair fee", "pricing quote", "expensive"]):
        return "pricing_charges"
        
    # 14. Product Specs / Information
    if any(w in t for w in ["specs", "specification", "features", "dimension", "release date", "colors", "screen size", "storage capacity"]):
        return "product_information"
        
    # 15. Complaints / Frustration
    if any(w in t for w in ["terrible", "worst", "hate Company", "garbage", "trash", "useless", "disappointed", "complaint", "sucks", "ridiculous", "frustrated", "annoying", "angry", "ripped off", "joke of a company", "boycott", "switching to android"]):
        return "complaint_poor_experience"
        
    # 16. Hardware / Physical Service
    if any(w in t for w in ["broken", "crack", "shatter", "defect", "damage", "hardware", "swollen", "battery swelling", "shattered screen", "cracked screen", "repair", "physical damage", "earpiece", "speaker defect", "charging port", "lightning port", "water damage", "dropped my phone", "popped", "genius bar"]):
        return "product_service_issue"
        
    # 17. Technical / Software / Wi-Fi / Battery Drain
    if any(w in t for w in ["bug", "glitch", "crash", "freeze", "froze", "wifi", "wi-fi", "bluetooth", "update", "OS", "software", "connection", "error", "restore", "storage", "app", "photos", "sync", "backup", "touch", "keyboard", "sound", "volume", "silent", "alarm", "disconnect", "connect", "slow", "lag", "not working", "fails", "failed", "acting up", "battery drain", "dies quick", "battery dying"]):
        return "technical_issue"
        
    # 18. General Info / How-To
    if any(w in t for w in ["how to", "how do i", "information", "what is", "guide", "advice", "help with", "where can i"]):
        return "general_information"
        
    return "other_unclear"

def process_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    count = 0
    for item in data:
        text = item.get("customer_text", "")
        new_intent = heuristic_classify(text)
        item["intent"] = new_intent
        count += 1
        
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Successfully relabeled {count} items in {filepath}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    for filename in ["train_data.json", "val_data.json", "golden_eval_set.json"]:
        p = os.path.join(data_dir, filename)
        process_file(p)
