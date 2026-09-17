"""
Universal 20-Intent Taxonomy for Customer Service and Support.
Relabeled and expanded from foundational interactions to cover full-lifecycle customer queries.
"""

from typing import Dict, Any, List

INTENT_LABELS = [
    "order_delivery_status",
    "delayed_delivery",
    "cancellation",
    "refund_request",
    "refund_status",
    "payment_billing_issue",
    "unauthorized_fraudulent_transaction",
    "account_access_login",
    "account_problem",
    "password_reset",
    "product_service_issue",
    "product_information",
    "pricing_charges",
    "subscription_plan",
    "technical_issue",
    "verification_identity",
    "complaint_poor_experience",
    "general_information",
    "request_for_human_support",
    "other_unclear"
]

INTENT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "order_delivery_status": {
        "name": "Order & Delivery Status",
        "description": "Tracking shipments, parcel locations, carrier progress, and expected delivery dates.",
        "example_utterance": "Where is my shipment? Can you track my order status?",
        "default_routing": "auto_handle (order lookup)"
    },
    "delayed_delivery": {
        "name": "Delayed Delivery",
        "description": "Inquiries regarding packages arriving late or past estimated delivery date.",
        "example_utterance": "My delivery has been delayed for three days now.",
        "default_routing": "auto_handle (carrier escalation)"
    },
    "cancellation": {
        "name": "Order / Service Cancellation",
        "description": "Requests to cancel orders, items, or services prior to shipment/fulfillment.",
        "example_utterance": "I want to cancel my recent purchase immediately.",
        "default_routing": "auto_handle (cancellation processing)"
    },
    "refund_request": {
        "name": "Refund Request",
        "description": "Customer asking for money back, reimbursement, or returns.",
        "example_utterance": "I want a refund for the accidental purchase I made.",
        "default_routing": "escalate (billing refund queue)"
    },
    "refund_status": {
        "name": "Refund Status",
        "description": "Checking progress or status of a previously processed refund.",
        "example_utterance": "When will my refund show up in my bank account?",
        "default_routing": "auto_handle (refund lookup)"
    },
    "payment_billing_issue": {
        "name": "Payment & Billing Issues",
        "description": "Overcharges, failed payments, declined cards, duplicate charges.",
        "example_utterance": "I was charged twice on my credit card this morning.",
        "default_routing": "escalate (financial dispute)"
    },
    "unauthorized_fraudulent_transaction": {
        "name": "Unauthorized / Fraudulent Transaction",
        "description": "Compromised accounts, stolen card usage, suspicious charges.",
        "example_utterance": "Someone used my card to buy items without permission.",
        "default_routing": "escalate (fraud & security)"
    },
    "account_access_login": {
        "name": "Account Access & Login",
        "description": "Trouble signing in, locked accounts, credentials not recognized.",
        "example_utterance": "I am locked out of my account and cannot log in.",
        "default_routing": "escalate (identity verification)"
    },
    "account_problem": {
        "name": "Account Problem",
        "description": "Profile settings, changing contact info, linking accounts.",
        "example_utterance": "Need to update my registered email and phone number.",
        "default_routing": "auto_handle (profile guide)"
    },
    "password_reset": {
        "name": "Password Reset",
        "description": "Requests to reset password, forgotten PIN, or passcode recovery.",
        "example_utterance": "Forgot my passcode and need to reset my password.",
        "default_routing": "auto_handle (automated reset link)"
    },
    "product_service_issue": {
        "name": "Product & Hardware Service Issue",
        "description": "Damaged items, cracked screens, defective hardware, repair booking.",
        "example_utterance": "Screen is shattered and speaker stopped functioning.",
        "default_routing": "escalate (Genius Bar / repair booking)"
    },
    "product_information": {
        "name": "Product Information",
        "description": "Specs, compatibility, features, and release dates.",
        "example_utterance": "Does this model support wireless fast charging?",
        "default_routing": "auto_handle (spec sheet lookup)"
    },
    "pricing_charges": {
        "name": "Pricing & Fee Inquiries",
        "description": "Questions regarding pricing structures, repair fees, or quotes.",
        "example_utterance": "How much does a battery replacement cost out of warranty?",
        "default_routing": "auto_handle (pricing table)"
    },
    "subscription_plan": {
        "name": "Subscription & Plan Management",
        "description": "Upgrading, downgrading, or managing recurring memberships.",
        "example_utterance": "How do I cancel my monthly iCloud storage plan?",
        "default_routing": "auto_handle (subscription portal)"
    },
    "technical_issue": {
        "name": "Technical & Software Issue",
        "description": "Bugs, crashes, frozen screen, Wi-Fi drops, Bluetooth pairing.",
        "example_utterance": "Phone keeps freezing and Wi-Fi drops constantly after update.",
        "default_routing": "auto_handle (troubleshooting workflow)"
    },
    "verification_identity": {
        "name": "Identity Verification & 2FA",
        "description": "Two-factor authentication codes, security keys, document checks.",
        "example_utterance": "I'm not receiving my two-factor authentication verification code.",
        "default_routing": "escalate (security specialist)"
    },
    "complaint_poor_experience": {
        "name": "Complaint & Poor Experience",
        "description": "Customer venting frustration, dissatisfaction, or service complaints.",
        "example_utterance": "This is the worst customer experience I have ever had.",
        "default_routing": "auto_handle (empathy response + supervisor flag)"
    },
    "general_information": {
        "name": "General Information",
        "description": "Store hours, policies, guidelines, and non-technical how-tos.",
        "example_utterance": "What are your support center operating hours?",
        "default_routing": "auto_handle (knowledge base FAQ)"
    },
    "request_for_human_support": {
        "name": "Request for Human Support",
        "description": "Explicit demand to speak to a real representative or agent.",
        "example_utterance": "Transfer me to a live human representative right now.",
        "default_routing": "escalate (human handoff)"
    },
    "other_unclear": {
        "name": "Other / Unclear",
        "description": "Ambiguous, out-of-domain, or incomplete queries requiring clarification.",
        "example_utterance": "Hey",
        "default_routing": "clarify (diagnostic follow-up)"
    }
}

def silver_label_interaction(customer_text: str, agent_text: str = "") -> str:
    """Heuristic classifier for assigning silver intent from customer inquiry."""
    t = customer_text.lower()
    
    if any(w in t for w in ["unauthorized", "fraud", "hacked", "stolen", "scam", "suspicious", "compromised"]):
        return "unauthorized_fraudulent_transaction"
    if any(w in t for w in ["refund status", "where is my refund", "refund update", "status of my refund"]):
        return "refund_status"
    if any(w in t for w in ["refund", "money back", "reimburse", "reimbursement", "return my money", "want my money"]):
        return "refund_request"
    if any(w in t for w in ["password", "passcode", "reset", "forgot", "senha", "iforgot", "change password", "recover password"]):
        return "password_reset"
    if any(w in t for w in ["2fa", "two-factor", "two factor", "security code", "verification code", "trusted number", "authenticator"]):
        return "verification_identity"
    if any(w in t for w in ["login", "log in", "sign in", "can't access", "locked out", "activation lock", "cloud lock", "touch id", "face id", "account id locked"]):
        return "account_access_login"
    if any(w in t for w in ["account id", "cloud account", "account profile", "my account"]):
        return "account_problem"
    if any(w in t for w in ["subscription", "plan", "renewal", "renew", "cancel subscription", "monthly charge", "music sub"]):
        return "subscription_plan"
    if any(w in t for w in ["bill", "billing", "payment", "card", "charged", "overcharged", "double charge", "invoice", "receipt", "deducted", "itunes purchase", "in-app purchase", "credit card"]):
        return "payment_billing_issue"
    if any(w in t for w in ["human", "agent", "representative", "real person", "speak to someone", "talk to a person", "customer service rep", "live agent"]):
        return "request_for_human_support"
    if any(w in t for w in ["cancel my order", "order cancellation", "cancel order", "cancellation"]):
        return "cancellation"
    if any(w in t for w in ["delay", "late delivery", "hasn't arrived", "not arrived", "taking so long"]):
        return "delayed_delivery"
    if any(w in t for w in ["order", "delivery", "track", "tracking", "shipping", "shipped", "package", "parcel", "dispatch"]):
        return "order_delivery_status"
    if any(w in t for w in ["how much does", "price", "cost", "repair fee", "pricing quote", "expensive"]):
        return "pricing_charges"
    if any(w in t for w in ["specs", "specification", "features", "dimension", "release date", "colors", "screen size"]):
        return "product_information"
    if any(w in t for w in ["terrible", "worst", "hate company", "garbage", "trash", "useless", "disappointed", "complaint", "sucks", "ridiculous", "frustrated", "annoying", "angry", "ripped off"]):
        return "complaint_poor_experience"
    if any(w in t for w in ["broken", "crack", "shatter", "defect", "damage", "hardware", "swollen", "battery swelling", "shattered screen", "cracked screen", "repair", "physical damage", "earpiece", "speaker defect", "charging port", "lightning port", "water damage", "dropped my phone", "popped", "genius bar"]):
        return "product_service_issue"
    if any(w in t for w in ["bug", "glitch", "crash", "freeze", "froze", "wifi", "wi-fi", "bluetooth", "update", "OS", "software", "connection", "error", "restore", "storage", "app", "photos", "sync", "backup", "touch", "keyboard", "sound", "volume", "silent", "alarm", "disconnect", "connect", "slow", "lag", "not working", "fails", "failed", "acting up", "battery drain", "dies quick", "battery dying"]):
        return "technical_issue"
    if any(w in t for w in ["how to", "how do i", "information", "what is", "guide", "advice", "help with", "where can i"]):
        return "general_information"
        
    return "other_unclear"
