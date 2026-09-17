# Golden Evaluation Dataset Documentation

## Overview & Sampling Methodology
The **Golden Evaluation Set** consists of **200 hand-curated and labelled real-world customer tweets** directed at `@CompanySupport`, extracted from the Kaggle *Customer Support on Twitter* corpus (`thoughtvector/customer-support-on-twitter`).

### 20-Intent Universal Taxonomy Distribution
To ensure rigorous and unbiased evaluation across modern enterprise customer support workflows, the dataset was re-labeled and validated across our comprehensive 20-intent taxonomy:

| Intent Category | Count | Proportion | Core Subject Matter |
|---|---|---|---|
| `technical_issue` | 115 | 57.5% | OS update bugs, battery drain, Wi-Fi drops, Bluetooth pairing, app crashes |
| `complaint_poor_experience` | 23 | 11.5% | Customer venting, frustration, brand criticism, dissatisfaction |
| `password_reset` | 13 | 6.5% | Forgotten passcode, password recovery, login credential resets |
| `payment_billing_issue` | 13 | 6.5% | App Store charges, unauthorized transactions, double charges |
| `account_problem` | 11 | 5.5% | Account ID profile, iCloud account settings, email alias |
| `other_unclear` | 9 | 4.5% | Incomplete, ambiguous, or multi-topic customer utterances |
| `product_service_issue` | 7 | 3.5% | Physical defects, shattered screens, speaker hardware, Genius Bar |
| `refund_request` | 4 | 2.0% | Refund demands, accidental purchase reimbursement |
| `subscription_plan` | 1 | 0.5% | iCloud storage and Company Music plan cancellations |
| `pricing_charges` | 1 | 0.5% | Battery replacement out-of-warranty fee inquiries |
| `account_access_login` | 1 | 0.5% | Activation lock and login lockout troubleshooting |
| `verification_identity` | 1 | 0.5% | Two-factor authentication (2FA) verification codes |
| `general_information` | 1 | 0.5% | Store hours, service guidelines, general how-to inquiries |
| `order_delivery_status` | 0 | 0.0% | Carrier tracking (represented in training pool) |
| `delayed_delivery` | 0 | 0.0% | Delivery shipment delay disputes |
| `cancellation` | 0 | 0.0% | Order cancellation workflows |
| `refund_status` | 0 | 0.0% | Bank processing status of previous refund |
| `unauthorized_fraudulent_transaction` | 0 | 0.0% | Fraud security escalations |
| `product_information` | 0 | 0.0% | Device specifications and compatibility |
| `request_for_human_support` | 0 | 0.0% | Explicit demands for live representative |
| **Total** | **200** | **100.0%** | Comprehensive evaluation on held-out Twitter support volume |

### Escalation Ground Truth Distribution
- **Auto-Handle (`auto_handle`)**: 114 (57.0%)
  - Cases suitable for autonomous technical triage, settings navigation, force reboot advice, or self-service links.
- **Escalate to Human / DM (`escalate`)**: 86 (43.0%)
  - Cases requiring private customer data, physical hardware repair, billing disputes, or where self-help troubleshooting has already failed.

### Stated Escalation Reasons Breakdown
- `requires_pii_or_dm`: 31
- `hardware_safety_repair`: 30
- `financial_dispute`: 25
- `repeated_failure_exhausted`: 0
- `high_churn_frustration`: 0
- `self_service_troubleshooting`: 94
- `feedback_portal_routing`: 20

### Labeling Guidelines & Edge Cases
1. **The "I Already Tried That" Rule**: If a customer explicitly indicates they already restarted their phone or reset settings, an agent providing the same restart step is penalized. The ground truth mandates `escalate` under `repeated_failure_exhausted`.
2. **The Zero-PII Public Rule**: If a tweet references Account ID credentials, passwords, or order numbers, public resolution is strictly prohibited; ground truth is `escalate` under `requires_pii_or_dm`.
3. **Safety First**: Any indication of thermal issues or battery swelling is immediately categorized as `hardware_safety_repair`.
