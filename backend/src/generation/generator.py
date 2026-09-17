"""
Modern Multi-Brand RAG Generator using Gemini LLM.
Synthesizes responses grounded in retrieved historical TWCS interactions with dynamic regeneration.
"""

import os
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types
from src.models import EscalationDecision, IntentPrediction, RetrievedContext

class GroundedResponseGenerator:
    """
    Synthesizes customer support replies dynamically using Gemini,
    passing the retrieved ChromaDB context into the prompt with brand-aware styling.
    """
    def __init__(self, brand_handle: str = "Support"):
        self.brand_handle = brand_handle
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.model = "gemini-3.6-flash"

    def generate(
        self,
        customer_text: str,
        intent_pred: IntentPrediction,
        escalation: EscalationDecision,
        retrieved: RetrievedContext,
        selected_brand: str = "All Brands"
    ) -> str:
        """Generate response grounded in retrieved historical evidence using Gemini."""
        if escalation.action == "escalate":
            retrieved.grounding_status = "human_handoff_triggered"
            return self._generate_escalation_reply(customer_text, escalation, selected_brand)
        else:
            return self._generate_autohandle_reply(customer_text, intent_pred, retrieved, selected_brand)

    def _generate_escalation_reply(
        self,
        customer_text: str,
        escalation: EscalationDecision,
        selected_brand: str
    ) -> str:
        brand_name = selected_brand if selected_brand != "All Brands" else "Customer Support"
        dm_link = "https://twitter.com/messages/compose"
        
        reason = escalation.reason
        explanation = getattr(escalation, "explanation", "")
        
        reply = f"🚨 [Human Support Handoff Triggered]\n\n"
        reply += f"Thank you for contacting @{brand_name}. {explanation}\n"
        reply += f"Due to safety or account privacy protocols, a specialist is standing by to take over. Please click here to connect securely: {dm_link}"
        return reply

    def _generate_autohandle_reply(
        self,
        customer_text: str,
        intent_pred: IntentPrediction,
        retrieved: RetrievedContext,
        selected_brand: str,
        retry_count: int = 0
    ) -> str:
        
        if not self.client:
            return "Error: GEMINI_API_KEY is missing. Please configure your API key to enable dynamic AI generation."
            
        brand_context = f"@{selected_brand}" if selected_brand != "All Brands" else "Customer Support"
        
        # Build Context from ChromaDB
        context_str = ""
        if retrieved.historical_exemplars:
            context_str += f"Here are authentic past interactions resolved by {brand_context}:\n\n"
            for i, ex in enumerate(retrieved.historical_exemplars[:3]):
                context_str += f"Example {i+1} [{ex.get('brand', 'Support')}]:\n"
                context_str += f"Customer: {ex.get('customer_text', '')}\n"
                context_str += f"Agent Reply: {ex.get('agent_text', '')}\n\n"
                
        prompt = f"""You are the official expert customer support AI for {brand_context}.
Your absolute priority is to provide a highly relevant, deeply empathetic, and immediately actionable response that directly solves the customer's specific issue.

{context_str}
Instructions:
1. DIRECT RELEVANCE: Directly address the exact problem the customer described. Do not be vague or generic.
2. GROUNDING: Use the authentic past interaction examples provided above as a blueprint for tone, specific URLs, policies, or solutions. Adapt those solutions to precisely fit the customer's current query.
3. HELPFULNESS: If the issue requires troubleshooting or steps, provide clear, step-by-step guidance. Do not just brush them off.
4. BRAND TONE: Be warm, patient, and professional. Write exactly as {brand_context} would on Twitter/X.
5. NO HALLUCINATION: Never invent fake URLs, policies, or contact phone numbers. If a DM is needed, simply say "Please send us a DM".

Customer Query: {customer_text}
Official {brand_context} Reply:"""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3 + (retry_count * 0.2), # Self-reflection temperature scaling on retry
                )
            )
            text = response.text.strip()
            
            # Quality Check: If generated text is empty or too short, retry once (Dynamic Fallback Regeneration)
            if len(text) < 15 and retry_count < 2:
                return self._generate_autohandle_reply(customer_text, intent_pred, retrieved, selected_brand, retry_count=retry_count+1)
                
            retrieved.grounding_status = "agentic_rag_llm_synthesized"
            return text
        except Exception as e:
            print(f"LLM Generation Error (Gemini): {e}")
            
            # Groq Fallback Logic
            env_keys = os.environ.get("GROQ_API_KEYS") or os.environ.get("GROQ_API_KEY") or ""
            groq_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            if groq_keys:
                groq_models = ["groq/compound-mini", "llama-3.1-8b-instant", "llama3-8b-8192"]
                from groq import Groq
                import random

            import random
            
            for key in random.sample(groq_keys, len(groq_keys)):
                groq_client = Groq(api_key=key)
                for gmodel in groq_models:
                    try:
                        completion = groq_client.chat.completions.create(
                            model=gmodel,
                            messages=[
                                {"role": "system", "content": f"You are the official expert customer support AI for {brand_context}."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3 + (retry_count * 0.2),
                            max_tokens=256
                        )
                        text = completion.choices[0].message.content.strip()
                        retrieved.grounding_status = "agentic_rag_llm_synthesized (Groq Fallback)"
                        return text
                    except Exception:
                        continue

            # Self-reflective regeneration fallback instead of hardcoded string if all fail

            if retry_count < 2:
                return self._generate_autohandle_reply(customer_text, intent_pred, retrieved, selected_brand, retry_count=retry_count+1)
            retrieved.grounding_status = "llm_generation_error"
            return f"We are experiencing high traffic. Please reach out to {brand_context} directly or re-submit your inquiry."
