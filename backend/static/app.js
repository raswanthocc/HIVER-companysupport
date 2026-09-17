document.addEventListener('DOMContentLoaded', () => {
    const brandSelect = document.getElementById('brandSelect');
    const currentBrandName = document.getElementById('currentBrandName');
    const presetsContainer = document.getElementById('presetsContainer');
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');
    const typingIndicator = document.getElementById('typingIndicator');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const toggleEvidenceBtn = document.getElementById('toggleEvidenceBtn');
    const evidenceDrawer = document.getElementById('evidenceDrawer');
    const closeDrawerBtn = document.getElementById('closeDrawerBtn');
    const evidenceContent = document.getElementById('evidenceContent');

    let currentEvidence = [];

    // Brand Presets Database
    const brandPresets = {
        "AmazonHelp": [
            "My package was marked as delivered today but I never received it!",
            "Order cancellation request for item shipped by mistake",
            "Received a damaged package with broken items inside"
        ],
        "AppleSupport": [
            "My iPhone battery drains in 2 hours since updating to latest iOS",
            "Screen cracked and battery feels hot and swollen",
            "Locked out of my Apple ID account and 2FA is not sending codes"
        ],
        "Uber_Support": [
            "Driver took a much longer route than expected and charged extra",
            "Left my wallet in the back seat of the Uber car last night"
        ],
        "SpotifyCares": [
            "My Premium subscription charged me twice this month",
            "Music keeps pausing automatically every time screen turns off"
        ],
        "BofA_Help": [
            "I lost my debit card and need to freeze my account immediately!",
            "Unauthorized transaction appearing on my bank statement"
        ],
        "All Brands": [
            "My order was supposed to arrive yesterday but tracking hasn't updated!",
            "I lost my card and suspect someone is trying to make unauthorized charges!",
            "Battery gets hot and screen popped out, looks swollen!",
            "My subscription renewed automatically, can I get a refund?"
        ]
    };

    // Load Available Brands from API
    async function fetchBrands() {
        try {
            const res = await fetch('/api/brands');
            const data = await res.json();
            if (data.brands && data.brands.length > 0) {
                brandSelect.innerHTML = '';
                data.brands.forEach(brand => {
                    const opt = document.createElement('option');
                    opt.value = brand;
                    opt.textContent = brand === 'All Brands' ? 'All Brands (Universal)' : `@${brand}`;
                    brandSelect.appendChild(opt);
                });
            }
        } catch (err) {
            console.error('Error fetching brands:', err);
        }
    }

    // Render Query Presets
    function renderPresets(brand) {
        presetsContainer.innerHTML = '';
        const list = brandPresets[brand] || brandPresets["All Brands"];
        list.forEach(text => {
            const chip = document.createElement('div');
            chip.className = 'preset-chip';
            chip.textContent = text;
            chip.addEventListener('click', () => {
                userInput.value = text;
                userInput.focus();
            });
            presetsContainer.appendChild(chip);
        });
    }

    // Update UI on Brand Switch
    brandSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        currentBrandName.textContent = val === 'All Brands' ? '@All Brands' : `@${val}`;
        renderPresets(val);
    });

    // Form Submit
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        const selectedBrand = brandSelect.value;
        
        // Remove welcome card if present
        const welcomeCard = document.querySelector('.welcome-card');
        if (welcomeCard) welcomeCard.remove();

        // Render User Message
        appendMessage('user', text);
        userInput.value = '';

        // Show Typing Indicator
        typingIndicator.classList.remove('hidden');
        chatContainer.scrollTop = chatContainer.scrollHeight;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, brand: selectedBrand })
            });

            const data = await response.json();
            typingIndicator.classList.add('hidden');

            if (data.status === 'success') {
                appendMessage('agent', data.generated_reply, data);
                currentEvidence = data.retrieved_evidence || [];
                renderEvidence(currentEvidence);
            } else {
                appendMessage('agent', 'Error: Unable to process request.');
            }
        } catch (err) {
            typingIndicator.classList.add('hidden');
            appendMessage('agent', 'Error connecting to server.');
            console.error(err);
        }
    });

    // Append Message to Chat Container
    function appendMessage(sender, text, data = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'chat-avatar';
        avatar.innerHTML = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';

        if (sender === 'agent' && data) {
            // Handoff Banner
            if (data.escalation_action === 'escalate') {
                const banner = document.createElement('div');
                banner.className = 'handoff-banner danger';
                banner.innerHTML = `<strong>🚨 HUMAN HANDOFF TRIGGERED (${data.risk_level.toUpperCase()} RISK)</strong><br><small>Reason: ${data.escalation_reason} | ${data.escalation_explanation || ''}</small>`;
                bubble.appendChild(banner);
            } else {
                const banner = document.createElement('div');
                banner.className = 'handoff-banner success';
                banner.innerHTML = `<strong>✅ AUTONOMOUS RAG HANDLE (${data.risk_level.toUpperCase()} RISK)</strong><br><small>Grounding: ${data.grounding_status}</small>`;
                bubble.appendChild(banner);
            }

            // Meta Info
            const meta = document.createElement('div');
            meta.className = 'message-meta';
            meta.innerHTML = `<span>Target: <b>@${data.selected_brand}</b></span> • <span>Latency: <b>${data.latency_ms} ms</b></span>`;
            bubble.appendChild(meta);
        }

        const textDiv = document.createElement('div');
        textDiv.innerHTML = text.replace(/\n/g, '<br>');
        bubble.appendChild(textDiv);

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(bubble);

        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Render Evidence in Modal Drawer
    function renderEvidence(evidenceList) {
        evidenceContent.innerHTML = '';
        if (!evidenceList || evidenceList.length === 0) {
            evidenceContent.innerHTML = '<p class="empty-state">No historical evidence retrieved for this query.</p>';
            return;
        }

        evidenceList.forEach((ev, i) => {
            const item = document.createElement('div');
            item.className = 'evidence-item';
            item.innerHTML = `
                <div class="evidence-item-header">
                    <span>Exemplar #${i + 1} [@${ev.brand || 'Unknown'}]</span>
                    <span>Dist: ${ev.similarity_score}</span>
                </div>
                <div style="margin-bottom:0.4rem;"><strong>Customer:</strong> ${ev.customer_text}</div>
                <div><strong>Agent Reply:</strong> <em>${ev.agent_text}</em></div>
            `;
            evidenceContent.appendChild(item);
        });
    }

    // Drawer Controls
    const toggleRulesBtn = document.getElementById('toggleRulesBtn');
    const rulesDrawer = document.getElementById('rulesDrawer');
    const closeRulesBtn = document.getElementById('closeRulesBtn');

    if (toggleRulesBtn && rulesDrawer && closeRulesBtn) {
        toggleRulesBtn.addEventListener('click', () => rulesDrawer.classList.toggle('hidden'));
        closeRulesBtn.addEventListener('click', () => rulesDrawer.classList.add('hidden'));
    }

    toggleEvidenceBtn.addEventListener('click', () => evidenceDrawer.classList.toggle('hidden'));
    closeDrawerBtn.addEventListener('click', () => evidenceDrawer.classList.add('hidden'));

    // Clear Chat
    clearChatBtn.addEventListener('click', () => {
        chatContainer.innerHTML = `
            <div class="welcome-card">
                <div class="welcome-icon"><i class="fa-solid fa-comments"></i></div>
                <h2>Welcome to Multi-Brand Agentic RAG</h2>
                <p>Select a brand on the left sidebar and ask a customer question to see real-time vector retrieval and Gemini synthesis!</p>
            </div>
        `;
        evidenceContent.innerHTML = '<p class="empty-state">Submit a query to inspect raw historical ChromaDB evidence matches.</p>';
    });

    // Initialize
    fetchBrands();
    renderPresets("All Brands");
});
