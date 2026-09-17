import { useState, useEffect } from 'react'
import './index.css'

function App() {
  const [brands, setBrands] = useState([])
  const [selectedBrand, setSelectedBrand] = useState("")
  const [query, setQuery] = useState("")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // Fetch available brands from backend
  useEffect(() => {
    fetch('http://localhost:8000/api/brands')
      .then(res => res.json())
      .then(data => {
        if (data.brands) {
          setBrands(data.brands)
        }
      })
      .catch(err => console.error("Failed to load brands:", err))
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!selectedBrand || !query.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: query,
          brand: selectedBrand
        })
      })

      if (!response.ok) throw new Error("Failed to process query")
      
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Mock company details based on selection
  const getCompanyDetails = (brand) => {
    const details = {
      "AmazonHelp": { industry: "E-Commerce", avgWait: "2m", priority: "High" },
      "AppleSupport": { industry: "Consumer Electronics", avgWait: "5m", priority: "High" },
      "Uber_Support": { industry: "Ride Sharing", avgWait: "1m", priority: "Critical" },
      "SpotifyCares": { industry: "Media Streaming", avgWait: "10m", priority: "Medium" },
      "BofA_Help": { industry: "Banking & Finance", avgWait: "3m", priority: "Critical" },
      "default": { industry: "General", avgWait: "5m", priority: "Standard" }
    }
    return details[brand] || details["default"]
  }

  const companyInfo = selectedBrand ? getCompanyDetails(selectedBrand) : null

  return (
    <>
      <h1>Agentic RAG Engine</h1>
      <p className="subtitle">Multi-Brand AI Customer Support Triage</p>

      <div className="app-grid">
        {/* Left Column: Controls & Input */}
        <div className="glass-panel">
          <h2>1. Target Selection</h2>
          <select 
            value={selectedBrand} 
            onChange={(e) => {
              setSelectedBrand(e.target.value)
              setResult(null)
            }}
          >
            <option value="" disabled>Select a brand to begin...</option>
            {brands.map(b => (
              <option key={b} value={b}>@{b}</option>
            ))}
          </select>

          {companyInfo && (
            <div className="brand-info">
              <div className="brand-avatar">
                {selectedBrand.charAt(0).toUpperCase()}
              </div>
              <div className="brand-details">
                <h3>@{selectedBrand}</h3>
              </div>
            </div>
          )}

          <h2 style={{marginTop: '2rem'}}>2. Incoming Query</h2>
          <form onSubmit={handleSubmit}>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={selectedBrand ? `Type a customer message directed at @${selectedBrand}...` : "Please select a brand above to unlock query input."}
              rows="5"
              disabled={!selectedBrand}
            />
            
            <button 
              type="submit" 
              disabled={!selectedBrand || !query.trim() || loading}
              style={{marginTop: '1rem'}}
            >
              {loading ? <div className="spinner"></div> : "Process Query"}
            </button>
          </form>
          
          {error && <div className="alert alert-error">{error}</div>}
        </div>

        {/* Right Column: Results */}
        <div className="glass-panel" style={{display: 'flex', flexDirection: 'column'}}>
          <h2>Agent Analysis & Response</h2>
          
          {!result && !loading && (
            <div style={{opacity: 0.5, textAlign: 'center', margin: 'auto'}}>
              <p>No query processed yet.</p>
              <p>Select a brand and submit a query to see the RAG agent in action.</p>
            </div>
          )}

          {loading && (
             <div style={{margin: 'auto', textAlign: 'center'}}>
               <div className="spinner" style={{borderColor: 'rgba(59, 130, 246, 0.3)', borderTopColor: '#3b82f6', width: '48px', height: '48px'}}></div>
               <p style={{marginTop: '1rem'}}>Analyzing intent & querying ChromaDB...</p>
             </div>
          )}

          {result && !loading && (
            <div style={{animation: 'fadeIn 0.5s ease-out'}}>
              {result.escalation_action === "escalate" ? (
                <div className="alert alert-error" style={{borderLeftColor: '#ef4444'}}>
                  <strong>🚨 HUMAN HANDOFF TRIGGERED ({result.risk_level} RISK)</strong>
                  <p style={{margin: '0.5rem 0 0 0'}}>Reason: {result.escalation_reason}</p>
                </div>
              ) : (
                <div className="alert alert-success">
                  <strong>✅ AUTONOMOUS RAG HANDLE ({result.risk_level} RISK)</strong>
                </div>
              )}

              <div style={{marginTop: '2rem'}}>
                <h3 style={{fontSize: '1.1rem', marginBottom: '0.5rem'}}>Synthesized Reply</h3>
                <div style={{background: 'rgba(15, 23, 42, 0.6)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(148, 163, 184, 0.1)'}}>
                  {result.generated_reply}
                </div>
              </div>

              {result.judge_evaluation && result.judge_evaluation.status === "EXECUTED" && (
                <div style={{marginTop: '2rem'}}>
                  <h3 style={{fontSize: '1.1rem', marginBottom: '0.5rem'}}>LLM Judge Evaluation <span style={{color: '#fbbf24'}}>★ {result.judge_evaluation.composite_score}/5.0</span></h3>
                  <div style={{background: 'rgba(15, 23, 42, 0.6)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(148, 163, 184, 0.1)'}}>
                    <div style={{display: 'flex', flexWrap: 'wrap', gap: '1rem', marginBottom: '1rem', fontSize: '0.9rem'}}>
                      <div className="badge">Relevance: {result.judge_evaluation.dimensional_scores.relevance}/5</div>
                      <div className="badge">Correctness: {result.judge_evaluation.dimensional_scores.correctness}/5</div>
                      <div className="badge">Groundedness: {result.judge_evaluation.dimensional_scores.groundedness}/5</div>
                      <div className="badge">Tone: {result.judge_evaluation.dimensional_scores.tone}/5</div>
                      <div className="badge">Helpfulness: {result.judge_evaluation.dimensional_scores.helpfulness}/5</div>
                    </div>
                    <p style={{margin: 0, fontSize: '0.9rem', color: '#94a3b8'}}><strong>Critique:</strong> {result.judge_evaluation.critique}</p>
                  </div>
                </div>
              )}

              {result.retrieved_evidence && result.retrieved_evidence.length > 0 && (
                <div style={{marginTop: '2rem'}}>
                  <h3 style={{fontSize: '1.1rem', marginBottom: '0.5rem'}}>Retrieved Evidence</h3>
                  {result.retrieved_evidence.map((ev, idx) => (
                    <div key={idx} style={{background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '8px', marginBottom: '1rem', fontSize: '0.9rem'}}>
                      <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                        <span className="badge">@{ev.brand || 'Unknown'}</span>
                        <span style={{color: '#94a3b8'}}>Sim: {ev.similarity_score ? ev.similarity_score.toFixed(4) : 'N/A'}</span>
                      </div>
                      <p style={{margin: '0 0 0.5rem 0'}}><strong>Q:</strong> {ev.customer_text}</p>
                      <p style={{margin: '0', color: '#94a3b8'}}><strong>A:</strong> {ev.agent_text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default App
