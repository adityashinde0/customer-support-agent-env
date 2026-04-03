from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from environment import CustomerSupportEnv
from models import Action

# Initialize the FastAPI application
app = FastAPI(title="Customer Support OpenEnv API")

# Create a single, persistent instance of our environment
env_instance = CustomerSupportEnv()

@app.post("/reset")
def reset_environment():
    """Starts a new episode and returns the initial observation."""
    try:
        obs = env_instance.reset()
        # FastAPI automatically converts Pydantic models to JSON!
        return {"observation": obs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_environment(action: Action):
    """Takes an action from the AI and advances the environment by one step."""
    # Prevent the AI from taking a step if the game hasn't started
    if env_instance.obs is None:
        raise HTTPException(status_code=400, detail="You must call /reset before calling /step.")
    
    try:
        obs, reward, done, info = env_instance.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_current_state():
    """Returns the current observation without taking a step."""
    if env_instance.obs is None:
        raise HTTPException(status_code=400, detail="Environment has not been initialized. Call /reset.")
    
    return {"observation": env_instance.state()}

@app.get("/", response_class=HTMLResponse)
def home():
    """Provides a fully interactive GUI for the environment."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Support Buddy OpenEnv</title>
        <style>
            :root { --primary: #2563eb; --bg: #f8fafc; --panel: #ffffff; --text: #1e293b; --border: #e2e8f0; --success: #22c55e; --danger: #ef4444; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 20px; display: flex; justify-content: center; }
            .header-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; width: 100%; max-width: 1200px; }
            .docs-btn { background-color: #475569; color: white; padding: 8px 16px; text-decoration: none; border-radius: 6px; font-size: 14px; font-weight: bold; transition: 0.2s; }
            .docs-btn:hover { background-color: #334155; }
            .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1200px; width: 100%; }
            .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            h2 { margin-top: 0; font-size: 18px; color: #0f172a; border-bottom: 2px solid var(--border); padding-bottom: 10px; margin-bottom: 15px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; font-weight: 600; margin-bottom: 5px; font-size: 14px; }
            select, input[type="text"] { width: 100%; padding: 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 14px; box-sizing: border-box; }
            .btn { background-color: var(--primary); color: white; border: none; padding: 10px 15px; border-radius: 6px; cursor: pointer; font-weight: bold; width: 100%; margin-top: 10px; }
            .btn:hover { background-color: #1d4ed8; }
            .btn-secondary { background-color: #64748b; width: auto; margin-top: 0; }
            .btn-secondary:hover { background-color: #475569; }
            .controls { display: flex; gap: 10px; margin-top: 20px; }
            .status-box { background: #f1f5f9; padding: 15px; border-radius: 6px; margin-top: 20px; font-family: monospace; font-size: 14px; line-height: 1.5; }
            .observation-clean { background: #f8fafc; border: 1px solid var(--border); padding: 15px; border-radius: 6px; min-height: 100px; font-size: 14px; line-height: 1.6; }
            .history-log { background: #1e293b; color: #f8fafc; padding: 15px; border-radius: 6px; margin-top: 20px; height: 300px; overflow-y: auto; font-family: monospace; font-size: 13px; }
            .log-entry { margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 10px; }
            .log-action { color: #38bdf8; }
            .log-reward { color: var(--success); font-weight: bold; }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div style="width: 100%; max-width: 1200px;">
            <div class="header-bar">
                <h1 style="margin: 0; font-size: 24px;">🎧 Support Buddy Interface</h1>
                <a href="/docs" class="docs-btn" target="_blank">View API Documentation ↗</a>
            </div>

            <div class="dashboard">
                <div>
                    <div class="panel">
                        <h2>Take Action</h2>
                        <div class="form-group">
                            <label>Action Type:</label>
                            <select id="actionType" onchange="toggleInputs()">
                                <option value="classify_issue">Classify Issue</option>
                                <option value="search_kb">Search Knowledge Base</option>
                                <option value="ask_clarifying_question">Ask Clarifying Question</option>
                                <option value="resolve_ticket">Resolve Ticket</option>
                                <option value="escalate_to_human">Escalate to Human</option>
                            </select>
                        </div>
                        
                        <div class="form-group hidden" id="categoryGroup">
                            <label>Category Guess:</label>
                            <select id="categoryGuess">
                                <option value="Billing">Billing</option>
                                <option value="Technical">Technical</option>
                                <option value="Refund_Request">Refund Request</option>
                            </select>
                        </div>

                        <div class="form-group hidden" id="searchGroup">
                            <label>Search Query:</label>
                            <input type="text" id="searchQuery" placeholder="e.g., refund policy">
                        </div>

                        <div class="form-group hidden" id="messageGroup">
                            <label>Message to Customer:</label>
                            <input type="text" id="messageToCustomer" placeholder="Type your message here...">
                        </div>

                        <button class="btn" onclick="takeAction()">Submit Action</button>
                    </div>

                    <div class="controls">
                        <button class="btn btn-secondary" onclick="resetEnv()">Reset Environment</button>
                        <button class="btn btn-secondary" onclick="getState()">Refresh State</button>
                    </div>

                    <div class="status-box" id="stateDisplay">
                        <strong>Status:</strong> Waiting to start...<br>
                        <strong>Episode ID:</strong> N/A<br>
                        <strong>Step Count:</strong> 0
                    </div>
                </div>

                <div>
                    <div class="panel">
                        <h2>Current Observation</h2>
                        <div class="observation-clean" id="obsDisplay">
                            Click 'Reset Environment' to load a ticket.
                        </div>

                        <h2>Action History</h2>
                        <div class="history-log" id="historyLog">
                            </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentStep = 0;

            function toggleInputs() {
                const action = document.getElementById('actionType').value;
                document.getElementById('categoryGroup').classList.add('hidden');
                document.getElementById('searchGroup').classList.add('hidden');
                document.getElementById('messageGroup').classList.add('hidden');

                if (action === 'classify_issue') document.getElementById('categoryGroup').classList.remove('hidden');
                if (action === 'search_kb') document.getElementById('searchGroup').classList.remove('hidden');
                if (action === 'ask_clarifying_question' || action === 'resolve_ticket') document.getElementById('messageGroup').classList.remove('hidden');
            }

            function updateUI(state, reward = null, done = null) {
                // Update State Panel
                document.getElementById('stateDisplay').innerHTML = `
                    <strong>Status:</strong> ${done ? '<span style="color:red">DONE</span>' : '<span style="color:green">RUNNING</span>'}<br>
                    <strong>Episode ID:</strong> ${state.ticket_id || 'N/A'}<br>
                    <strong>Step Count:</strong> ${state.step_count || 0}
                `;

                // Update Observation Panel cleanly
                let obsHtml = `<strong>Customer Tier:</strong> ${state.customer_tier || 'N/A'}<br>`;
                obsHtml += `<strong>Issue Category:</strong> ${state.issue_category || 'Not Classified Yet'}<br>`;
                obsHtml += `<strong>KB Result:</strong> ${state.kb_search_result || 'None'}<br><br>`;
                
                if (state.conversation_history && state.conversation_history.length > 0) {
                    obsHtml += `<strong>Conversation:</strong><br>`;
                    state.conversation_history.forEach(msg => {
                        obsHtml += `<em>${msg.sender}:</em> ${msg.content}<br>`;
                    });
                }
                document.getElementById('obsDisplay').innerHTML = obsHtml;
            }

            function logHistory(actionPayload, reward, done) {
                const logDiv = document.getElementById('historyLog');
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.innerHTML = `
                    <div class="log-action">Action: ${JSON.stringify(actionPayload)}</div>
                    <div class="log-reward">Reward: ${reward !== null ? reward : 0} ${done ? ' [EPISODE FINISHED]' : ''}</div>
                `;
                logDiv.prepend(entry);
            }

            async function resetEnv() {
                document.getElementById('historyLog').innerHTML = '';
                const response = await fetch('/reset', { method: 'POST' });
                const data = await response.json();
                updateUI(data);
                logHistory({ command: "System Reset" }, 0, false);
            }

            async function getState() {
                const response = await fetch('/state');
                const data = await response.json();
                updateUI(data);
            }

            async function takeAction() {
                const actionType = document.getElementById('actionType').value;
                let payload = { action_type: actionType };

                if (actionType === 'classify_issue') payload.category_guess = document.getElementById('categoryGuess').value;
                if (actionType === 'search_kb') payload.search_query = document.getElementById('searchQuery').value;
                if (actionType === 'ask_clarifying_question' || actionType === 'resolve_ticket') {
                    payload.message_to_customer = document.getElementById('messageToCustomer').value;
                }

                try {
                    const response = await fetch('/step', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    
                    const data = await response.json();
                    if (response.ok) {
                        updateUI(data.observation, data.reward, data.done);
                        logHistory(payload, data.reward, data.done);
                    } else {
                        alert("Error processing action. Check inputs.");
                    }
                } catch (err) {
                    console.error(err);
                }
            }

            // Initialize UI
            toggleInputs();
        </script>
    </body>
    </html>
    """