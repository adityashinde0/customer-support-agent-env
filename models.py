from pydantic import BaseModel, Field
from typing import Optional, Literal, List

class Observation(BaseModel):
    """The current state of the environment that the agent can see."""
    
    ticket_id: str = Field(description="The unique ID of the current ticket.")
    customer_tier: Literal["Standard", "VIP"] = Field(description="The status level of the customer.")
    
    # These start empty and get updated as the AI works
    issue_category: Optional[str] = Field(default=None, description="The classification of the ticket if the agent has categorized it.")
    knowledge_base_result: Optional[str] = Field(default=None, description="The policy text returned after searching the KB.")
    
    # State tracking for the multi-step flow
    conversation_history: List[str] = Field(default_factory=list, description="A transcript of messages between the customer and the agent.")
    step_count: int = Field(default=0, description="How many actions the agent has taken so far. High step counts will be penalized.")
    
    # Episode boundary
    is_resolved: bool = Field(default=False, description="True if the ticket has been completely handled.")
    last_action_error: Optional[str] = Field(default=None, description="Will contain an error message if the agent's last action was invalid.")

class Action(BaseModel):
    """The specific move the agent wants to make this turn."""
    
    action_type: Literal[
        "ask_clarifying_question", 
        "classify_issue",          
        "search_kb", 
        "resolve_ticket", 
        "escalate_to_human"
    ] = Field(description="The exact action to perform.")
    
    # These fields are "Optional" because the agent only uses them if they match the action_type above
    message_to_customer: Optional[str] = Field(default=None, description="Required if action_type is 'ask_clarifying_question' or 'resolve_ticket'.")
    category_guess: Optional[Literal["Billing", "Technical", "Refund_Request"]] = Field(default=None, description="Required if action_type is 'classify_issue'.")
    search_query: Optional[str] = Field(default=None, description="Required if action_type is 'search_kb'.")

class Reward(BaseModel):
    """The feedback score given to the agent after taking an action."""
    
    value: float = Field(description="The numerical points awarded or deducted. Must be between 0.0 and 1.0 at the end.")
    reason: str = Field(description="A human-readable explanation of why this score was given.")