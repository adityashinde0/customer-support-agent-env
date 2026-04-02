from fastapi import FastAPI, HTTPException
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