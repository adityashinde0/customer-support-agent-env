import uvicorn
import sys
import os

# Add the parent directory to the path so it can find your existing api.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    uvicorn.run("api:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()