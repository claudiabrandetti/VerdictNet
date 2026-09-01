"""
VerdictNet / Texas Legal Graph — Local Development Server
Boots the FastAPI server and static frontend on http://localhost:8000
"""

import sys
import os
import webbrowser
import uvicorn

def main():
    print("=" * 65)
    print("🏛️  Starting Texas Legal Graph (VerdictNet) Local Dev Server")
    print("⚡ Powered by Google Gemini 3.7 Flash & Neo4j Citation Graph")
    print("=" * 65)
    print("\n👉 Web UI available at: http://localhost:8000")
    print("👉 API Documentation:    http://localhost:8000/docs\n")
    
    # Try to open browser automatically
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass
        
    uvicorn.run("api.index:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
