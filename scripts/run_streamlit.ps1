$ErrorActionPreference = "Stop"

Set-Location "C:\Users\bridg\OneDrive\Desktop\house-ml-ops"
python -m streamlit run streamlit_app.py --server.headless true --server.port 8501
