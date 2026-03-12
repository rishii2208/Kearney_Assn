# FastAPI Project

## Project Structure

```
.
├── backend/          # FastAPI application code
├── frontend/         # Frontend assets
├── data/             # Data storage
│   ├── raw/          # Raw data files
│   ├── processed/    # Processed data files
│   ├── index/        # Index files
│   ├── eval/         # Evaluation data
│   └── metrics/      # Metrics data
└── docs/             # Documentation
```

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
uvicorn backend.main:app --reload
```

The API will be available at http://localhost:8000
API documentation at http://localhost:8000/docs
