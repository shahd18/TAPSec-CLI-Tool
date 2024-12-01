
# Getting Started with TAPSec

This guide helps new users set up TAPSec and run it for the first time.

---

## Prerequisites
1. Python 3.8 or higher
2. Git installed on your system
3. An active internet connection for downloading dependencies

---

## Installation

### Step 1: Clone the Repository
Run the following command to clone the TAPSec repository:
```bash
git clone https://github.com/<your-username>/TAPSec.git
cd TAPSec
```

### Step 2: Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data
Run the following Python code to download necessary NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

---

## Running TAPSec
To verify the setup, run:
```bash
python TAPSec.py --help
```

You should see a list of available commands and options.

---

## Next Steps
- [Usage Guide](usage.md): Learn how to use TAPSec for rule classification and CVE mapping.
- [Architecture Guide](architecture.md): Understand the internal structure and workflow.
