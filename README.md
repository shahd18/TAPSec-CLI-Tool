
# TAPSec — A CLI Tool for Threat Analysis

TAPSec is a Python-based Command-Line Interface (CLI) tool designed for efficient threat analysis. It classifies input rules, maps them to CVEs, and provides actionable mitigations. TAPSec is designed for security analysts who need quick, reliable, and scalable threat analysis.

---

![Python](https://img.shields.io/badge/python-3.8-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)

---

## 🚀 Features

- **Rule Classification**: Classifies rules as malicious or innocent using a trained ML model.
- **CVE Mapping**: Maps rules to the best-matched CVEs using semantic similarity.
- **MITRE Mapping**: Provides actionable mitigations based on MITRE ATT&CK techniques.
- **Interactive CLI**: Offers a seamless user experience through the command line.
- **Extensibility**: Allows easy addition of new models and features.

---

## TAPSec CLI

Below is the TAPSec ASCII art displayed in the CLI:

![TAPSec ASCII Art](images/Screenshot%202024-12-01%20051753.png)

## 🛠️ Installation

Follow these steps to install and set up TAPSec:

1. **Clone the Repository**:
   Use the following command to clone the repository:
   ```bash
   git clone https://github.com/<your-username>/TAPSec.git
   cd TAPSec
   ```

2. **Install Dependencies**:
   Run the following command to install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Necessary NLTK Data**:
   Open a Python shell and run these commands to download stopwords and other necessary resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

4. **Verify Installation**:
   Ensure everything is set up correctly by running:
   ```bash
   python TAPSec.py --help
   ```

---

## 📜 Usage

Run the following command to classify a rule and retrieve CVE mappings:
```bash
python TAPSec.py -r "<rule>" -a "<application>"
```

### Examples

#### Example 1: Classify a Firewall Rule
```bash
python TAPSec.py -r "deny IP address 192.168.1.1" -a "firewall"
```

**Output:**
- Classification: Malicious
- Best Matched CVE: CVE-2023-12345
- Suggested Mitigations: Update firewall rules and monitor logs.

#### Example 2: Analyze an Email Rule
```bash
python TAPSec.py -r "block emails from suspicious domains" -a "email"
```

**Output:**
- Classification: Malicious
- Best Matched CVE: CVE-2024-56789
- Suggested Mitigations: Enable spam filtering and review policies.

---

## 📁 Directory Structure

Below is the folder structure of the TAPSec repository, with explanations for each component:

```
TAPSec/
├── TAPSec.py                  # Main Python script for the CLI tool.
├── Classification-Model/      # Folder for classification models.
│   ├── tfidf_vectorizer.pkl     # Pre-trained TF-IDF vectorizer.
│   ├── xgboost_best_model.joblib # Pre-trained XGBoost model for rule classification.
├── CVE-Model/                 # Folder for CVE mapping models.
│   ├── model.safetensors        # Model file for CVE embeddings.
│   ├── vocab.txt                # Vocabulary file for the CVE model.
├── MITRE-Model/               # Folder for MITRE mapping models.
│   ├── model.safetensors        # Model file for MITRE mappings.
├── Dataset/                   # Folder containing datasets for analysis.
│   ├── Processed_CVE_withSpace.csv # Processed CVE dataset for mapping.
│   ├── Processed_mitre_recommidtaion.csv # Processed MITRE dataset for mitigations.
├── docs/                      # Documentation files.
│   ├── architecture.md          # Detailed description of the project structure and workflow.
│   ├── usage.md                 # Usage instructions and examples.
│   ├── getting_started.md       # Setup guide for new users.
├── Design/                    # ASCII art and design assets.
│   ├── tooldesign1.txt          # ASCII art displayed in the CLI.
├── LICENSE                    # License file (MIT License for this project).
├── README.md                  # Main README file for the repository.
├── requirements.txt           # List of Python dependencies required for the project.
└── scripts/                   # Utility scripts.
    ├── setup_environment.sh     # Shell script for setting up the environment (optional).
    ├── test_tapsec.py           # Script for testing TAPSec functionality.
```

---

## 📚 Documentation

Refer to the following documentation files for more details:

- [Architecture](docs/architecture.md): Explains the internal structure and workflow of TAPSec.
- [Usage Guide](docs/usage.md): Provides examples and step-by-step instructions for using the tool.
- [Getting Started](docs/getting_started.md): Offers setup instructions for new users.

---

## 🌐 Contributing

We welcome contributions! Follow these steps to contribute:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/<feature-name>
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add <feature-name>"
   ```
4. Push your changes:
   ```bash
   git push origin feature/<feature-name>
   ```
5. Submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🎉 Acknowledgments

Special thanks to the open-source libraries and tools that make TAPSec possible.
