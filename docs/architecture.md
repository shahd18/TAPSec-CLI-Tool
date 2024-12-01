
# Architecture of TAPSec

## Overview
TAPSec is designed as a modular tool for threat analysis, combining classification, CVE mapping, and MITRE recommendations. The tool leverages machine learning and natural language processing (NLP) techniques to ensure scalability and accuracy.

---

## Key Components

1. **Rule Classification**:
   - Identifies whether a rule is malicious or innocent using an XGBoost model.
   - Features are extracted using a TF-IDF vectorizer.

2. **CVE Mapping**:
   - Maps input to the closest matching CVEs using Sentence Transformers.
   - Embedding-based similarity ensures accurate mapping.

3. **MITRE Mapping**:
   - Maps CVEs to MITRE techniques for actionable recommendations.
   - Outputs relevant mitigations based on the MITRE ATT&CK framework.

---

## Workflow

1. **Input Preprocessing**:
   - User inputs (rules) are cleaned and tokenized.
   - Stopwords are removed using NLTK.

2. **Classification**:
   - The processed rule is classified as malicious or innocent.
   - If classified as innocent, no further action is performed.

3. **CVE Matching**:
   - For malicious rules, embeddings are generated using Sentence Transformers.
   - Cosine similarity is used to find the best-matched CVE.

4. **MITRE Mapping**:
   - Matched CVEs are used to identify relevant MITRE techniques.
   - Recommendations are displayed for mitigating the threat.

---

## Directory Structure

```
TAPSec/
├── TAPSec.py                  # Main Python script for the CLI tool.
├── Classification-Model/      # Pre-trained classification models.
├── CVE-Model/                 # Embedding models for CVE mapping.
├── MITRE-Model/               # Embedding models for MITRE recommendations.
├── Dataset/                   # Processed datasets for CVE and MITRE mappings.
├── docs/                      # Documentation files.
├── Design/                    # ASCII art and design assets.
├── LICENSE                    # License file (MIT License).
├── README.md                  # Main README file.
├── requirements.txt           # Python dependencies.
└── scripts/                   # Utility scripts for testing and setup.
```

---

## Technologies Used

- **spaCy**: Text preprocessing and tokenization.
- **NLTK**: Stopword removal and lemmatization.
- **XGBoost**: Rule classification model.
- **Sentence Transformers**: Embedding generation for CVE mapping.
- **Typer**: CLI framework for building user-friendly commands.

---

## Modular Design
TAPSec adopts a modular design to simplify development and scalability:
1. **Models**: Pre-trained models for classification and mapping are stored separately.
2. **Datasets**: Organized datasets for ease of processing and updates.
3. **Documentation**: Comprehensive documentation for developers and users.

---

## Extensibility
The modular design allows for:
- Easy integration of new classification models.
- Addition of new datasets for CVEs and mitigations.
- Customization of workflows for specific use cases.
