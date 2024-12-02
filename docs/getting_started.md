
# Getting Started with TAPSec

Welcome to **TAPSec**, a CLI tool for efficient threat hunting. This guide will help you set up TAPSec and get started.

---

## 🚀 Prerequisites

1. **Python**: Version 3.8 or higher.
2. **Git**: Installed on your system.
3. **Active Internet Connection**: Required for downloading dependencies and additional resources.

---

## 🛠️ Installation

### Step 1: Clone the Repository
Run the following command to clone the TAPSec repository:
```bash
git clone https://github.com/shahd18/TAPSec.git
cd TAPSec
```

### Step 2: Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data
Run the following Python commands to download necessary NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Step 4: Set Up Models
1. Download the **CVE Model**  files from [Google Drive](https://drive.google.com/drive/folders/1fe2G1kv7KfhjrJzlwvPmc1oFFkajwUir?usp=sharing).
2. Download the **MITRE Model** files from [Google Drive](https://drive.google.com/drive/folders/1cRH1H58kDWgJZSQjaAm2CddAApaL8E8G?usp=sharing).
3. Place the downloaded files in the correct directories:
   ```
   TAPSec/
   ├── CVE-Model/
   │   ├── model.safetensors
   │   ├── vocab.txt
   │   ├── config.json
   │   ├── special_tokens_map.json
   │   └── tokenizer_config.json
   ├── MITRE-Model/
   │   ├── model.safetensors
   │   ├── vocab.txt
   │   ├── config.json
   │   ├── special_tokens_map.json
   │   └── tokenizer_config.json
   ```

---

## 🎉 Running TAPSec

To verify your setup, run:
```bash
python TAPSec.py --help
```

This should display a list of available commands and options.

---

## 📚 Next Steps

- **[Usage Guide](usage.md)**: Learn how to use TAPSec for rule classification and CVE mapping.
- **[Architecture Guide](architecture.md)**: Understand the internal structure and workflow of TAPSec.

You’re now ready to start using TAPSec! 🚀
