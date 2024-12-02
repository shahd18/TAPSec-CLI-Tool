
# Usage Guide for TAPSec

This guide provides detailed instructions on how to use TAPSec to classify input rules, map them to CVEs, and retrieve actionable mitigations.

---

## 📜 Overview

TAPSec offers a command-line interface for performing threat analysis with the following key features:
1. Rule Classification: Determine if a rule is malicious or innocent.
2. CVE Mapping: Find the best-matched CVEs for the given input.
3. MITRE Mapping: Suggest mitigations based on the matched CVEs.

---

## 🛠️ Basic Usage

To run TAPSec, use the following command:
```bash
python TAPSec.py -r "<rule>" -a "<application>"
```

- `-r <rule>`: Specify the rule text you want to classify.
- `-a <application>`: Specify the application or context of the rule.

### Example:
```bash
python TAPSec.py -r "if in my location then heat the air" -a "heat"
```

---

## 🔍 Command-Line Options

TAPSec supports the following CLI options:

| Option       | Description                                    | Example Usage                               |
|--------------|------------------------------------------------|---------------------------------------------|
| `-r`         | Rule text to be analyzed                       | `-r "if in my location then heat the air"`  |
| `-a`         | Application name (e.g., facebook, email)       | `-a "email"`                                |
| `--help`     | Display help for TAPSec commands               | `python TAPSec.py --help`                   |
| `--version`  | Display the current version of TAPSec          | `python TAPSec.py --version`                |

---

## 📂 Input Examples

### **1. Dropbox Rule Classification**
Command:
```bash
python TAPSec.py -r "Save SMS to Dropbox" -a "Dropbox"
```
**Output:**
- Classification: Malicious
- Best Matched CVE: CVE-2022-23597
- Suggested Mitigations: Restrict use of certain websites, block downloads/attachments, block Javascript, restrict browser extensions.


---

### **2. iOS Rule Classification**
Command:
```bash
python TAPSec.py -r "if photo on iOS then save to BOX account" -a "iOS"
```
**Output:**
- Classification: Malicious
- Best Matched CVE: CVE-2022-22161
- Suggested Mitigations: Ensure proper network segmentation is followed to protect critical servers and devices.

---

## 📄 Notes

1. Ensure that all required datasets are available in the `Dataset/` folder before running TAPSec.
2. The results are displayed on the console and optionally saved as a CSV file for further analysis.
3. Use `--help` to view all available commands and options.

---

## 🧑‍💻 Advanced Usage

### **Performing Debugging or Testing**
To test TAPSec functionalities without analyzing real data, use the test script provided:
```bash
python scripts/test_tapsec.py
```

### **Adding New Applications**
If you'd like to add a new application type:
1. Update the application mapping in `TAPSec.py`.
2. Add relevant models or data to the `Dataset/` folder.

---

## 📝 FAQ

### **Q: What happens if my input is classified as innocent?**
- TAPSec will terminate the process and notify you that no further action is required.

### **Q: Where are the results saved?**
- Results are displayed in the console and can optionally be saved as a CSV file in the current directory.

---

## 🚀 Next Steps

Once you've classified your rules and retrieved CVE mappings, you can:
- Integrate TAPSec into your workflow by calling it programmatically.
- Use the output CVEs and suggested mitigations for further security analysis.

---

For additional support, refer to the [Architecture Guide](architecture.md) or contact the project contributors.
