# GenAI Case Study Summary

## Overview

This project implements an Augmented Document Reader (ADOR) system for extracting financial entities from three types of documents: structured DOCX files, chat messages, and unstructured PDFs.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Packages

Install all dependencies using:
```bash
pip install python-docx spacy
python -m spacy download en_core_web_sm
```

Or install individually:
```bash
# For DOCX parser
pip install python-docx

# For NER chat extractor
pip install spacy
python -m spacy download en_core_web_sm
```

---

## Deliverables

### 1. docx_parser.py

- Rule-based parser for structured Word documents
- Extracts 9 entities (Counterparty, Initial Valuation Date, Notional, Valuation Date, Maturity, Underlying, Coupon, Barrier, Calendar)
- **Technology used**: python-docx library
- **Reason to choose python-docx library**: DOCX files have structured tables that can be parsed reliably with rule-based extraction.

## Usage

### Running docx_parser.py
```bash
python docx_parser.py 
```

**Example:**
```bash
python docx_parser.py ZF4894_ALV_07Aug2026_physical.docx
```

#### Expected Output:

![DOCX parser output](https://drive.google.com/uc?export=view&id=1Ani4EVjT7DXlY0OSo26Wmz6jP_ZKegBh)

---

### 2. ner_chat_extractor.py

- NER-based extractor for chat messages
- Extracts 7-8 entities (Counterparty, Notional, ISIN, Underlying, Maturity, Bid, Offer, PaymentFrequency)
- **Technology used**: spaCy NER model + custom regex patterns
- **Reason**: Chat messages are informal and unstructured, requiring NER models to understand context and extract entities accurately

### Running ner_chat_extractor.py
```bash
python ner_chat_extractor.py 
```

**Example:**
```bash
python ner_chat_extractor.py FR001400QV82_AVMAFC_30Jun2028.txt
```

#### Expected Output:

![NER chat Output](https://drive.google.com/uc?export=view&id=1k6qFYRJUgOYBNhNFQr038a3efcVPaobR)

---

### 3. NER_Fine_Tuning_Methodology.md

- Methodology document for fine-tuning NER models
- Contains training data requirements, fine-tuning process, evaluation metrics, expected accuracy improvements
- Explains how to improve NER accuracy from 80% to 92% through fine-tuning on financial domain data

---

### 4. Global_Architecture_Document.md

- Complete system architecture for production deployment
- Contains 16 components including API Gateway, services, databases, security, logging, monitoring, scalability
- **Technologies used in the architecture**: FastAPI, PostgreSQL, MongoDB, Redis, Kubernetes, RabbitMQ, Prometheus, Grafana
- **Reason**: Production system requires proper architecture with security, logging, error handling, and scalability

---

### 5. LLM_Methodology_Document.md

- Methodology for extracting entities from unstructured PDFs using LLMs
- Contains RAG implementation, prompting techniques, cost management, evaluation
- **Technologies used**: GPT-4, OpenAI embeddings, Pinecone vector database, pdfplumber
- **Reason**: PDFs are lengthy and unstructured, requiring LLMs with RAG to understand context and extract entities accurately
