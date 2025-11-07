#!/usr/bin/env python3
"""
NER Chat Entity Extractor
Uses spaCy NER model + custom patterns for financial entity extraction
"""

import re
import json
import spacy
from spacy.matcher import Matcher
from typing import Dict, List, Optional
from pathlib import Path


class NERChatExtractor:
    """NER-based entity extractor for financial chat messages"""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize with spaCy model

        Args:
            model_name: spaCy model name (en_core_web_sm, en_core_web_md, en_core_web_lg)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Downloading...")
            import os
            os.system(f"python3 -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)

        # Initialize matcher for custom patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._add_custom_patterns()

        self.entities = {}
        self.text = ""

    def _add_custom_patterns(self):
        """Add custom patterns for financial entities"""

        # Pattern for payment frequency
        frequency_pattern = [
            {"LOWER": {"IN": ["quarterly", "monthly", "annual", "semi-annual", "daily"]}}
        ]
        self.matcher.add("FREQUENCY", [frequency_pattern])

    def extract_from_text(self, text: str) -> Dict[str, any]:
        """
        Main extraction method

        Args:
            text: Chat message text

        Returns:
            Dictionary of extracted entities
        """
        self.text = text
        self.entities = {}

        # Process with spaCy NER
        doc = self.nlp(text)

        # Step 1: Extract using pre-trained NER
        self._extract_with_pretrained_ner(doc)

        # Step 2: Extract using custom patterns
        self._extract_with_patterns(doc)

        # Step 3: Extract using regex (for specific formats)
        self._extract_with_regex()

        # Step 4: Post-process and clean
        self._clean_entities()

        return self.entities

    def extract_from_file(self, file_path: str) -> Dict[str, any]:
        """
        Extract entities from a text file

        Args:
            file_path: Path to the chat text file

        Returns:
            Dictionary of extracted entities
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return self.extract_from_text(text)

    def _extract_with_pretrained_ner(self, doc):
        """Extract entities using spaCy's pre-trained NER"""

        # Look for organizations with "BANK" prefix to capture full name
        org_pattern = r'\b(BANK\s+[A-Z]+)\b'
        org_matches = re.findall(org_pattern, self.text)
        if org_matches and "Counterparty" not in self.entities:
            self.entities["Counterparty"] = org_matches[0]

        # Fallback to spaCy NER for organizations
        for ent in doc.ents:
            # Organization → Counterparty
            if ent.label_ == "ORG":
                if "Counterparty" not in self.entities:
                    self.entities["Counterparty"] = ent.text

            # Money → Notional (partially)
            elif ent.label_ == "MONEY":
                if "Notional" not in self.entities:
                    self.entities["Notional"] = ent.text

    def _extract_with_patterns(self, doc):
        """Extract entities using custom patterns"""

        matches = self.matcher(doc)

        for match_id, start, end in matches:
            span = doc[start:end]
            match_label = self.nlp.vocab.strings[match_id]

            if match_label == "FREQUENCY":
                if "PaymentFrequency" not in self.entities:
                    self.entities["PaymentFrequency"] = span.text.capitalize()

    def _extract_with_regex(self):
        """Extract entities using regex patterns"""

        # ISIN pattern: 2 letters + 10 alphanumeric
        isin_pattern = r'\b([A-Z]{2}[A-Z0-9]{10})\b'
        isin_matches = re.findall(isin_pattern, self.text)
        if isin_matches:
            self.entities["ISIN"] = isin_matches[0]

        # Notional: "200 mio", "100 million", "50M"
        notional_pattern = r'(\d+(?:\.\d+)?)\s*(mio|million|M|bn|billion|B)'
        notional_matches = re.findall(notional_pattern, self.text, re.IGNORECASE)
        if notional_matches:
            amount, unit = notional_matches[0]
            self.entities["Notional"] = f"{amount} {unit}"

        # Bid/Offer rates: "estr+45bps", "3.5%", "euribor+20"
        rate_pattern = r'((?:estr|euribor|libor|sofr)\s*[+\-]\s*\d+(?:\.\d+)?\s*bps|\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*bps)'
        rate_matches = re.findall(rate_pattern, self.text, re.IGNORECASE)

        # In trading chats, "offer" keyword means they're providing a BID
        # Extract the rate that appears after "offer" keyword as Bid
        for i, rate in enumerate(rate_matches):
            # Check if "offer" appears before the rate
            rate_index = self.text.lower().find(rate.lower())
            text_before_rate = self.text[:rate_index].lower()

            # If "offer" keyword appears before this rate, it's a Bid
            if "offer" in text_before_rate and "Bid" not in self.entities:
                self.entities["Bid"] = rate
            # If "bid" keyword explicitly appears, it's also a Bid
            elif "bid" in text_before_rate and "Bid" not in self.entities:
                self.entities["Bid"] = rate

        # Underlying: Often appears after ISIN (tab-separated or space)
        # Pattern: After ISIN, capture until date or newline
        if "ISIN" in self.entities:
            isin = self.entities["ISIN"]
            isin_index = self.text.find(isin)

            # Get text after ISIN
            after_isin = self.text[isin_index + len(isin):].strip()

            # Extract until newline or certain keywords
            underlying_match = re.match(r'([^\n]+?)(?=\s+offer|\s+bid|\n|$)', after_isin)
            if underlying_match:
                underlying_text = underlying_match.group(1).strip()
                # Clean up tabs and extra spaces
                underlying_text = re.sub(r'\s+', ' ', underlying_text)
                if underlying_text and len(underlying_text) > 3:
                    self.entities["Underlying"] = underlying_text

        # Maturity - look for pattern with optional suffix (EVG, FWD, etc.)
        if "Maturity" not in self.entities:
            # Priority 1: Match "2Y EVG" type patterns from "offer" context
            maturity_pattern_offer = r'offer\s+(\d+[YMD]\s+[A-Z]{2,4})'
            offer_maturity = re.search(maturity_pattern_offer, self.text, re.IGNORECASE)
            if offer_maturity:
                self.entities["Maturity"] = offer_maturity.group(1).strip()
            else:
                # Priority 2: General maturity pattern with suffix
                maturity_pattern = r'\b(\d+[YMD]\s+(?:EVG|FWD|[A-Z]{2,4}))\b'
                maturity_matches = re.findall(maturity_pattern, self.text)
                if maturity_matches:
                    self.entities["Maturity"] = maturity_matches[0].strip()
                else:
                    # Priority 3: Just the period without suffix
                    maturity_pattern_simple = r'\b(\d+[YMD])\b'
                    simple_matches = re.findall(maturity_pattern_simple, self.text)
                    if simple_matches:
                        # Take the last occurrence as it's usually in the offer line
                        self.entities["Maturity"] = simple_matches[-1]

    def _clean_entities(self):
        """Clean and normalize extracted entities"""

        # Normalize case for Payment Frequency
        if "PaymentFrequency" in self.entities:
            self.entities["PaymentFrequency"] = self.entities["PaymentFrequency"].capitalize()

    def to_json(self, indent=2) -> str:
        """Convert entities to JSON string"""
        return json.dumps(self.entities, indent=indent, ensure_ascii=False)

    def to_dict(self) -> Dict:
        """Return entities as dictionary"""
        return self.entities


def main():
    """Command-line interface"""
    import sys

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python3 ner_chat_extractor.py <path_to_chat_file>")
        print("\nExample: python3 ner_chat_extractor.py FR001400QV82_AVMAFC_30Jun2028.txt")
        sys.exit(1)

    chat_file = sys.argv[1]

    # Check file exists
    if not Path(chat_file).exists():
        print(f"Error: File '{chat_file}' not found!")
        sys.exit(1)

    # Extract entities
    print(f"\n{'='*60}")
    print(f"NER Chat Entity Extractor - spaCy NER Model")
    print(f"{'='*60}")
    print(f"\nProcessing: {chat_file}\n")

    try:
        # Initialize extractor
        print("Loading spaCy NER model...")
        extractor = NERChatExtractor(model_name="en_core_web_sm")

        # Read chat file
        with open(chat_file, 'r', encoding='utf-8') as f:
            chat_text = f.read()

        print(f"\nChat Text:\n{'-'*60}\n{chat_text}\n{'-'*60}\n")

        # Extract entities
        print("Extracting entities...")
        entities = extractor.extract_from_text(chat_text)

        # Display results
        print("\nExtracted Entities:")
        print("-" * 60)
        print(extractor.to_json())
        print("-" * 60)
        print(f"\nTotal entities extracted: {len(entities)}")

    except Exception as e:
        print(f"Error processing chat: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
