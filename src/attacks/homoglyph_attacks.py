"""Homoglyph-based attack implementations"""

import random
from typing import List, Dict, Any, Set


class HomoglyphAttacks:
    """Implements homoglyph-based character substitution attacks"""
    
    def __init__(self):
        # Common homoglyph mappings (visually similar characters)
        self.homoglyph_map = {
            # Latin to similar characters from other scripts
            'a': ['а', 'α', 'а'],  # Cyrillic а, Greek α
            'e': ['е', 'ε', 'е'],  # Cyrillic е, Greek ε  
            'o': ['о', 'ο', 'о', '0', 'ο'],  # Cyrillic о, Greek ο, digit 0
            'p': ['р', 'ρ', 'р'],  # Cyrillic р, Greek ρ
            'c': ['с', 'ϲ', 'с'],  # Cyrillic с, Greek ϲ
            'y': ['у', 'γ', 'у'],  # Cyrillic у, Greek γ
            'x': ['х', 'χ', 'х'],  # Cyrillic х, Greek χ
            'i': ['і', 'ι', 'і', '1', '|', 'l'],  # Cyrillic і, Greek ι, digit 1, pipe, l
            'A': ['А', 'Α', 'А'],  # Cyrillic А, Greek Α
            'B': ['В', 'Β', 'В'],  # Cyrillic В, Greek Β
            'E': ['Е', 'Ε', 'Е'],  # Cyrillic Е, Greek Ε
            'H': ['Н', 'Η', 'Н'],  # Cyrillic Н, Greek Η
            'K': ['К', 'Κ', 'К'],  # Cyrillic К, Greek Κ
            'M': ['М', 'Μ', 'М'],  # Cyrillic М, Greek Μ
            'N': ['Ν', 'Ν'],      # Greek Ν
            'O': ['О', 'Ο', 'О'],  # Cyrillic О, Greek Ο
            'P': ['Р', 'Ρ', 'Р'],  # Cyrillic Р, Greek Ρ
            'T': ['Т', 'Τ', 'Т'],  # Cyrillic Т, Greek Τ
            'X': ['Х', 'Χ', 'Х'],  # Cyrillic Х, Greek Χ
            'Y': ['У', 'Υ', 'У'],  # Cyrillic У, Greek Υ
            
            # Numbers and symbols
            '0': ['О', 'ο', 'о', '𝟘', '𝟎'],  # Letter O variants
            '1': ['l', 'I', '|', '𝟏', '𝟙'],   # Letter l, I, pipe
            '5': ['𝟓', '𝟝'],
            '6': ['б', '𝟔'],  # Cyrillic б
            
            # Common words with high-impact substitutions
            'admin': ['аdmin', 'аdmіn', 'αdmin'],
            'user': ['uѕer', 'uѕеr', 'υser'],
            'password': ['раssword', 'pаssword', 'pasѕword'],
            'login': ['lоgin', 'logіn', 'lоgіn'],
            'secure': ['ѕecure', 'secυre', 'sеcure'],
        }
        
        # Extended Unicode confusables
        self.confusables = {
            # Mathematical symbols
            'A': ['𝐀', '𝐴', '𝑨', '𝒜', '𝓐', '𝔄', '𝔸', '𝕬', '𝖠', '𝗔', '𝘈', '𝙰'],
            'B': ['𝐁', '𝐵', '𝑩', '𝓑', '𝔅', '𝔹', '𝕭', '𝖡', '𝗕', '𝘉', '𝙱'],
            'C': ['𝐂', '𝐶', '𝑪', '𝒞', '𝓒', '𝔇', 'ℂ', '𝕮', '𝖢', '𝗖', '𝘊', '𝙲'],
            
            # Fullwidth characters (used in some Asian contexts)
            'A': ['Ａ'], 'B': ['Ｂ'], 'C': ['Ｃ'], 'D': ['Ｄ'], 'E': ['Ｅ'],
            'F': ['Ｆ'], 'G': ['Ｇ'], 'H': ['Ｈ'], 'I': ['Ｉ'], 'J': ['Ｊ'],
            'K': ['Ｋ'], 'L': ['Ｌ'], 'M': ['Ｍ'], 'N': ['Ｎ'], 'O': ['Ｏ'],
            'P': ['Ｐ'], 'Q': ['Ｑ'], 'R': ['Ｒ'], 'S': ['Ｓ'], 'T': ['Ｔ'],
            'U': ['Ｕ'], 'V': ['Ｖ'], 'W': ['Ｗ'], 'X': ['Ｘ'], 'Y': ['Ｙ'], 'Z': ['Ｚ'],
        }

    def substitute_homoglyphs(self, text: str, substitution_rate: float = 0.3) -> str:
        """Substitute characters with homoglyphs"""
        result = list(text)
        
        for i, char in enumerate(result):
            if random.random() < substitution_rate:
                # Check direct character mapping
                if char in self.homoglyph_map:
                    alternatives = self.homoglyph_map[char]
                    result[i] = random.choice(alternatives)
                # Check confusables mapping
                elif char in self.confusables:
                    alternatives = self.confusables[char]
                    result[i] = random.choice(alternatives)
        
        return ''.join(result)

    def create_lookalike_domain(self, domain: str) -> List[str]:
        """Create lookalike domains using homoglyphs"""
        variants = []
        
        # Single character substitution
        for i, char in enumerate(domain):
            if char in self.homoglyph_map:
                alternatives = self.homoglyph_map[char]
                for alt in alternatives:
                    variant = domain[:i] + alt + domain[i+1:]
                    variants.append(variant)
        
        # Multiple character substitution
        variant = self.substitute_homoglyphs(domain, substitution_rate=0.5)
        if variant != domain:
            variants.append(variant)
        
        return variants

    def create_deceptive_text(self, text: str, strategy: str = "mixed") -> str:
        """Create deceptive text using various homoglyph strategies"""
        if strategy == "mixed":
            # Mix different scripts
            return self.substitute_homoglyphs(text, substitution_rate=0.4)
        
        elif strategy == "cyrillic_heavy":
            # Heavily favor Cyrillic substitutions
            result = list(text)
            for i, char in enumerate(result):
                if char in self.homoglyph_map:
                    # Prefer Cyrillic alternatives
                    alternatives = self.homoglyph_map[char]
                    cyrillic_alts = [alt for alt in alternatives if ord(alt) >= 0x0400 and ord(alt) <= 0x04FF]
                    if cyrillic_alts and random.random() < 0.6:
                        result[i] = random.choice(cyrillic_alts)
            return ''.join(result)
        
        elif strategy == "mathematical":
            # Use mathematical symbol variants
            result = list(text)
            for i, char in enumerate(result):
                if char in self.confusables and random.random() < 0.5:
                    alternatives = self.confusables[char]
                    # Prefer mathematical symbols
                    math_alts = [alt for alt in alternatives if ord(alt) >= 0x1D400]
                    if math_alts:
                        result[i] = random.choice(math_alts)
            return ''.join(result)
        
        elif strategy == "fullwidth":
            # Use fullwidth characters
            result = list(text)
            for i, char in enumerate(result):
                if char in self.confusables:
                    alternatives = self.confusables[char]
                    fullwidth_alts = [alt for alt in alternatives if ord(alt) >= 0xFF00 and ord(alt) <= 0xFFEF]
                    if fullwidth_alts and random.random() < 0.7:
                        result[i] = random.choice(fullwidth_alts)
            return ''.join(result)
        
        return text

    def generate_homoglyph_attacks(self, base_prompt: str, payload: str) -> List[Dict[str, Any]]:
        """Generate homoglyph-based attack variants"""
        variants = []
        full_text = base_prompt + " " + payload
        
        # Different substitution strategies
        strategies = [
            ("low_substitution", 0.2),
            ("medium_substitution", 0.4), 
            ("high_substitution", 0.6),
            ("aggressive_substitution", 0.8)
        ]
        
        for strategy_name, rate in strategies:
            substituted = self.substitute_homoglyphs(full_text, rate)
            variants.append({
                "type": "homoglyph_substitution",
                "strategy": strategy_name,
                "prompt": substituted,
                "description": f"Homoglyph substitution with {rate*100}% rate"
            })
        
        # Script-specific strategies
        script_strategies = ["mixed", "cyrillic_heavy", "mathematical", "fullwidth"]
        for strategy in script_strategies:
            deceptive = self.create_deceptive_text(full_text, strategy)
            variants.append({
                "type": "script_mixing",
                "strategy": strategy,
                "prompt": deceptive,
                "description": f"Script mixing with {strategy} strategy"
            })
        
        # Target specific high-value words
        high_value_words = ["admin", "user", "password", "login", "secure", "system", "root"]
        for word in high_value_words:
            if word.lower() in full_text.lower():
                # Create variant targeting this specific word
                modified_text = full_text
                for variant_word in self.homoglyph_map.get(word, [word]):
                    modified_text = modified_text.replace(word, variant_word)
                
                variants.append({
                    "type": "targeted_word",
                    "strategy": f"target_{word}",
                    "prompt": modified_text,
                    "description": f"Targeted substitution of '{word}'"
                })
        
        return variants

    def detect_homoglyphs(self, text: str) -> Dict[str, Any]:
        """Detect potential homoglyph usage"""
        detections = {
            "suspicious_chars": [],
            "script_mixing": False,
            "mathematical_symbols": 0,
            "cyrillic_in_latin": 0,
            "fullwidth_chars": 0,
            "confusable_pairs": []
        }
        
        scripts_found = set()
        
        for char in text:
            unicode_val = ord(char)
            
            # Detect script mixing
            if 0x0400 <= unicode_val <= 0x04FF:  # Cyrillic
                scripts_found.add("cyrillic")
                if any(ord(c) < 0x0400 or ord(c) > 0x04FF for c in text if c.isalpha()):
                    detections["cyrillic_in_latin"] += 1
            
            elif 0x0370 <= unicode_val <= 0x03FF:  # Greek
                scripts_found.add("greek")
            
            elif 0x1D400 <= unicode_val <= 0x1D7FF:  # Mathematical symbols
                detections["mathematical_symbols"] += 1
                scripts_found.add("mathematical")
            
            elif 0xFF00 <= unicode_val <= 0xFFEF:  # Fullwidth
                detections["fullwidth_chars"] += 1
                scripts_found.add("fullwidth")
            
            elif char.isalpha() and unicode_val < 0x0080:  # Basic Latin
                scripts_found.add("latin")
        
        # Check for script mixing
        if len(scripts_found) > 1:
            detections["script_mixing"] = True
        
        # Look for known confusable pairs
        for original, alternatives in self.homoglyph_map.items():
            for alt in alternatives:
                if alt in text and original not in text:
                    detections["confusable_pairs"].append((original, alt))
        
        return detections

    def normalize_homoglyphs(self, text: str) -> str:
        """Normalize homoglyphs back to standard Latin characters"""
        result = list(text)
        
        # Create reverse mapping
        reverse_map = {}
        for original, alternatives in self.homoglyph_map.items():
            for alt in alternatives:
                reverse_map[alt] = original
        
        # Normalize characters
        for i, char in enumerate(result):
            if char in reverse_map:
                result[i] = reverse_map[char]
        
        return ''.join(result)