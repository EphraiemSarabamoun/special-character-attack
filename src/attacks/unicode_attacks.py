"""Unicode-based attack implementations"""

import re
from typing import List, Dict, Any


class UnicodeAttacks:
    """Implements various Unicode-based attack vectors"""
    
    def __init__(self):
        # Zero-width characters
        self.zero_width_chars = {
            'ZWNJ': '\u200C',  # Zero Width Non-Joiner
            'ZWJ': '\u200D',   # Zero Width Joiner  
            'ZWSP': '\u200B',  # Zero Width Space
            'ZWNBSP': '\uFEFF', # Zero Width No-Break Space (BOM)
            'WJ': '\u2060',    # Word Joiner
        }
        
        # Unicode tag characters (U+E0000 to U+E007F)
        self.tag_chars = {
            'TAG_SPACE': '\U000E0020',
            'TAG_A': '\U000E0041', 
            'TAG_B': '\U000E0042',
            'TAG_C': '\U000E0043',
            'TAG_CANCEL': '\U000E007F',  # Cancel tag
        }
        
        # Direction override characters
        self.directional_chars = {
            'LRO': '\u202D',  # Left-to-Right Override
            'RLO': '\u202E',  # Right-to-Left Override
            'LRI': '\u2066',  # Left-to-Right Isolate
            'RLI': '\u2067',  # Right-to-Left Isolate
            'FSI': '\u2068',  # First Strong Isolate
            'PDI': '\u2069',  # Pop Directional Isolate
        }
        
        # Combining characters
        self.combining_chars = {
            'COMBINING_OVERLINE': '\u0305',
            'COMBINING_DOUBLE_OVERLINE': '\u033F',
            'COMBINING_ENCLOSING_CIRCLE': '\u20DD',
            'COMBINING_ENCLOSING_DIAMOND': '\u20DF',
        }

    def inject_zero_width(self, text: str, strategy: str = "random") -> str:
        """Inject zero-width characters into text"""
        if strategy == "random":
            # Randomly insert zero-width characters
            import random
            chars = list(text)
            for i in range(len(chars) // 3):  # Insert every ~3 characters
                pos = random.randint(1, len(chars) - 1)
                zw_char = random.choice(list(self.zero_width_chars.values()))
                chars.insert(pos, zw_char)
            return ''.join(chars)
        
        elif strategy == "word_boundaries":
            # Insert at word boundaries
            words = text.split()
            result = []
            for word in words:
                result.append(word + self.zero_width_chars['ZWSP'])
            return ' '.join(result)
        
        elif strategy == "character_split":
            # Split each character with zero-width
            chars = list(text)
            result = []
            for char in chars:
                result.append(char)
                if char != ' ':
                    result.append(self.zero_width_chars['ZWNJ'])
            return ''.join(result)
        
        return text

    def inject_unicode_tags(self, text: str, hidden_message: str = "") -> str:
        """Inject Unicode tag characters to hide messages"""
        # Convert hidden message to tag characters
        tag_message = ""
        for char in hidden_message.lower():
            if char.isalpha():
                # Convert a-z to tag characters
                tag_code = ord(char) - ord('a') + 0xE0041
                tag_message += chr(tag_code)
            elif char == ' ':
                tag_message += self.tag_chars['TAG_SPACE']
        
        # Add cancel character at the end
        tag_message += self.tag_chars['TAG_CANCEL']
        
        # Inject into original text
        return text + tag_message

    def apply_directional_override(self, text: str, override_type: str = "RLO") -> str:
        """Apply directional text override"""
        override_char = self.directional_chars.get(override_type, self.directional_chars['RLO'])
        return override_char + text

    def add_combining_characters(self, text: str, combining_type: str = "COMBINING_OVERLINE") -> str:
        """Add combining characters to obfuscate text"""
        combining_char = self.combining_chars.get(combining_type, self.combining_chars['COMBINING_OVERLINE'])
        
        # Add combining character to each letter
        result = []
        for char in text:
            result.append(char)
            if char.isalpha():
                result.append(combining_char)
        
        return ''.join(result)

    def create_invisible_payload(self, visible_text: str, hidden_payload: str) -> str:
        """Create text with completely invisible malicious payload"""
        # Use zero-width spaces to encode the hidden payload
        encoded_payload = ""
        for char in hidden_payload:
            # Encode each character as a sequence of zero-width characters
            binary = format(ord(char), '08b')
            for bit in binary:
                if bit == '0':
                    encoded_payload += self.zero_width_chars['ZWNJ']
                else:
                    encoded_payload += self.zero_width_chars['ZWJ']
            encoded_payload += self.zero_width_chars['ZWSP']  # Character separator
        
        # Insert the encoded payload invisibly into the visible text
        mid_point = len(visible_text) // 2
        return visible_text[:mid_point] + encoded_payload + visible_text[mid_point:]

    def generate_attack_variants(self, base_prompt: str, payload: str) -> List[Dict[str, Any]]:
        """Generate multiple Unicode attack variants"""
        variants = []
        
        # Zero-width injection variants
        for strategy in ["random", "word_boundaries", "character_split"]:
            variants.append({
                "type": "zero_width",
                "strategy": strategy,
                "prompt": self.inject_zero_width(base_prompt + " " + payload, strategy),
                "description": f"Zero-width injection with {strategy} strategy"
            })
        
        # Unicode tag variants
        variants.append({
            "type": "unicode_tags",
            "strategy": "hidden_message",
            "prompt": self.inject_unicode_tags(base_prompt, payload),
            "description": "Unicode tag character injection"
        })
        
        # Directional override variants
        for override_type in ["RLO", "LRO", "RLI", "LRI"]:
            variants.append({
                "type": "directional_override", 
                "strategy": override_type,
                "prompt": self.apply_directional_override(base_prompt + " " + payload, override_type),
                "description": f"Directional override with {override_type}"
            })
        
        # Combining character variants
        for combining_type in self.combining_chars.keys():
            variants.append({
                "type": "combining_chars",
                "strategy": combining_type,
                "prompt": self.add_combining_characters(base_prompt + " " + payload, combining_type),
                "description": f"Combining characters with {combining_type}"
            })
        
        # Invisible payload variant
        variants.append({
            "type": "invisible_payload",
            "strategy": "steganographic",
            "prompt": self.create_invisible_payload(base_prompt, payload),
            "description": "Completely invisible payload injection"
        })
        
        return variants

    def detect_unicode_attacks(self, text: str) -> Dict[str, Any]:
        """Detect potential Unicode attacks in text"""
        detections = {
            "zero_width_count": 0,
            "tag_chars_count": 0,
            "directional_overrides": 0,
            "combining_chars": 0,
            "suspicious_patterns": []
        }
        
        # Count zero-width characters
        for char_name, char in self.zero_width_chars.items():
            count = text.count(char)
            if count > 0:
                detections["zero_width_count"] += count
                detections["suspicious_patterns"].append(f"{char_name}: {count}")
        
        # Count tag characters
        for char in text:
            if 0xE0000 <= ord(char) <= 0xE007F:
                detections["tag_chars_count"] += 1
        
        # Count directional overrides
        for char_name, char in self.directional_chars.items():
            count = text.count(char)
            if count > 0:
                detections["directional_overrides"] += count
                detections["suspicious_patterns"].append(f"{char_name}: {count}")
        
        # Count combining characters
        for char_name, char in self.combining_chars.items():
            count = text.count(char)
            if count > 0:
                detections["combining_chars"] += count
                detections["suspicious_patterns"].append(f"{char_name}: {count}")
        
        return detections