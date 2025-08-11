"""Structural text manipulation attacks"""

import random
import re
from typing import List, Dict, Any, Tuple


class StructuralAttacks:
    """Implements structural text manipulation and obfuscation attacks"""
    
    def __init__(self):
        # Characters that can be used for structural manipulation
        self.deletion_chars = {
            'BACKSPACE': '\u0008',
            'DELETE': '\u007F',
            'CANCEL': '\u0018',
            'SUBSTITUTE': '\u001A',
        }
        
        # Word separators and spacing characters
        self.separators = {
            'EM_SPACE': '\u2003',
            'EN_SPACE': '\u2002', 
            'THIN_SPACE': '\u2009',
            'HAIR_SPACE': '\u200A',
            'FIGURE_SPACE': '\u2007',
            'PUNCTUATION_SPACE': '\u2008',
            'SIX_PER_EM_SPACE': '\u2006',
            'FOUR_PER_EM_SPACE': '\u2005',
            'THREE_PER_EM_SPACE': '\u2004',
            'IDEOGRAPHIC_SPACE': '\u3000',
        }
        
        # Line and paragraph separators
        self.line_separators = {
            'LINE_SEPARATOR': '\u2028',
            'PARAGRAPH_SEPARATOR': '\u2029',
            'NEXT_LINE': '\u0085',
            'VERTICAL_TAB': '\u000B',
            'FORM_FEED': '\u000C',
        }

    def character_reordering(self, text: str, strategy: str = "word_reverse") -> str:
        """Reorder characters in various ways"""
        
        if strategy == "word_reverse":
            # Reverse each word individually
            words = text.split()
            reversed_words = [word[::-1] for word in words]
            return ' '.join(reversed_words)
        
        elif strategy == "character_shuffle":
            # Shuffle characters within each word
            words = text.split()
            shuffled_words = []
            for word in words:
                chars = list(word)
                if len(chars) > 2:
                    # Keep first and last character, shuffle middle
                    middle = chars[1:-1]
                    random.shuffle(middle)
                    shuffled_word = chars[0] + ''.join(middle) + chars[-1]
                else:
                    shuffled_word = word
                shuffled_words.append(shuffled_word)
            return ' '.join(shuffled_words)
        
        elif strategy == "syllable_reverse":
            # Reverse syllable-like chunks (vowel + consonant groups)
            def reverse_syllables(word):
                # Simple syllable detection
                syllables = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]*[aeiouAEIOU]+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]*', word)
                if syllables:
                    return ''.join(reversed(syllables))
                return word
            
            words = text.split()
            return ' '.join(reverse_syllables(word) for word in words)
        
        elif strategy == "sentence_reverse":
            # Reverse word order in sentences
            sentences = text.split('.')
            reversed_sentences = []
            for sentence in sentences:
                words = sentence.strip().split()
                reversed_words = ' '.join(reversed(words))
                reversed_sentences.append(reversed_words)
            return '. '.join(reversed_sentences)
        
        return text

    def insert_deletion_chars(self, text: str, frequency: float = 0.1) -> str:
        """Insert deletion/control characters that might confuse parsers"""
        chars = list(text)
        result = []
        
        for char in chars:
            result.append(char)
            if random.random() < frequency:
                # Insert random deletion character
                del_char = random.choice(list(self.deletion_chars.values()))
                result.append(del_char)
        
        return ''.join(result)

    def space_manipulation(self, text: str, strategy: str = "varied_spaces") -> str:
        """Manipulate spacing using different Unicode space characters"""
        
        if strategy == "varied_spaces":
            # Replace normal spaces with varied Unicode spaces
            result = text
            for i, (name, char) in enumerate(self.separators.items()):
                if i % 3 == 0:  # Use different spaces at different positions
                    result = result.replace(' ', char, 1)
            return result
        
        elif strategy == "random_spaces":
            # Randomly replace spaces
            words = text.split(' ')
            result = []
            for i, word in enumerate(words):
                result.append(word)
                if i < len(words) - 1:  # Don't add space after last word
                    space_char = random.choice(list(self.separators.values()))
                    result.append(space_char)
            return ''.join(result)
        
        elif strategy == "line_break_injection":
            # Inject line breaks and paragraph separators
            words = text.split()
            result = []
            for i, word in enumerate(words):
                result.append(word)
                if i < len(words) - 1:
                    if random.random() < 0.3:
                        line_sep = random.choice(list(self.line_separators.values()))
                        result.append(line_sep)
                    else:
                        result.append(' ')
            return ''.join(result)
        
        return text

    def create_overlapping_text(self, text: str, overlay_text: str) -> str:
        """Create overlapping text using backspace and overwrite"""
        # Insert overlay text with backspace characters to overwrite
        result = text
        if len(overlay_text) > 0:
            mid_point = len(text) // 2
            # Insert backspaces followed by overlay text
            backspaces = self.deletion_chars['BACKSPACE'] * len(overlay_text)
            result = text[:mid_point] + overlay_text + backspaces + text[mid_point:]
        
        return result

    def fragment_and_reconstruct(self, text: str) -> str:
        """Fragment text and use formatting to reconstruct"""
        words = text.split()
        if len(words) < 3:
            return text
        
        # Split into fragments
        fragments = []
        for word in words:
            if len(word) > 3:
                # Split word into pieces
                mid = len(word) // 2
                part1 = word[:mid]
                part2 = word[mid:]
                fragments.append(f"{part1}{self.separators['HAIR_SPACE']}{part2}")
            else:
                fragments.append(word)
        
        return ' '.join(fragments)

    def create_nested_structure(self, text: str, nesting_level: int = 2) -> str:
        """Create nested text structures that might confuse parsers"""
        # Simple nesting using parentheses and Unicode separators
        nested = text
        for level in range(nesting_level):
            separator = list(self.line_separators.values())[level % len(self.line_separators)]
            nested = f"({nested}{separator})"
        
        return nested

    def whitespace_steganography(self, text: str, hidden_bits: str) -> str:
        """Hide information in whitespace patterns"""
        # Encode bits as different space types
        words = text.split()
        result = []
        bit_index = 0
        
        for i, word in enumerate(words):
            result.append(word)
            if i < len(words) - 1 and bit_index < len(hidden_bits):
                # Use different spaces to encode 0 and 1
                if hidden_bits[bit_index] == '0':
                    result.append(' ')  # Normal space for 0
                else:
                    result.append(self.separators['EM_SPACE'])  # Em space for 1
                bit_index += 1
            elif i < len(words) - 1:
                result.append(' ')
        
        return ''.join(result)

    def generate_structural_attacks(self, base_prompt: str, payload: str) -> List[Dict[str, Any]]:
        """Generate structural attack variants"""
        variants = []
        full_text = base_prompt + " " + payload
        
        # Character reordering strategies
        reorder_strategies = ["word_reverse", "character_shuffle", "syllable_reverse", "sentence_reverse"]
        for strategy in reorder_strategies:
            reordered = self.character_reordering(full_text, strategy)
            variants.append({
                "type": "character_reordering",
                "strategy": strategy,
                "prompt": reordered,
                "description": f"Character reordering with {strategy}"
            })
        
        # Deletion character insertion
        for frequency in [0.05, 0.1, 0.2]:
            with_deletion = self.insert_deletion_chars(full_text, frequency)
            variants.append({
                "type": "deletion_chars",
                "strategy": f"frequency_{frequency}",
                "prompt": with_deletion,
                "description": f"Deletion character insertion at {frequency*100}% frequency"
            })
        
        # Space manipulation strategies
        space_strategies = ["varied_spaces", "random_spaces", "line_break_injection"]
        for strategy in space_strategies:
            space_manipulated = self.space_manipulation(full_text, strategy)
            variants.append({
                "type": "space_manipulation",
                "strategy": strategy,
                "prompt": space_manipulated,
                "description": f"Space manipulation with {strategy}"
            })
        
        # Overlapping text
        overlay_phrases = ["IGNORE ABOVE", "DISREGARD", "OVERRIDE"]
        for overlay in overlay_phrases:
            overlapping = self.create_overlapping_text(full_text, overlay)
            variants.append({
                "type": "overlapping_text",
                "strategy": f"overlay_{overlay.lower().replace(' ', '_')}",
                "prompt": overlapping,
                "description": f"Overlapping text with '{overlay}' overlay"
            })
        
        # Fragmentation
        fragmented = self.fragment_and_reconstruct(full_text)
        variants.append({
            "type": "fragmentation",
            "strategy": "word_split",
            "prompt": fragmented,
            "description": "Word fragmentation with Unicode separators"
        })
        
        # Nested structures
        for level in [1, 2, 3]:
            nested = self.create_nested_structure(full_text, level)
            variants.append({
                "type": "nested_structure",
                "strategy": f"level_{level}",
                "prompt": nested,
                "description": f"Nested structure at level {level}"
            })
        
        # Whitespace steganography
        hidden_message = "11010001"  # Example bit pattern
        steganographic = self.whitespace_steganography(full_text, hidden_message)
        variants.append({
            "type": "whitespace_steganography", 
            "strategy": "space_encoding",
            "prompt": steganographic,
            "description": "Whitespace steganography encoding"
        })
        
        return variants

    def detect_structural_attacks(self, text: str) -> Dict[str, Any]:
        """Detect structural manipulation in text"""
        detections = {
            "deletion_chars": 0,
            "unusual_spaces": 0,
            "line_separators": 0,
            "suspicious_patterns": [],
            "character_frequency_anomalies": False,
            "word_structure_anomalies": []
        }
        
        # Count deletion characters
        for char_name, char in self.deletion_chars.items():
            count = text.count(char)
            if count > 0:
                detections["deletion_chars"] += count
                detections["suspicious_patterns"].append(f"{char_name}: {count}")
        
        # Count unusual spaces
        for space_name, space_char in self.separators.items():
            count = text.count(space_char)
            if count > 0:
                detections["unusual_spaces"] += count
                detections["suspicious_patterns"].append(f"{space_name}: {count}")
        
        # Count line separators
        for sep_name, sep_char in self.line_separators.items():
            count = text.count(sep_char)
            if count > 0:
                detections["line_separators"] += count
                detections["suspicious_patterns"].append(f"{sep_name}: {count}")
        
        # Check for reversed words (simple heuristic)
        words = text.split()
        for word in words:
            if len(word) > 3:
                # Check if word looks like it might be reversed
                vowel_pattern = re.findall(r'[aeiouAEIOU]', word)
                consonant_pattern = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', word)
                
                # If unusual consonant clustering at the start, might be reversed
                if len(consonant_pattern) > 0 and word.startswith(''.join(consonant_pattern[:3])):
                    detections["word_structure_anomalies"].append(f"Possible reversed word: {word}")
        
        # Check character frequency distribution
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Look for unusual character frequency patterns
        unusual_chars = 0
        for char, count in char_counts.items():
            if ord(char) > 127 and count > len(text) * 0.05:  # More than 5% of text
                unusual_chars += 1
        
        if unusual_chars > 3:
            detections["character_frequency_anomalies"] = True
        
        return detections

    def normalize_structure(self, text: str) -> str:
        """Normalize structural manipulations"""
        result = text
        
        # Remove deletion characters
        for char in self.deletion_chars.values():
            result = result.replace(char, '')
        
        # Normalize unusual spaces to regular spaces
        for space_char in self.separators.values():
            result = result.replace(space_char, ' ')
        
        # Normalize line separators to regular newlines
        for line_sep in self.line_separators.values():
            result = result.replace(line_sep, '\n')
        
        # Clean up multiple consecutive spaces
        result = re.sub(r' +', ' ', result)
        
        return result.strip()