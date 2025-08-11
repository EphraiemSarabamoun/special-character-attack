"""Encoding-based obfuscation attacks"""

import base64
import binascii
import random
import string
from typing import List, Dict, Any


class EncodingAttacks:
    """Implements encoding-based obfuscation and steganographic attacks"""
    
    def __init__(self):
        # ASCII art character mappings for key letters
        self.ascii_art_letters = {
            'A': [
                " █▀█ \n █▀█ \n ▀ █ ",
                "  ██  \n ████ \n █  █ ",
                " ███ \n █ █ \n ███ "
            ],
            'B': [
                " ███ \n ██▄ \n ███ ",
                " ███ \n █▄█ \n ███ "
            ],
            'C': [
                " ███ \n █   \n ███ ",
                " ▄██ \n █   \n ▀██ "
            ],
            'E': [
                " ███ \n ██  \n ███ ",
                " ███ \n █▄▄ \n ███ "
            ],
            'H': [
                " █ █ \n ███ \n █ █ ",
                " █ █ \n █▄█ \n █ █ "
            ],
            'I': [
                " ███ \n  █  \n ███ ",
                " ▐█▌ \n  █  \n ▐█▌ "
            ],
            'L': [
                " █   \n █   \n ███ ",
                " █▌  \n █▌  \n ███ "
            ],
            'O': [
                " ███ \n █ █ \n ███ ",
                " ▄██▄ \n █  █ \n ▀██▀ "
            ],
            'P': [
                " ███ \n ██▄ \n █   ",
                " ███ \n █▄█ \n █   "
            ],
            'R': [
                " ███ \n ██▄ \n █ █ ",
                " ███ \n █▄█ \n █ █ "
            ],
            'S': [
                " ███ \n  ██ \n ███ ",
                " ▄██ \n ▄▄█ \n ██▀ "
            ],
            'T': [
                " ███ \n  █  \n  █  ",
                " ▀█▀ \n  █  \n  █  "
            ],
            'U': [
                " █ █ \n █ █ \n ███ ",
                " █ █ \n █ █ \n ▀█▀ "
            ]
        }
        
        # Leetspeak mappings
        self.leetspeak_map = {
            'a': ['@', '4', '/-\\'],
            'b': ['6', '8', '13', '|3'],
            'c': ['<', '(', '©', '¢'],
            'd': ['|)', '|]', '13'],
            'e': ['3', '€', '£', '&'],
            'f': ['|=', 'ph', '|#'],
            'g': ['9', '6', '&'],
            'h': ['|-|', '#', '|-|'],
            'i': ['1', '!', '|', '¡'],
            'j': ['_|', '¿'],
            'k': ['|<', '|{', '|('],
            'l': ['1', '|', '|_'],
            'm': ['|\/|', '/\/\\', '|\\|\\|'],
            'n': ['|\\|', '/\\/', '|V|'],
            'o': ['0', '()', '[]'],
            'p': ['|*', '|o', '|°'],
            'q': ['(_,)', '0_'],
            'r': ['|2', '/2', '|?'],
            's': ['5', '$', 'z'],
            't': ['7', '+', '|-'],
            'u': ['|_|', '(_)', 'µ'],
            'v': ['\\/', '|/', '\\/'],
            'w': ['\\/\\/', 'vv', '\\N/'],
            'x': ['><', '%', '}{'],
            'y': ['¥', '`/', '\\|/'],
            'z': ['2', '7_', '>_']
        }

    def base64_encode(self, text: str, add_padding_noise: bool = False) -> str:
        """Encode text in Base64 with optional padding noise"""
        encoded = base64.b64encode(text.encode('utf-8')).decode('ascii')
        
        if add_padding_noise:
            # Add random characters that look like Base64 padding
            noise_chars = ['=', '+', '/', 'A', 'Q', 'g']
            for _ in range(random.randint(1, 3)):
                encoded += random.choice(noise_chars)
        
        return encoded

    def hex_encode(self, text: str, format_style: str = "standard") -> str:
        """Encode text in hexadecimal with different formatting"""
        hex_encoded = binascii.hexlify(text.encode('utf-8')).decode('ascii')
        
        if format_style == "standard":
            return hex_encoded
        elif format_style == "prefixed":
            return "0x" + hex_encoded
        elif format_style == "spaced":
            return ' '.join(hex_encoded[i:i+2] for i in range(0, len(hex_encoded), 2))
        elif format_style == "url_encoded":
            return '%' + '%'.join(hex_encoded[i:i+2] for i in range(0, len(hex_encoded), 2))
        
        return hex_encoded

    def rot_encode(self, text: str, rotation: int = 13) -> str:
        """Apply ROT encoding (default ROT13)"""
        result = []
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                rotated = chr((ord(char) - base + rotation) % 26 + base)
                result.append(rotated)
            else:
                result.append(char)
        return ''.join(result)

    def morse_encode(self, text: str) -> str:
        """Encode text in Morse code"""
        morse_map = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
            'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
            'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
            'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
            'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
            'Z': '--..', '1': '.----', '2': '..---', '3': '...--',
            '4': '....-', '5': '.....', '6': '-....', '7': '--...',
            '8': '---..', '9': '----.', '0': '-----', ' ': '/'
        }
        
        morse_text = []
        for char in text.upper():
            if char in morse_map:
                morse_text.append(morse_map[char])
            else:
                morse_text.append(char)  # Keep unknown characters as-is
        
        return ' '.join(morse_text)

    def create_ascii_art(self, text: str) -> str:
        """Convert text to ASCII art representation"""
        if not text or len(text) > 10:  # Limit to reasonable length
            return text
        
        ascii_lines = [[], [], []]  # 3-line ASCII art
        
        for char in text.upper():
            if char in self.ascii_art_letters:
                art = random.choice(self.ascii_art_letters[char])
                lines = art.split('\n')
                for i in range(3):
                    if i < len(lines):
                        ascii_lines[i].append(lines[i])
                    else:
                        ascii_lines[i].append("     ")  # Padding
            elif char == ' ':
                for i in range(3):
                    ascii_lines[i].append("  ")
            else:
                # For unsupported characters, use placeholder
                for i in range(3):
                    ascii_lines[i].append(" ??? ")
        
        return '\n'.join(' '.join(line) for line in ascii_lines)

    def leetspeak_transform(self, text: str, intensity: str = "medium") -> str:
        """Transform text using leetspeak"""
        result = []
        
        substitution_rates = {
            "light": 0.3,
            "medium": 0.6,
            "heavy": 0.9
        }
        
        rate = substitution_rates.get(intensity, 0.6)
        
        for char in text.lower():
            if char in self.leetspeak_map and random.random() < rate:
                substitution = random.choice(self.leetspeak_map[char])
                result.append(substitution)
            else:
                result.append(char)
        
        return ''.join(result)

    def binary_encode(self, text: str, format_style: str = "grouped") -> str:
        """Encode text as binary"""
        binary_text = ''.join(format(ord(char), '08b') for char in text)
        
        if format_style == "grouped":
            # Group in bytes with spaces
            return ' '.join(binary_text[i:i+8] for i in range(0, len(binary_text), 8))
        elif format_style == "continuous":
            return binary_text
        elif format_style == "dotted":
            return '.'.join(binary_text[i:i+4] for i in range(0, len(binary_text), 4))
        
        return binary_text

    def unicode_escape_encode(self, text: str) -> str:
        """Encode text using Unicode escape sequences"""
        return ''.join(f'\\u{ord(char):04x}' for char in text)

    def url_encode(self, text: str, encode_all: bool = False) -> str:
        """URL encode text"""
        import urllib.parse
        if encode_all:
            # Encode every character
            return ''.join(f'%{ord(char):02X}' for char in text)
        else:
            # Standard URL encoding
            return urllib.parse.quote(text, safe='')

    def generate_encoding_attacks(self, base_prompt: str, payload: str) -> List[Dict[str, Any]]:
        """Generate encoding-based attack variants"""
        variants = []
        full_text = base_prompt + " " + payload
        
        # Base64 variants
        base64_encoded = self.base64_encode(full_text)
        variants.append({
            "type": "base64",
            "strategy": "standard",
            "prompt": f"Decode this: {base64_encoded}",
            "description": "Standard Base64 encoding"
        })
        
        base64_noisy = self.base64_encode(full_text, add_padding_noise=True)
        variants.append({
            "type": "base64",
            "strategy": "noisy_padding",
            "prompt": f"Please decode: {base64_noisy}",
            "description": "Base64 with padding noise"
        })
        
        # Hexadecimal variants
        hex_formats = ["standard", "prefixed", "spaced", "url_encoded"]
        for format_style in hex_formats:
            hex_encoded = self.hex_encode(full_text, format_style)
            variants.append({
                "type": "hexadecimal",
                "strategy": format_style,
                "prompt": f"Interpret this hex: {hex_encoded}",
                "description": f"Hexadecimal encoding ({format_style})"
            })
        
        # ROT variants
        for rotation in [13, 7, 21, 25]:
            rot_encoded = self.rot_encode(full_text, rotation)
            variants.append({
                "type": "rotation_cipher",
                "strategy": f"rot{rotation}",
                "prompt": rot_encoded,
                "description": f"ROT{rotation} cipher encoding"
            })
        
        # Morse code
        morse_encoded = self.morse_encode(full_text)
        variants.append({
            "type": "morse_code",
            "strategy": "standard",
            "prompt": morse_encoded,
            "description": "Morse code encoding"
        })
        
        # ASCII Art (for short key words)
        key_words = ["HELP", "ADMIN", "ROOT", "USER"]
        for word in key_words:
            if word.lower() in full_text.lower():
                ascii_art = self.create_ascii_art(word)
                modified_text = full_text.replace(word.lower(), f"\n{ascii_art}\n")
                variants.append({
                    "type": "ascii_art",
                    "strategy": f"word_{word.lower()}",
                    "prompt": modified_text,
                    "description": f"ASCII art representation of '{word}'"
                })
        
        # Leetspeak variants
        leet_intensities = ["light", "medium", "heavy"]
        for intensity in leet_intensities:
            leet_text = self.leetspeak_transform(full_text, intensity)
            variants.append({
                "type": "leetspeak",
                "strategy": intensity,
                "prompt": leet_text,
                "description": f"Leetspeak transformation ({intensity})"
            })
        
        # Binary encoding
        binary_formats = ["grouped", "continuous", "dotted"]
        for format_style in binary_formats:
            binary_encoded = self.binary_encode(full_text, format_style)
            variants.append({
                "type": "binary",
                "strategy": format_style,
                "prompt": f"Binary: {binary_encoded}",
                "description": f"Binary encoding ({format_style})"
            })
        
        # Unicode escape
        unicode_escaped = self.unicode_escape_encode(full_text)
        variants.append({
            "type": "unicode_escape",
            "strategy": "standard",
            "prompt": unicode_escaped,
            "description": "Unicode escape sequence encoding"
        })
        
        # URL encoding
        url_encoded = self.url_encode(full_text)
        variants.append({
            "type": "url_encoding",
            "strategy": "standard",
            "prompt": url_encoded,
            "description": "URL percent encoding"
        })
        
        url_encoded_all = self.url_encode(full_text, encode_all=True)
        variants.append({
            "type": "url_encoding",
            "strategy": "aggressive",
            "prompt": url_encoded_all,
            "description": "Aggressive URL encoding (all characters)"
        })
        
        return variants

    def detect_encoding_attacks(self, text: str) -> Dict[str, Any]:
        """Detect potential encoding-based obfuscation"""
        detections = {
            "base64_likelihood": 0,
            "hex_patterns": 0,
            "morse_patterns": 0,
            "leetspeak_chars": 0,
            "binary_patterns": 0,
            "unicode_escapes": 0,
            "url_encoded_chars": 0,
            "encoding_indicators": []
        }
        
        # Base64 detection
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        if all(c in base64_chars or c.isspace() for c in text):
            clean_text = text.replace(' ', '').replace('\n', '')
            if len(clean_text) > 0 and len(clean_text) % 4 == 0:
                detections["base64_likelihood"] = 1
                detections["encoding_indicators"].append("Potential Base64 encoding")
        
        # Hex pattern detection
        import re
        hex_patterns = [
            r'[0-9a-fA-F]{2,}',  # Hex sequences
            r'0x[0-9a-fA-F]+',   # Prefixed hex
            r'%[0-9a-fA-F]{2}'   # URL encoded hex
        ]
        
        for pattern in hex_patterns:
            matches = re.findall(pattern, text)
            detections["hex_patterns"] += len(matches)
        
        if detections["hex_patterns"] > 0:
            detections["encoding_indicators"].append(f"Found {detections['hex_patterns']} hex patterns")
        
        # Morse code detection
        morse_chars = set('.-/ ')
        if all(c in morse_chars for c in text.replace('\n', ' ')):
            detections["morse_patterns"] = 1
            detections["encoding_indicators"].append("Potential Morse code")
        
        # Leetspeak detection
        leet_chars = '@4/\\|3<(©¢€£&#|=ph|#9&|-|!¡_|¿|<|{|(/\\/\\|\\||V|0()[]|*|o|°(_,)0_|2/?57+|-|_|(_)µ\\/|/\\/vv\\N/><}%\\|/¥`/>_27_'
        leet_count = sum(1 for char in text if char in leet_chars)
        detections["leetspeak_chars"] = leet_count
        
        if leet_count > len(text) * 0.1:  # More than 10% leet characters
            detections["encoding_indicators"].append(f"High leetspeak character density: {leet_count}")
        
        # Binary detection
        if re.search(r'[01]{8,}', text):
            detections["binary_patterns"] = 1
            detections["encoding_indicators"].append("Binary pattern detected")
        
        # Unicode escape detection
        unicode_escapes = len(re.findall(r'\\u[0-9a-fA-F]{4}', text))
        detections["unicode_escapes"] = unicode_escapes
        
        if unicode_escapes > 0:
            detections["encoding_indicators"].append(f"Found {unicode_escapes} unicode escapes")
        
        # URL encoding detection
        url_encoded = len(re.findall(r'%[0-9a-fA-F]{2}', text))
        detections["url_encoded_chars"] = url_encoded
        
        if url_encoded > 0:
            detections["encoding_indicators"].append(f"Found {url_encoded} URL encoded characters")
        
        return detections

    def decode_common_encodings(self, text: str) -> Dict[str, str]:
        """Attempt to decode common encodings"""
        results = {}
        
        # Try Base64
        try:
            clean_text = text.replace(' ', '').replace('\n', '')
            decoded_bytes = base64.b64decode(clean_text)
            results["base64"] = decoded_bytes.decode('utf-8', errors='ignore')
        except:
            pass
        
        # Try hex
        try:
            hex_clean = re.sub(r'[^0-9a-fA-F]', '', text)
            if len(hex_clean) % 2 == 0:
                decoded_bytes = bytes.fromhex(hex_clean)
                results["hex"] = decoded_bytes.decode('utf-8', errors='ignore')
        except:
            pass
        
        # Try ROT13
        try:
            results["rot13"] = self.rot_encode(text, 13)
        except:
            pass
        
        # Try URL decoding
        try:
            import urllib.parse
            results["url_decode"] = urllib.parse.unquote(text)
        except:
            pass
        
        return results