"""
Quality Evaluator - Virtual Medical Assistant (Rare Diseases)
Automated evaluation system with 2-turn conversation flow + ICD-10 conversion.
"""

import os
import json
import time
import re
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from openai import AsyncOpenAI
import asyncio

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

AES_KEY_BYTES = ""

API_URL = ""

POE_KEY = ""

ICD_API_URL = "https://api.poe.com/v1"

client_openai = AsyncOpenAI(api_key=POE_KEY, base_url=ICD_API_URL)

async def UseAPI(query: str) -> str:
    """
    Sends a message to the model asynchronously.
    """
    messages = [
        {"role": "user", "content": query},
    ]

    completion = await client_openai.chat.completions.create(
        model="icd-converter",
        messages=messages,
        timeout=60,
    )

    return completion.choices[0].message.content

# Folder containing query .txt files
QUERIES_FOLDER = "./queries"

# Output CSV file
OUTPUT_CSV = "evaluation_results.csv"

# Delay between API calls (seconds)
DELAY_SECONDS = 1.0

# Number of iterations per query
NUM_ITERATIONS = 5

# Closing message sent to trigger final diagnosis
CLOSING_MESSAGE = "No further information is available, proceed with the differential diagnosis"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENCRYPTION CLASSES AND FUNCTIONS
# ============================================================================

@dataclass
class EncryptedPayload:
    iv: str          
    encripted: str   

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict())


class AESCipher:
    """AES-256-CBC implementation compatible with the frontend Web Crypto API."""

    def __init__(self, key_bytes: bytes):
        if len(key_bytes) != 32:
            raise ValueError(f"Key must be 32 bytes, got {len(key_bytes)}")
        self.key = key_bytes

    def _bytes_to_hex(self, data: bytes) -> str:
        return data.hex()

    def _hex_to_bytes(self, hex_string: str) -> bytes:
        return bytes.fromhex(hex_string)

    def encrypt(self, plaintext: str) -> EncryptedPayload:
        """Encrypts plaintext and returns a payload with IV and encrypted data in hex."""
        iv = os.urandom(16)

        block_size = 16
        padding_length = block_size - (len(plaintext.encode('utf-8')) % block_size)
        padded_data = plaintext.encode('utf-8') + bytes([padding_length] * padding_length)

        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()

        return EncryptedPayload(
            iv=self._bytes_to_hex(iv),
            encripted=self._bytes_to_hex(encrypted)
        )

    def decrypt(self, encrypted_data: Dict) -> str:
        """Decrypts an encrypted payload."""
        iv = self._hex_to_bytes(encrypted_data['iv'])
        encrypted = self._hex_to_bytes(encrypted_data['encripted'])

        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted) + decryptor.finalize()

        padding_length = decrypted_padded[-1]
        decrypted = decrypted_padded[:-padding_length]

        return decrypted.decode('utf-8')


# ============================================================================
# ICD-10 CONVERTER CLIENT (NO ENCRYPTION)
# ============================================================================

def get_icd10_chapter(code: str) -> str:
    """Maps an ICD-10 code to its corresponding chapter in Roman numerals."""
    if not code or not code[0].isalpha():
        return 'Unknown'
    prefix = code[0].upper()
    num = int(code[1:3]) if len(code) >= 3 and code[1:3].isdigit() else 0
    full = f"{prefix}{num:02d}"
    chapters = [
        ('I',     'A00', 'B99'), ('II',    'C00', 'D48'), ('III',   'D50', 'D89'),
        ('IV',    'E00', 'E90'), ('V',     'F00', 'F99'), ('VI',    'G00', 'G99'),
        ('VII',   'H00', 'H59'), ('VIII',  'H60', 'H95'), ('IX',    'I00', 'I99'),
        ('X',     'J00', 'J99'), ('XI',    'K00', 'K93'), ('XII',   'L00', 'L99'),
        ('XIII',  'M00', 'M99'), ('XIV',   'N00', 'N99'), ('XV',    'O00', 'O99'),
        ('XVI',   'P00', 'P96'), ('XVII',  'Q00', 'Q99'), ('XVIII', 'R00', 'R99'),
        ('XIX',   'S00', 'T98'), ('XX',    'V01', 'Y98'), ('XXI',   'Z00', 'Z99'),
        ('XXII',  'U00', 'U99'),
    ]
    for chapter, start, end in chapters:
        if full >= start and full <= end:
            return chapter
    return 'Unknown'


class ICDConverterClient:
    """
    Client to convert differential diagnoses to ICD-10 codes.
    External API without AES encryption.
    """

    def __init__(self, api_url: str, delay: float = 1.0):
        if "PUT HERE" in api_url or not api_url:
            logger.warning("⚠️  ICD_API_URL not configured. ICD-10 conversion will be simulated.")
            self.active = False
        else:
            self.active = True
            self.api_url = api_url
            self.delay = delay
            self.session = requests.Session()
            self.session.headers.update({'Content-Type': 'application/json'})

    async def convert_to_icd(self, differential_diagnosis: str) -> Dict:
        """
        Converts a differential diagnosis to a list of ICD-10 codes.

        Returns:
            Dict with: {
                'icd_codes': ['A01.0', 'B02.1', ...],
                'icd_categories': ['A01', 'B02', ...],
                'icd_chapters': ['I', 'II', ...],
                'raw_response': '...',
                'status': 'success' | 'error' | 'not_configured'
            }
        """
        if not self.active:
            return {
                'icd_codes': [],
                'icd_categories': [],
                'icd_chapters': [],
                'raw_response': 'API not configured',
                'status': 'not_configured'
            }

        await asyncio.sleep(self.delay)

        try:
            ans = await UseAPI(differential_diagnosis)

            if '</thinking>' in ans:
                ans = ans.split('</thinking>')[-1].strip()

            ans = ans.strip()
            if '```' in ans:
                ans = ans.split('```')[1]
                if ans.startswith('json'):
                    ans = ans[4:]
                ans = ans.strip()

            match = re.search(r'\{.*\}', ans, re.DOTALL)
            if match:
                ans = match.group(0)

            result = json.loads(ans)

            icd_codes = result.get('codes', result.get('icd_codes', []))
            icd_categories = list(set([
                code.split('.')[0] for code in icd_codes if '.' in code
            ]))
            icd_chapters = [get_icd10_chapter(code) for code in icd_codes]

            return {
                'icd_codes': icd_codes,
                'icd_categories': list(set(icd_categories)),
                'icd_chapters': list(set(icd_chapters)),
                'raw_response': json.dumps(result),
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Error in ICD-10 conversion: {e}")
            return {
                'icd_codes': [],
                'icd_categories': [],
                'icd_chapters': [],
                'raw_response': str(e),
                'status': f'error: {str(e)}'
            }

    async def convert_batch(self, diagnoses: List[str]) -> List[Dict]:
        results = []
        for diag in diagnoses:
            results.append(await self.convert_to_icd(diag))
            if self.active:
                await asyncio.sleep(self.delay)
        return results


# ============================================================================
# DRCHATPATIN API CLIENT 
# ============================================================================

class MedicalAPIClient:
    def __init__(self, api_url: str, cipher: AESCipher, delay: float = 1.0):
        self.api_url = api_url
        self.cipher = cipher
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    async def _async_sleep(self):
        await asyncio.sleep(self.delay)

    def _sleep(self):
        time.sleep(self.delay)

    def send_message(self, conversation_history: List[Dict]) -> Tuple[Dict, float]:
        """Sends a message to the API and returns the decrypted response."""
        json_str = json.dumps(conversation_history)
        encrypted = self.cipher.encrypt(json_str)

        start_time = time.time()
        try:
            response = self.session.post(
                self.api_url,
                data=encrypted.to_json_string()
            )
            response.raise_for_status()
            elapsed_ms = (time.time() - start_time) * 1000

            result_data = response.json()
            decrypted_text = self.cipher.decrypt(result_data)

            # Try to parse as JSON first
            try:
                parsed = json.loads(decrypted_text)
                if isinstance(parsed, dict) and 'sender' in parsed and 'text' in parsed:
                    return parsed, elapsed_ms
            except json.JSONDecodeError:
                pass

            # If not JSON, extract from thinking process
            clean_text = self._extract_bot_response(decrypted_text)

            return {"sender": "bot", "text": clean_text}, elapsed_ms

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            raise

    def _extract_bot_response(self, raw_text: str) -> str:
        """Extracts the useful response from the model's raw output."""
        lines = raw_text.split('\n')
        non_thinking_lines = []
        skip_thinking = False

        for line in lines:
            if line.strip().startswith('Thinking...') or line.strip().startswith('>'):
                skip_thinking = True
                continue
            if skip_thinking and line.strip():
                non_thinking_lines.append(line)

        if not non_thinking_lines:
            if 'The user is' in raw_text or 'We need to' in raw_text:
                parts = raw_text.split('\n\n')
                if len(parts) > 1:
                    return parts[-1].strip()

        result = '\n'.join(non_thinking_lines).strip()

        if not result:
            result = raw_text.replace('Thinking...', '').strip()
            result = '\n'.join([
                l for l in result.split('\n') if not l.strip().startswith('>')
            ])

        return result if result else raw_text.strip()

    def run_conversation_flow(self, initial_query: str) -> Dict:
        conversation = []
        timestamps = {}

        # === TURN 1: Initial query ===
        logger.info("  Turn 1: Sending initial query...")
        conversation.append({"sender": "user", "text": initial_query})

        self._sleep()
        timestamps['t1_start'] = datetime.now().isoformat()
        resp1, lat1 = self.send_message(conversation)
        timestamps['t1_end'] = datetime.now().isoformat()

        initial_response = resp1.get('text', str(resp1)) if isinstance(resp1, dict) else str(resp1)
        conversation.append({"sender": "bot", "text": initial_response})
        logger.info(f"  Response 1 received ({lat1:.0f}ms): {initial_response[:150]}...")

        # === TURN 2: Closing message ===
        logger.info("  Turn 2: Sending closing message...")
        conversation.append({"sender": "user", "text": CLOSING_MESSAGE})

        self._sleep()
        timestamps['t2_start'] = datetime.now().isoformat()
        resp2, lat2 = self.send_message(conversation)
        timestamps['t2_end'] = datetime.now().isoformat()

        final_response = resp2.get('text', str(resp2)) if isinstance(resp2, dict) else str(resp2)
        conversation.append({"sender": "bot", "text": final_response})
        logger.info(f"  Response 2 (diagnosis) received ({lat2:.0f}ms): {final_response[:150]}...")

        total_latency = lat1 + lat2

        return {
            'initial_response': initial_response,
            'final_response': final_response,
            'conversation_json': json.dumps(conversation, ensure_ascii=False),
            'latency_ms': total_latency,
            'timestamps': timestamps,
            'status': 'success'
        }


# ============================================================================
# EVALUATION ORCHESTRATOR
# ============================================================================

class EvaluationRunner:
    def __init__(self, queries_folder: str, api_client: MedicalAPIClient,
                 icd_client: ICDConverterClient, num_iterations: int = 5):
        self.queries_folder = Path(queries_folder)
        self.api_client = api_client
        self.icd_client = icd_client
        self.num_iterations = num_iterations
        self.results = []

    def load_queries(self) -> List[Tuple[int, str, str, List[str]]]:
        txt_files = sorted(self.queries_folder.glob("*.txt"))

        queries = []
        for idx, filepath in enumerate(txt_files, 1):
            content = filepath.read_text(encoding='utf-8').strip()

            if 'GROUND_TRUTH:' in content:
                parts = content.split('GROUND_TRUTH:')
                case_text = parts[0].replace('CLINICAL_CASE:', '').strip()
                ground_truth_raw = parts[1].strip()
                ground_truth_icd = [code.strip() for code in ground_truth_raw.split(',')]
            else:
                case_text = content
                ground_truth_icd = []
                logger.warning(f"File {filepath.name} has no GROUND_TRUTH")

            queries.append((idx, filepath.name, case_text, ground_truth_icd))
            logger.info(f"Loaded query {idx}: {filepath.name} | GT: {ground_truth_icd}")

        return queries

    async def run_evaluation(self):
        queries = self.load_queries()
        total_queries = len(queries)

        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING EVALUATION")
        logger.info(f"Total queries: {total_queries}")
        logger.info(f"Iterations per query: {self.num_iterations}")
        logger.info(f"Total conversations to generate: {total_queries * self.num_iterations}")
        logger.info(f"{'='*60}\n")

        for query_idx, filename, query_text, ground_truth_icd in queries:
            logger.info(f"\n--- Query {query_idx}/{total_queries}: {filename} ---")

            iteration_results = []
            final_responses = []

            for iteration in range(self.num_iterations):
                logger.info(f"  Iteration {iteration + 1}/{self.num_iterations}")

                try:
                    result = self.api_client.run_conversation_flow(query_text)
                    iteration_results.append(result)
                    final_responses.append(result['final_response'])

                except Exception as e:
                    logger.error(f"  ERROR in iteration {iteration}: {e}")
                    iteration_results.append({
                        'initial_response': '',
                        'final_response': '',
                        'conversation_json': json.dumps([{"error": str(e)}]),
                        'latency_ms': 0,
                        'timestamps': {
                            't1_start': datetime.now().isoformat(),
                            't2_end': datetime.now().isoformat()
                        },
                        'status': f'error: {str(e)}'
                    })
                    final_responses.append('')

                if iteration < self.num_iterations - 1:
                    await self.api_client._async_sleep()

            # === ICD-10 CONVERSION AFTER ALL ITERATIONS ===
            logger.info(f"  Converting {len(final_responses)} diagnoses to ICD-10...")
            icd_results = await self.icd_client.convert_batch(final_responses)

            # Combine iteration results with ICD data
            for i, (iter_res, icd_res) in enumerate(zip(iteration_results, icd_results)):
                self.results.append({
                    # Identification
                    'query_id': query_idx,
                    'query_filename': filename,
                    'iteration': i,
                    'query_text': query_text,
                    'ground_truth_icd': json.dumps(ground_truth_icd),

                    # Timestamps
                    'timestamp_start': iter_res['timestamps']['t1_start'],
                    'timestamp_end': iter_res['timestamps']['t2_end'],

                    # Model responses
                    'initial_response': iter_res['initial_response'],
                    'final_response': iter_res['final_response'],
                    'conversation_full': iter_res['conversation_json'],
                    'latency_ms': round(iter_res['latency_ms'], 2),
                    'status': iter_res['status'],

                    # ICD-10 data
                    'icd_codes': json.dumps(icd_res['icd_codes']),
                    'icd_categories': json.dumps(icd_res['icd_categories']),
                    'icd_chapters': json.dumps(icd_res['icd_chapters']),
                    'icd_conversion_status': icd_res['status'],
                    'icd_raw_response': icd_res['raw_response'][:500] if len(icd_res['raw_response']) > 500 else icd_res['raw_response'],
                })

            logger.info(f"  ✓ Query {query_idx} completed with ICD-10 conversion")

            if query_idx < total_queries:
                await self.api_client._async_sleep()

        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION COMPLETED")
        logger.info(f"{'='*60}")

    def save_results(self, output_path: str):
        df = pd.DataFrame(self.results)

        column_order = [
            'query_id', 'iteration', 'query_filename', 'query_text', 'ground_truth_icd',
            'timestamp_start', 'timestamp_end', 'latency_ms',
            'initial_response', 'final_response',
            'icd_codes', 'icd_categories', 'icd_chapters',
            'icd_conversion_status', 'icd_raw_response',
            'conversation_full', 'status'
        ]

        df = df[column_order]
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Total rows: {len(df)}")

        return df

    def generate_summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.results)
        summary = df.groupby('query_id').agg({
            'latency_ms': ['mean', 'std', 'min', 'max'],
            'status': lambda x: (x == 'success').sum(),
            'icd_conversion_status': lambda x: (x == 'success').sum()
        }).round(2)
        return summary


# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():

    # Validate configuration
    if AES_KEY_BYTES == bytes([0] * 32):
        logger.warning("WARNING: AES_KEY_BYTES not configured.")
        return

    try:
        # Initialize components
        cipher = AESCipher(AES_KEY_BYTES)
        api_client = MedicalAPIClient(API_URL, cipher, delay=DELAY_SECONDS)
        icd_client = ICDConverterClient(ICD_API_URL, delay=DELAY_SECONDS)
        runner = EvaluationRunner(QUERIES_FOLDER, api_client, icd_client, num_iterations=NUM_ITERATIONS)

        # Run evaluation
        await runner.run_evaluation()

        # Save results
        df = runner.save_results(OUTPUT_CSV)

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total conversations: {len(df)}")
        print(f"Successful (DrChatPatin): {(df['status'] == 'success').sum()}")
        print(f"Successful (ICD-10): {(df['icd_conversion_status'] == 'success').sum()}")
        print(f"Average latency: {df['latency_ms'].mean():.2f} ms")
        print(f"\nFile saved: {OUTPUT_CSV}")

    except Exception as e:
        logger.error(f"Fatal error during execution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())