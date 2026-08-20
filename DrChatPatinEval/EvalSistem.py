#!/usr/bin/env python3
"""
Evaluador de Calidad - Asistente Médico Virtual (Enfermedades Raras)
Sistema de evaluación automatizada con flujo de conversación de 2 turnos + conversión ICD-10.
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
# CONFIGURACIÓN - EDITAR ESTOS VALORES
# ============================================================================

# CLAVE AES-256 (32 bytes) - Insertar aquí los 32 bytes de la clave
AES_KEY_BYTES = bytes([
   69, 208, 152, 181, 130, 129, 228,
   16, 193, 207, 179, 148, 151,  31,
   89, 255, 157, 218,  41, 182, 204,
   37, 166,  36,  97,  57, 215, 192,
  198, 102, 178, 202
])

# URL del endpoint API de DrChatPatin (cifrado)
API_URL = "https://drchatpatin.uan.mx/differential_diagnosis"
# API_URL = "https://drchatpatin.uan.mx/use_rag"

# URL del endpoint API para conversión ICD-10 (sin cifrado)
# PON AQUI TU API - Ejemplo: "https://api.ejemplo.com/icd10/convert"
POE_KEY = "wkyPK_lCauPp0psjBeyZWWwqT69_mie1Q4zOaAgA0rQ"
ICD_API_URL = "https://api.poe.com/v1"

client_openai = AsyncOpenAI(api_key=POE_KEY, base_url="https://api.poe.com/v1")

async def UseAPI(query: str) -> str:
    """
    Envía un mensaje al modelo de OpenAI de manera asíncrona.
    Combina instrucciones (ins) y consulta (query).
    """
    messages = [
        {"role": "user", "content": query},
    ]

    # Llamada asíncrona al modelo
    completion = await client_openai.chat.completions.create(
        model="icd-converter", 
        messages=messages,
        timeout=60,  # 
    )

    return completion.choices[0].message.content

# Carpeta con archivos .txt de queries
QUERIES_FOLDER = "./queries"

# Archivo de salida CSV
OUTPUT_CSV = "resultados_evaluacion.csv"

# Delay entre llamadas (segundos)
DELAY_SECONDS = 1.0

# Número de iteraciones por query
NUM_ITERATIONS = 5

# Mensaje de cierre del usuario
CLOSING_MESSAGE = "No se tiene mas informacion disponible, procede con el diagnostico diferencial"

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluacion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CLASES Y FUNCIONES DE CIFRADO
# ============================================================================

@dataclass
class EncryptedPayload:
    iv: str  # hex string
    encripted: str  # hex string
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json_string(self) -> str:
        return json.dumps(self.to_dict())


class AESCipher:
    """Implementación AES-256-CBC compatible con Web Crypto API del frontend."""
    
    def __init__(self, key_bytes: bytes):
        if len(key_bytes) != 32:
            raise ValueError(f"La clave debe ser de 32 bytes, se recibieron {len(key_bytes)}")
        self.key = key_bytes
    
    def _bytes_to_hex(self, data: bytes) -> str:
        return data.hex()
    
    def _hex_to_bytes(self, hex_string: str) -> bytes:
        return bytes.fromhex(hex_string)
    
    def encrypt(self, plaintext: str) -> EncryptedPayload:
        """Cifra un texto plano y retorna payload con IV y datos cifrados en hex."""
        iv = os.urandom(16)
        
        # Padding PKCS7 manual
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
        """Descifra un payload encriptado."""
        iv = self._hex_to_bytes(encrypted_data['iv'])
        encrypted = self._hex_to_bytes(encrypted_data['encripted'])
        
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted) + decryptor.finalize()
        
        # Remover padding PKCS7
        padding_length = decrypted_padded[-1]
        decrypted = decrypted_padded[:-padding_length]
        
        return decrypted.decode('utf-8')


# ============================================================================
# CLIENTE DE API ICD-10 (SIN CIFRADO)
# ============================================================================

class ICDConverterClient:
    """
    Cliente para convertir diagnósticos diferenciales a códigos ICD-10.
    API externa sin cifrado AES.
    """
    
    def __init__(self, api_url: str, delay: float = 1.0):
        if "PON AQUI" in api_url or not api_url:
            logger.warning("⚠️  ICD_API_URL no configurada. La conversión ICD-10 será simulada.")
            self.active = False
        else:
            self.active = True
            self.api_url = api_url
            self.delay = delay
            self.session = requests.Session()
            self.session.headers.update({'Content-Type': 'application/json'})
    
    

    async def convert_to_icd(self, differential_diagnosis: str) -> Dict:
        """
        Convierte un diagnóstico diferencial a lista de códigos ICD-10.
        
        Returns:
            Dict con: {
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
                'raw_response': 'API no configurada',
                'status': 'not_configured'
            }
        
        time.sleep(self.delay)
        
        try:
            def get_icd10_chapter(code: str) -> str:
                if not code or not code[0].isalpha():
                    return 'Unknown'
                prefix = code[0].upper()
                num = int(code[1:3]) if len(code) >= 3 and code[1:3].isdigit() else 0
                full = f"{prefix}{num:02d}"

                chapters = [
                    ('I',    ['A00','B99']),
                    ('II',   ['C00','D48']),
                    ('III',  ['D50','D89']),
                    ('IV',   ['E00','E90']),
                    ('V',    ['F00','F99']),
                    ('VI',   ['G00','G99']),
                    ('VII',  ['H00','H59']),
                    ('VIII', ['H60','H95']),
                    ('IX',   ['I00','I99']),
                    ('X',    ['J00','J99']),
                    ('XI',   ['K00','K93']),
                    ('XII',  ['L00','L99']),
                    ('XIII', ['M00','M99']),
                    ('XIV',  ['N00','N99']),
                    ('XV',   ['O00','O99']),
                    ('XVI',  ['P00','P96']),
                    ('XVII', ['Q00','Q99']),
                    ('XVIII',['R00','R99']),
                    ('XIX',  ['S00','T98']),
                    ('XX',   ['V01','Y98']),
                    ('XXI',  ['Z00','Z99']),
                    ('XXII', ['U00','U99']),
                ]
                for chapter, (start, end) in chapters:
                    if full >= start and full <= end:
                        return chapter
                return 'Unknown'
            
            ans = await UseAPI(differential_diagnosis)
            result = json.loads(ans)
            print(result, type(result))
            
            icd_codes = result.get('codes', result.get('icd_codes', []))
            icd_categories = list(set([code.split('.')[0] for code in icd_codes if '.' in code]))
            
            icd_chapters = []
            for code in icd_codes:
                if code[0].isalpha():
                    icd_chapters.append(code[0])
                elif code[0].isdigit():
                    chapter_map = {...}
                    icd_chapters.append(chapter_map.get(code[0], code[0]))

            icd_chapters = [get_icd10_chapter(code) for code in icd_codes]
            
            return {
                'icd_codes': icd_codes,
                'icd_categories': list(set(icd_categories)),
                'icd_chapters': list(set(icd_chapters)),
                'raw_response': json.dumps(result),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error en conversión ICD-10: {e}")
            return {
                'icd_codes': [],
                'icd_categories': [],
                'icd_chapters': [],
                'raw_response': str(e),
                'status': f'error: {str(e)}'
            }
    
    async def convert_batch(self, diagnoses: List[str]) -> List[Dict]:
        """Convierte múltiples diagnósticos (para las 5 iteraciones)."""
        results = []
        for diag in diagnoses:
            results.append(await self.convert_to_icd(diag))
            if self.active:
                await asyncio.sleep(self.delay)
        return results


# ============================================================================
# CLIENTE DE API DRCHATPATIN (CON CIFRADO)
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
        """Envía un mensaje a la API y retorna la respuesta descifrada."""
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
            
            # Intentar parsear como JSON primero
            try:
                parsed = json.loads(decrypted_text)
                if isinstance(parsed, dict) and 'sender' in parsed and 'text' in parsed:
                    return parsed, elapsed_ms
            except json.JSONDecodeError:
                pass
            
            # Si no es JSON, extraer de thinking process
            clean_text = self._extract_bot_response(decrypted_text)
            
            return {
                "sender": "bot",
                "text": clean_text
            }, elapsed_ms
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en petición HTTP: {e}")
            raise
        except Exception as e:
            logger.error(f"Error procesando respuesta: {e}")
            raise
    
    def _extract_bot_response(self, raw_text: str) -> str:
        """Extrae la respuesta útil del texto crudo del modelo."""
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
            result = '\n'.join([l for l in result.split('\n') if not l.strip().startswith('>')])
        
        return result if result else raw_text.strip()
    
    def run_conversation_flow(self, initial_query: str) -> Dict:
        """Ejecuta el flujo completo de conversación de 2 turnos."""
        conversation = []
        timestamps = {}
        
        # === TURNO 1: Query inicial ===
        logger.info("  Turno 1: Enviando query inicial...")
        conversation.append({"sender": "user", "text": initial_query})
        
        self._sleep()
        timestamps['t1_start'] = datetime.now().isoformat()
        resp1, lat1 = self.send_message(conversation)
        timestamps['t1_end'] = datetime.now().isoformat()
        
        respuesta_inicial = resp1.get('text', str(resp1)) if isinstance(resp1, dict) else str(resp1)
        conversation.append({"sender": "bot", "text": respuesta_inicial})
        logger.info(f"  Respuesta 1 recibida ({lat1:.0f}ms): {respuesta_inicial[:150]}...")
        
        # === TURNO 2: Respuesta de cierre ===
        logger.info("  Turno 2: Enviando mensaje de cierre...")
        conversation.append({"sender": "user", "text": CLOSING_MESSAGE})
        
        self._sleep()
        timestamps['t2_start'] = datetime.now().isoformat()
        resp2, lat2 = self.send_message(conversation)
        timestamps['t2_end'] = datetime.now().isoformat()
        
        respuesta_final = resp2.get('text', str(resp2)) if isinstance(resp2, dict) else str(resp2)
        conversation.append({"sender": "bot", "text": respuesta_final})
        logger.info(f"  Respuesta 2 (diagnóstico) recibida ({lat2:.0f}ms): {respuesta_final[:150]}...")
        
        total_latency = lat1 + lat2
        
        return {
            'respuesta_inicial': respuesta_inicial,
            'respuesta_final': respuesta_final,
            'conversation_json': json.dumps(conversation, ensure_ascii=False),
            'latency_ms': total_latency,
            'timestamps': timestamps,
            'status': 'success'
        }


# ============================================================================
# ORQUESTADOR DE EVALUACIÓN
# ============================================================================

class EvaluationRunner:
    def __init__(self, queries_folder: str, api_client: MedicalAPIClient, 
                 icd_client: ICDConverterClient, num_iterations: int = 5):
        self.queries_folder = Path(queries_folder)
        self.api_client = api_client
        self.icd_client = icd_client
        self.num_iterations = num_iterations
        self.results = []
    
    async def _async_sleep(self):  
        await asyncio.sleep(self.delay)
        
    def load_queries(self) -> List[Tuple[int, str, str, str]]:
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
                logger.warning(f"Archivo {filepath.name} no tiene GROUND_TRUTH")
            
            queries.append((idx, filepath.name, case_text, ground_truth_icd))
            logger.info(f"Cargada query {idx}: {filepath.name} | GT: {ground_truth_icd}")
        
        return queries
    
    async def run_evaluation(self):
        """Ejecuta la evaluación completa."""
        queries = self.load_queries()
        total_queries = len(queries)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"INICIANDO EVALUACIÓN")
        logger.info(f"Total queries: {total_queries}")
        logger.info(f"Iteraciones por query: {self.num_iterations}")
        logger.info(f"Total de conversaciones a generar: {total_queries * self.num_iterations}")
        logger.info(f"{'='*60}\n")
        
        for query_idx, filename, query_text, ground_truth_icd in queries:
            logger.info(f"\n--- Query {query_idx}/{total_queries}: {filename} ---")
            
            # Almacenar respuestas de las 5 iteraciones para conversión ICD posterior
            iteracion_results = []
            respuestas_finales = []
            
            for iteration in range(self.num_iterations):
                logger.info(f"  Iteración {iteration + 1}/{self.num_iterations}")
                
                try:
                    result = self.api_client.run_conversation_flow(query_text)
                    iteracion_results.append(result)
                    respuestas_finales.append(result['respuesta_final'])
                    
                except Exception as e:
                    logger.error(f"  ERROR en iteración {iteration}: {e}")
                    iteracion_results.append({
                        'respuesta_inicial': '',
                        'respuesta_final': '',
                        'conversation_json': json.dumps([{"error": str(e)}]),
                        'latency_ms': 0,
                        'timestamps': {'t1_start': datetime.now().isoformat(), 't2_end': datetime.now().isoformat()},
                        'status': f'error: {str(e)}'
                    })
                    respuestas_finales.append('')
                
                if iteration < self.num_iterations - 1:
                    await self.api_client._async_sleep()
            
            # === CONVERSIÓN ICD-10 DESPUÉS DE LAS 5 ITERACIONES ===
            logger.info(f"  Convirtiendo {len(respuestas_finales)} diagnósticos a ICD-10...")
            icd_results = await self.icd_client.convert_batch(respuestas_finales)
            
            # Combinar resultados de iteraciones con ICD
            for i, (iter_res, icd_res) in enumerate(zip(iteracion_results, icd_results)):
                self.results.append({
                    # Identificación
                    'query_id': query_idx,
                    'query_filename': filename,
                    'iteracion': i,
                    'query_text': query_text,
                    'ground_truth_icd': json.dumps(ground_truth_icd), 
                    
                    # Timestamps
                    'timestamp_inicio': iter_res['timestamps']['t1_start'],
                    'timestamp_fin': iter_res['timestamps']['t2_end'],
                    
                    # Respuestas del modelo
                    'respuesta_inicial': iter_res['respuesta_inicial'],
                    'respuesta_final': iter_res['respuesta_final'],
                    'conversation_full': iter_res['conversation_json'],
                    'latency_ms': round(iter_res['latency_ms'], 2),
                    'status': iter_res['status'],
                    
                    # Datos ICD-10 (nuevos)
                    'icd_codes': json.dumps(icd_res['icd_codes']),
                    'icd_categories': json.dumps(icd_res['icd_categories']),
                    'icd_chapters': json.dumps(icd_res['icd_chapters']),
                    'icd_conversion_status': icd_res['status'],
                    'icd_raw_response': icd_res['raw_response'][:500] if len(icd_res['raw_response']) > 500 else icd_res['raw_response'],
                    'ground_truth_icd': json.dumps(ground_truth_icd)
                })
            
            logger.info(f"  ✓ Query {query_idx} completada con conversión ICD-10")
            
            if query_idx < total_queries:
                await self.api_client._async_sleep()
        
        logger.info(f"\n{'='*60}")
        logger.info("EVALUACIÓN COMPLETADA")
        logger.info(f"{'='*60}")
    
    def save_results(self, output_path: str):
        """Guarda los resultados en CSV."""
        df = pd.DataFrame(self.results)
        
        column_order = [
            'query_id', 'iteracion', 'query_filename', 'query_text', 'ground_truth_icd',
            'timestamp_inicio', 'timestamp_fin', 'latency_ms',
            'respuesta_inicial', 'respuesta_final', 
            'icd_codes', 'icd_categories', 'icd_chapters',
            'icd_conversion_status', 'icd_raw_response',
            'conversation_full', 'status'
        ]
        
        df = df[column_order]
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Resultados guardados en: {output_path}")
        logger.info(f"Total de filas: {len(df)}")
        
        return df
    
    def generate_summary(self) -> pd.DataFrame:
        """Genera resumen estadístico por query."""
        df = pd.DataFrame(self.results)
        summary = df.groupby('query_id').agg({
            'latency_ms': ['mean', 'std', 'min', 'max'],
            'status': lambda x: (x == 'success').sum(),
            'icd_conversion_status': lambda x: (x == 'success').sum()
        }).round(2)
        return summary


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

async def main():
    """Punto de entrada principal."""
    
    # Validar configuración
    if AES_KEY_BYTES == bytes([0] * 32):
        logger.warning("⚠️  ADVERTENCIA: AES_KEY_BYTES no configurada.")
        return
    
    if "PON AQUI" in API_URL or not API_URL:
        logger.warning("⚠️  ADVERTENCIA: API_URL no configurada.")
        return
    
    try:
        # Inicializar componentes
        cipher = AESCipher(AES_KEY_BYTES)
        api_client = MedicalAPIClient(API_URL, cipher, delay=DELAY_SECONDS)
        icd_client = ICDConverterClient(ICD_API_URL, delay=DELAY_SECONDS)
        runner = EvaluationRunner(QUERIES_FOLDER, api_client, icd_client, num_iterations=NUM_ITERATIONS)
        
        # Ejecutar evaluación
        await runner.run_evaluation()
        
        # Guardar resultados
        df = runner.save_results(OUTPUT_CSV)
        
        # Mostrar resumen
        print("\n" + "="*60)
        print("RESUMEN DE EVALUACIÓN")
        print("="*60)
        print(f"Total de conversaciones: {len(df)}")
        print(f"Exitosas (DrChatPatin): {(df['status'] == 'success').sum()}")
        print(f"Exitosas (ICD-10): {(df['icd_conversion_status'] == 'success').sum()}")
        print(f"Latencia promedio: {df['latency_ms'].mean():.2f} ms")
        print(f"\nArchivo guardado: {OUTPUT_CSV}")
        
        # Guardar también resumen estadístico
        summary = runner.generate_summary()
        summary.to_csv("resumen_estadistico.csv")
        print(f"Resumen estadístico guardado: resumen_estadistico.csv")
        
    except Exception as e:
        logger.error(f"Error fatal en la ejecución: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())