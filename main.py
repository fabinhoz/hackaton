#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinSight AI — Backend API v5.0
G9-BR-TEAM-20 | Hackathon OCI

Melhorias v5 (alinhadas ao FinanceAI.pdf):
- Score numérico de saúde financeira (0-1000)
- Metas/orçamento por categoria + alertas de gastos elevados
- Explicabilidade do perfil (motivos da classificação)
- Simulador de cenários "E se...?"
- Exportação estruturada de relatório
- Gráfico de evolução via endpoint /historico evoluído
- Fallback heurístico com score dinâmico e explicações
"""

import os
import sys
import json
import gc
import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any
from io import StringIO

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("finsight")

# ------------------------------------------------------------------
# Constantes & Ambiente
# ------------------------------------------------------------------
BASE_DIR = os.path.expanduser("~/finsight-ai")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "finsight.db")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

BUCKET_NAME = os.getenv("OCI_BUCKET_NAME", "hackathon-one-g9-team-20")
BUCKET_NAMESPACE = os.getenv("OCI_BUCKET_NAMESPACE", "")
MODEL_PREFIX = "models/"
DATASET_PREFIX = "datasets/"

CATEGORIAS = ["alimentacao", "transporte", "saude", "moradia", "educacao", "lazer", "servicos"]

# ------------------------------------------------------------------
# OCI Object Storage — Sync com retry e cache local
# ------------------------------------------------------------------
def sync_models_from_oci(max_retries: int = 3) -> bool:
    try:
        import oci
        from oci.object_storage import ObjectStorageClient

        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        client = ObjectStorageClient(config={}, signer=signer)

        namespace = BUCKET_NAMESPACE
        if not namespace:
            namespace = client.get_namespace().data
            logger.info(f"[OCI] Namespace detectado: {namespace}")

        logger.info(f"[OCI] Sincronizando modelos do bucket '{BUCKET_NAME}' ...")
        response = client.list_objects(
            namespace_name=namespace,
            bucket_name=BUCKET_NAME,
            prefix=MODEL_PREFIX
        )

        downloaded = 0
        for obj in response.data.objects:
            filename = os.path.basename(obj.name)
            if filename.startswith("models_"):
                filename = filename[7:]
            if not filename.endswith(".pkl"):
                continue
            local_path = os.path.join(MODEL_DIR, filename)
            try:
                head = client.head_object(namespace_name=namespace, bucket_name=BUCKET_NAME, object_name=obj.name)
                remote_size = int(head.headers.get("Content-Length", 0))
                if os.path.exists(local_path) and os.path.getsize(local_path) == remote_size:
                    logger.info(f"[OCI] Cache hit: {filename}")
                    downloaded += 1
                    continue
            except Exception:
                pass

            logger.info(f"[OCI] Baixando {obj.name} -> {local_path}")
            for attempt in range(max_retries):
                try:
                    get_resp = client.get_object(namespace_name=namespace, bucket_name=BUCKET_NAME, object_name=obj.name)
                    with open(local_path, "wb") as f:
                        for chunk in get_resp.data.raw.stream(1024 * 1024, decode_content=False):
                            f.write(chunk)
                    downloaded += 1
                    break
                except Exception as e:
                    logger.warning(f"[OCI] Tentativa {attempt+1}/{max_retries} falhou para {filename}: {e}")
                    if attempt == max_retries - 1:
                        raise

        logger.info(f"[OCI] Sync concluído. {downloaded} artefato(s) pronto(s).")
        return True

    except Exception as e:
        logger.warning(f"[OCI] Falha no sync (usando modelos locais): {e}")
        return False

# ------------------------------------------------------------------
# Carregamento de Modelos com joblib + OOM Prevention
# ------------------------------------------------------------------
class ModelLoader:
    MODELS = {
        "vetorizador": "vetorizador_tfidf.pkl",
        "modelo_categoria": "modelo_categoria_producao.pkl",
        "codificador_categoria": "codificador_categorias.pkl",
        "modelo_perfil": "modelo_perfil_producao.pkl",
        "codificador_perfil": "codificador_perfil.pkl"
    }

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.loaded = False
        self._sync_and_load()

    def _sync_and_load(self):
        sync_models_from_oci()
        self._load_all()
        gc.collect()
        logger.info(f"[GC] Coletor de lixo executado. Memória liberada.")

    def _load_all(self):
        import joblib
        for key, filename in self.MODELS.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                try:
                    self.models[key] = joblib.load(path)
                    logger.info(f"[joblib] Carregado: {filename}")
                except Exception as e:
                    logger.error(f"[joblib] Erro em {filename}: {e}")
            else:
                logger.warning(f"[joblib] Não encontrado: {filename}")

        self.loaded = len(self.models) == len(self.MODELS)
        status = "OK" if self.loaded else "FALLBACK"
        logger.info(f"[ModelLoader] Status: {status} ({len(self.models)}/{len(self.MODELS)})")

    def get(self, key: str):
        return self.models.get(key)

    def reload(self):
        self.models.clear()
        self._sync_and_load()

ml = ModelLoader()

# ------------------------------------------------------------------
# SQLite — Histórico de Análises
# ------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historico (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            renda_mensal REAL,
            nivel_endividamento REAL,
            frequencia_poupanca TEXT,
            perfil_financeiro TEXT,
            probabilidade REAL,
            score_saude INTEGER,
            total_gasto REAL,
            percentual_renda_gasta REAL,
            comprometimento_gastos REAL,
            resumo_gastos TEXT,
            transacoes TEXT,
            recomendacoes TEXT,
            alertas TEXT,
            explicabilidade TEXT,
            metas TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"[SQLite] Banco inicializado: {DB_PATH}")

init_db()

def salvar_historico(renda, endiv, poup, perfil, prob, score, total, pct_renda, comprometimento,
                     resumo, transacoes, recomendacoes, alertas, explicabilidade, metas):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO historico 
            (timestamp, renda_mensal, nivel_endividamento, frequencia_poupanca, perfil_financeiro,
             probabilidade, score_saude, total_gasto, percentual_renda_gasta, comprometimento_gastos,
             resumo_gastos, transacoes, recomendacoes, alertas, explicabilidade, metas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat() + "Z",
            renda, endiv, poup, perfil, prob, score, total, pct_renda, comprometimento,
            json.dumps(resumo, ensure_ascii=False),
            json.dumps(transacoes, ensure_ascii=False),
            json.dumps(recomendacoes, ensure_ascii=False),
            json.dumps(alertas, ensure_ascii=False),
            json.dumps(explicabilidade, ensure_ascii=False),
            json.dumps(metas, ensure_ascii=False)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"[SQLite] Erro ao salvar histórico: {e}")

# ------------------------------------------------------------------
# Keywords para Fallback
# ------------------------------------------------------------------
KEYWORDS = {
    "alimentacao": ["supermercado","mercado","restaurante","lanche","padaria","ifood","comida","pizza","burger","acougue","feira","cafe","sorvete","churrascaria","sushi","pastel","doceria","hamburguer","marmita","delivery","rappi","uber eats","alimentacao","alimento","refeicao"],
    "transporte": ["uber","99","combustivel","gasolina","onibus","metro","estacionamento","taxi","bilhete","pedagio","lavagem","mecanico","seguro auto","ipva","diesel","etanol","moto","carro","van","onibus","passagem","transporte","uberflash","corrida"],
    "saude": ["farmacia","remedio","consulta","medico","dentista","hospital","plano","vitamina","exame","fisioterapia","psicologo","academia","nutricionista","vacina","saude","droga","raia","drogasil","pague menos","convenio","terapia"],
    "moradia": ["aluguel","condominio","iptu","luz","agua","gas","internet","tv","moveis","reforma","manutencao","limpeza","jardinagem","eletricidade","energia","celpe","equatorial","enel","oi","vivo","claro","net","moradia","aluguel"],
    "educacao": ["curso","faculdade","livro","material","escola","universidade","certificacao","workshop","palestra","tutoria","idioma","tecnologia","mba","udemy","coursera","alura","rocketseat","dio","kenzie","educacao","estudo","apostila"],
    "lazer": ["cinema","netflix","spotify","show","viagem","hotel","passeio","jogo","teatro","museu","parque","praia","festa","bar","karaoke","disney","prime video","hbo","paramount","globo play","youtube","twitch","lazer","entretenimento","streaming"],
    "servicos": ["assinatura","manutencao","limpeza","contabilidade","advogado","seguro","consultoria","design","fotografia","streaming","cloud","hosting","dominio","aws","azure","google cloud","hotmart","monetize","servico","telefonia","celular"]
}

# ------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------
class Transacao(BaseModel):
    descricao: str = Field(..., min_length=1, max_length=200, examples=["Supermercado Extra"])
    valor: float = Field(..., gt=0, examples=[420.50])


class AnaliseRequest(BaseModel):
    renda_mensal: float = Field(..., gt=0, examples=[4500.00])
    nivel_endividamento: float = Field(..., ge=0, le=100, examples=[25.0])
    frequencia_poupanca: str = Field(..., examples=["Media"])
    transacoes: List[Transacao] = Field(..., min_items=1)
    metas: Optional[Dict[str, float]] = Field(default_factory=dict, examples=[{"alimentacao": 600, "lazer": 200}])

    @validator("frequencia_poupanca")
    def validar_poupanca(cls, v):
        v_norm = v.strip().capitalize()
        if v_norm not in ["Alta", "Media", "Baixa"]:
            raise ValueError("Valores válidos: Alta, Media, Baixa")
        return v_norm


class AnaliseResponse(BaseModel):
    perfil_financeiro: str
    probabilidade: float
    score_saude: int
    resumo_gastos: Dict[str, float]
    recomendacoes: List[str]
    alertas: List[str]
    explicabilidade: List[str]
    total_gasto: float
    percentual_renda_gasta: float
    comprometimento_gastos: float
    transacoes_classificadas: List[Dict]


class SimuladorRequest(BaseModel):
    renda_mensal: float = Field(..., gt=0)
    nivel_endividamento: float = Field(..., ge=0, le=100)
    frequencia_poupanca: str = Field(...)
    transacoes: List[Transacao] = Field(..., min_items=1)
    metas: Optional[Dict[str, float]] = Field(default_factory=dict)

    @validator("frequencia_poupanca")
    def validar_poupanca(cls, v):
        v_norm = v.strip().capitalize()
        if v_norm not in ["Alta", "Media", "Baixa"]:
            raise ValueError("Valores válidos: Alta, Media, Baixa")
        return v_norm


class RelatorioResponse(BaseModel):
    id: int
    timestamp: str
    renda_mensal: float
    nivel_endividamento: float
    frequencia_poupanca: str
    perfil_financeiro: str
    score_saude: int
    total_gasto: float
    percentual_renda_gasta: float
    comprometimento_gastos: float
    resumo_gastos: Dict[str, float]
    recomendacoes: List[str]
    alertas: List[str]
    explicabilidade: List[str]
    transacoes_classificadas: List[Dict]
    metas: Dict[str, float]

# ------------------------------------------------------------------
# Serviços
# ------------------------------------------------------------------
class CategoriaService:
    @classmethod
    def classificar(cls, descricao: str, valor: float) -> tuple:
        vetorizador = ml.get("vetorizador")
        modelo = ml.get("modelo_categoria")
        codificador = ml.get("codificador_categoria")

        if vetorizador and modelo and codificador:
            try:
                X = vetorizador.transform([descricao.lower()])
                pred = modelo.predict(X)[0]
                proba = modelo.predict_proba(X)[0]
                confianca = float(np.max(proba))
                categoria = codificador.inverse_transform([pred])[0]
                cat_norm = cls._normalizar_categoria(categoria)
                return cat_norm, confianca
            except Exception as e:
                logger.warning(f"[Categoria] Erro modelo: {e}")

        desc_lower = descricao.lower()
        for cat, keywords in KEYWORDS.items():
            if any(kw in desc_lower for kw in keywords):
                matches = sum(1 for kw in keywords if kw in desc_lower)
                conf = min(0.75 + (matches * 0.03), 0.92)
                return cat, round(conf, 4)

        if valor > 1000:
            return "moradia", 0.60
        elif valor > 200:
            return "servicos", 0.55
        else:
            return "lazer", 0.50

    @staticmethod
    def _normalizar_categoria(cat: str) -> str:
        cat_lower = cat.lower().strip()
        mapping = {
            "alimentação": "alimentacao", "alimentacao": "alimentacao",
            "transporte": "transporte",
            "saúde": "saude", "saude": "saude",
            "moradia": "moradia",
            "educação": "educacao", "educacao": "educacao",
            "lazer": "lazer",
            "serviços": "servicos", "servicos": "servicos"
        }
        return mapping.get(cat_lower, "servicos")


class PerfilService:
    POUPANCA_MAP = {"Alta": 1.0, "Media": 0.5, "Baixa": 0.25}

    @classmethod
    def analisar(cls, renda, endividamento, poupanca, resumo_gastos, total_gasto):
        modelo = ml.get("modelo_perfil")
        codificador = ml.get("codificador_perfil")

        comprometimento = cls._calcular_comprometimento(renda, total_gasto, endividamento)
        ratio_poup = cls.POUPANCA_MAP.get(poupanca, 0.0)

        if modelo and codificador:
            try:
                expected_names = None
                if hasattr(modelo, "feature_names_in_"):
                    expected_names = list(modelo.feature_names_in_)

                if expected_names:
                    row = cls._mapear_features(expected_names, renda, endividamento, ratio_poup, total_gasto, comprometimento, resumo_gastos)
                else:
                    row = cls._row_padrao(renda, endividamento, ratio_poup, total_gasto, comprometimento, resumo_gastos)

                df = pd.DataFrame([row])
                pred = modelo.predict(df)[0]
                proba = modelo.predict_proba(df)[0]
                confianca = float(np.max(proba))
                perfil = codificador.inverse_transform([pred])[0]
                score, explic = cls._calcular_score_e_explicacao(renda, endividamento, poupanca, total_gasto, resumo_gastos, str(perfil))
                return str(perfil), round(confianca, 4), score, explic
            except Exception as e:
                logger.warning(f"[Perfil] Erro modelo: {e}. Fallback heurístico ativado.")

        return cls._fallback_heuristico(renda, endividamento, poupanca, total_gasto, resumo_gastos)

    @staticmethod
    def _calcular_comprometimento(renda, total_gasto, endividamento):
        if renda <= 0:
            return 100.0
        pct_gasto = (total_gasto / renda) * 100
        comp = pct_gasto + endividamento
        return round(min(comp, 100.0), 2)

    @classmethod
    def _mapear_features(cls, expected_names, renda, endividamento, ratio_poup, total_gasto, comprometimento, resumo_gastos):
        row = {}
        for name in expected_names:
            nl = name.lower().strip()
            if any(x in nl for x in ["renda_mensal", "renda"]):
                row[name] = renda
            elif any(x in nl for x in ["nivel_endividamento", "endividamento"]):
                row[name] = endividamento
            elif any(x in nl for x in ["frequencia_poupanca", "poupanca", "freq_poup"]):
                row[name] = ratio_poup * 100
            elif any(x in nl for x in ["total_gasto", "gasto_total"]):
                row[name] = total_gasto
            elif any(x in nl for x in ["comprometimento_gastos", "comprometimento", "ratio_gasto"]):
                row[name] = comprometimento
            elif any(x in nl for x in ["alimentacao", "alimentação"]):
                row[name] = resumo_gastos.get("alimentacao", 0.0)
            elif any(x in nl for x in ["transporte"]):
                row[name] = resumo_gastos.get("transporte", 0.0)
            elif any(x in nl for x in ["saude", "saúde"]):
                row[name] = resumo_gastos.get("saude", 0.0)
            elif any(x in nl for x in ["moradia"]):
                row[name] = resumo_gastos.get("moradia", 0.0)
            elif any(x in nl for x in ["educacao", "educação"]):
                row[name] = resumo_gastos.get("educacao", 0.0)
            elif any(x in nl for x in ["lazer"]):
                row[name] = resumo_gastos.get("lazer", 0.0)
            elif any(x in nl for x in ["servicos", "serviços"]):
                row[name] = resumo_gastos.get("servicos", 0.0)
            else:
                row[name] = 0.0
                logger.warning(f"[Perfil] Feature desconhecida: {name}, usando 0")
        return row

    @classmethod
    def _row_padrao(cls, renda, endividamento, ratio_poup, total_gasto, comprometimento, resumo_gastos):
        return {
            "renda_mensal": renda,
            "nivel_endividamento": endividamento,
            "frequencia_poupanca": ratio_poup * 100,
            "total_gasto": total_gasto,
            "comprometimento_gastos": comprometimento,
            "alimentacao": resumo_gastos.get("alimentacao", 0.0),
            "transporte": resumo_gastos.get("transporte", 0.0),
            "saude": resumo_gastos.get("saude", 0.0),
            "moradia": resumo_gastos.get("moradia", 0.0),
            "educacao": resumo_gastos.get("educacao", 0.0),
            "lazer": resumo_gastos.get("lazer", 0.0),
        }

    @classmethod
    def _calcular_score_e_explicacao(cls, renda, endividamento, poupanca, total_gasto, resumo_gastos, perfil):
        ratio = total_gasto / renda if renda > 0 else 1.0
        score_endiv = max(0, 300 - (endividamento * 3))
        poup_map = {"Alta": 250, "Media": 150, "Baixa": 50}
        score_poup = poup_map.get(poupanca, 50)

        if ratio <= 0.30:
            score_gasto = 300
        elif ratio <= 0.50:
            score_gasto = 220
        elif ratio <= 0.70:
            score_gasto = 140
        elif ratio <= 0.90:
            score_gasto = 70
        else:
            score_gasto = 10

        perfil_bonus = {"Saudavel": 150, "Em observacao": 70, "Em risco": 0}
        bonus = perfil_bonus.get(perfil, 0)

        score = int(min(1000, score_endiv + score_poup + score_gasto + bonus))

        explic = []
        explic.append(f"Endividamento em {endividamento:.0f}% {'(ideal)' if endividamento <= 30 else '(acima do ideal de 30%)'}")
        explic.append(f"Poupança classificada como {poupanca}")
        explic.append(f"Gastos consomem {ratio*100:.1f}% da renda {'(saudável)' if ratio <= 0.5 else '(atenção)'}")
        maior_cat = max(resumo_gastos, key=resumo_gastos.get) if resumo_gastos else "N/A"
        explic.append(f"Maior categoria de gasto: {maior_cat} (R$ {resumo_gastos.get(maior_cat, 0):,.2f})")

        return score, explic

    @classmethod
    def _fallback_heuristico(cls, renda, endividamento, poupanca, total_gasto, resumo_gastos):
        ratio = total_gasto / renda if renda > 0 else 1.0
        score = 0
        if endividamento <= 30: score += 2
        elif endividamento <= 50: score += 1
        elif endividamento <= 70: score += 0
        else: score -= 2

        if poupanca == "Alta": score += 2
        elif poupanca == "Media": score += 1
        else: score += 0

        if ratio <= 0.30: score += 3
        elif ratio <= 0.50: score += 2
        elif ratio <= 0.70: score += 1
        elif ratio <= 0.90: score -= 1
        else: score -= 3

        if score >= 5:
            perfil = "Saudavel"
        elif score >= 2:
            perfil = "Em observacao"
        else:
            perfil = "Em risco"

        score_num, explic = cls._calcular_score_e_explicacao(renda, endividamento, poupanca, total_gasto, resumo_gastos, perfil)
        conf = 0.88 if perfil == "Saudavel" else (0.78 if perfil == "Em observacao" else 0.85)
        return perfil, round(conf, 4), score_num, explic


class AlertaService:
    @classmethod
    def gerar(cls, resumo_gastos, metas, renda, endividamento, total_gasto):
        alertas = []
        for cat, gasto in resumo_gastos.items():
            meta = metas.get(cat, 0)
            if meta > 0:
                pct = (gasto / meta) * 100
                if gasto > meta:
                    alertas.append(f"{cat.upper()}: gasto de R$ {gasto:,.2f} ultrapassou a meta de R$ {meta:,.2f} ({pct:.0f}%)")
                elif pct >= 90:
                    alertas.append(f"{cat.upper()}: próximo da meta ({pct:.0f}%)")
        if endividamento > 30:
            alertas.append(f"Endividamento elevado: {endividamento:.0f}% (recomendado <30%)")
        ratio = total_gasto / renda if renda > 0 else 1
        if ratio > 0.9:
            alertas.append(f"Gastos consomem {ratio*100:.0f}% da renda — risco de endividamento")
        elif ratio > 0.7:
            alertas.append(f"Gastos consomem {ratio*100:.0f}% da renda — atenção")
        return alertas


class RecomendacaoService:
    @classmethod
    def gerar(cls, perfil, resumo_gastos, renda, endividamento, poupanca, total_gasto, metas):
        recs = []
        ratio = total_gasto / renda if renda > 0 else 0

        if perfil == "Em risco":
            recs.append("Priorize o pagamento de dívidas com juros mais altos (cartão, cheque especial)")
            recs.append("Crie um plano emergencial de redução de gastos em 30 dias")
            recs.append("Evite novos compromissos de crédito até regularizar a situação")
            recs.append("Considere negociar dívidas e buscar renda extra imediata")
        elif perfil == "Em observacao":
            recs.append("Monitore gastos recorrentes e cancele assinaturas não utilizadas")
            recs.append("Estabeleça um orçamento mensal com limite por categoria")
            recs.append("Crie uma reserva de emergência de pelo menos 3 salários")
            recs.append("Revise seu plano de saúde e seguros para otimizar custos")
        else:
            recs.append("Continue mantendo hábitos saudáveis de organização financeira")
            recs.append("Considere diversificar investimentos em renda fixa e variável")
            recs.append("Avalie metas de longo prazo (aposentadoria, imóvel, intercâmbio)")
            recs.append("Com sua margem confortável, estude opções de independência financeira")

        for cat, val in resumo_gastos.items():
            pct = (val / renda * 100) if renda > 0 else 0
            meta = metas.get(cat, 0)
            if meta > 0 and val > meta:
                recs.append(f"{cat.capitalize()}: reduza para atingir a meta de R$ {meta:,.2f} (gasto atual R$ {val:,.2f})")
            elif cat == "lazer" and pct > 15:
                recs.append(f"Lazer está em {pct:.1f}% da renda — considere reduzir para até 10-15%")
            elif cat == "alimentacao" and pct > 30:
                recs.append(f"Alimentação: {pct:.1f}% da renda — planeje refeições em casa e evite delivery excessivo")
            elif cat == "transporte" and pct > 20:
                recs.append(f"Transporte: {pct:.1f}% da renda — avalie transporte público, carona ou bike")
            elif cat == "moradia" and pct > 35:
                recs.append(f"Moradia: {pct:.1f}% da renda — acima do recomendado (30%). Avalie renegociação")
            elif cat == "saude" and pct > 10:
                recs.append(f"Saúde: {pct:.1f}% da renda — verifique se seu plano cobre preventivo")
            elif cat == "educacao" and pct > 15:
                recs.append(f"Educação: {pct:.1f}% da renda — excelente investimento, mas monitore o retorno")
            elif cat == "servicos" and pct > 10:
                recs.append(f"Serviços: {pct:.1f}% da renda — revise assinaturas e serviços duplicados")

        if poupanca == "Baixa":
            recs.append("Estabeleça meta de poupança mensal mínima de 10% da renda")
        if endividamento > 30:
            recs.append(f"Endividamento ({endividamento:.0f}%) > 30% — foque em quitar dívidas caras primeiro")
        if ratio > 0.8:
            recs.append(f"Gastando {ratio*100:.0f}% da renda — revise despesas e adie compras não essenciais")
        if ratio < 0.5 and poupanca == "Alta":
            recs.append("Excelente margem de poupança! Considere investir o excedente em Tesouro ou CDBs")
        if endividamento < 10 and poupanca == "Alta" and ratio < 0.5:
            recs.append("Perfil exemplar! Você está no caminho da independência financeira")

        return recs[:8]


# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------
app = FastAPI(
    title="FinSight AI v5.0",
    description="Assistente Inteligente de Saúde Financeira — G9-BR-TEAM-20",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")

@app.get("/", response_class=CustomJSONResponse)
def root():
    return {
        "nome": "FinSight AI",
        "versao": "5.0.0",
        "equipe": "G9-BR-TEAM-20",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/analise-financeira",
            "/classificar",
            "/analise-batch-csv",
            "/simular",
            "/relatorio/{item_id}",
            "/historico",
            "/reload-models"
        ]
    }

@app.get("/health", response_class=CustomJSONResponse)
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models_loaded": ml.loaded,
        "models_count": len(ml.models),
        "model_dir": MODEL_DIR,
        "db_path": DB_PATH,
        "oci_bucket": BUCKET_NAME
    }

@app.post("/analise-financeira", response_class=CustomJSONResponse)
def analisar(request: AnaliseRequest):
    try:
        trans_class = []
        resumo = {cat: 0.0 for cat in CATEGORIAS}

        for t in request.transacoes:
            cat, conf = CategoriaService.classificar(t.descricao, t.valor)
            trans_class.append({
                "descricao": t.descricao,
                "valor": round(t.valor, 2),
                "categoria": cat,
                "confianca": round(conf, 4)
            })
            resumo[cat] = resumo.get(cat, 0.0) + t.valor

        total = sum(t.valor for t in request.transacoes)
        pct_renda = round(total / request.renda_mensal * 100, 2)
        comprometimento = PerfilService._calcular_comprometimento(request.renda_mensal, total, request.nivel_endividamento)

        perfil, prob, score, explic = PerfilService.analisar(
            request.renda_mensal,
            request.nivel_endividamento,
            request.frequencia_poupanca,
            resumo,
            total
        )

        metas = request.metas or {}
        alertas = AlertaService.gerar(resumo, metas, request.renda_mensal, request.nivel_endividamento, total)
        recs = RecomendacaoService.gerar(
            perfil, resumo, request.renda_mensal,
            request.nivel_endividamento,
            request.frequencia_poupanca,
            total,
            metas
        )

        resultado = {
            "perfil_financeiro": perfil,
            "probabilidade": round(prob, 4),
            "score_saude": score,
            "resumo_gastos": {k: round(v, 2) for k, v in resumo.items()},
            "recomendacoes": recs,
            "alertas": alertas,
            "explicabilidade": explic,
            "total_gasto": round(total, 2),
            "percentual_renda_gasta": pct_renda,
            "comprometimento_gastos": comprometimento,
            "transacoes_classificadas": trans_class
        }

        salvar_historico(
            request.renda_mensal, request.nivel_endividamento, request.frequencia_poupanca,
            perfil, prob, score, total, pct_renda, comprometimento,
            resultado["resumo_gastos"], trans_class, recs, alertas, explic, metas
        )

        return resultado

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro em /analise-financeira")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simular", response_class=CustomJSONResponse)
def simular(request: SimuladorRequest):
    try:
        trans_class = []
        resumo = {cat: 0.0 for cat in CATEGORIAS}

        for t in request.transacoes:
            cat, conf = CategoriaService.classificar(t.descricao, t.valor)
            trans_class.append({
                "descricao": t.descricao,
                "valor": round(t.valor, 2),
                "categoria": cat,
                "confianca": round(conf, 4)
            })
            resumo[cat] = resumo.get(cat, 0.0) + t.valor

        total = sum(t.valor for t in request.transacoes)
        pct_renda = round(total / request.renda_mensal * 100, 2)
        comprometimento = PerfilService._calcular_comprometimento(request.renda_mensal, total, request.nivel_endividamento)

        perfil, prob, score, explic = PerfilService.analisar(
            request.renda_mensal,
            request.nivel_endividamento,
            request.frequencia_poupanca,
            resumo,
            total
        )

        metas = request.metas or {}
        alertas = AlertaService.gerar(resumo, metas, request.renda_mensal, request.nivel_endividamento, total)
        recs = RecomendacaoService.gerar(
            perfil, resumo, request.renda_mensal,
            request.nivel_endividamento,
            request.frequencia_poupanca,
            total,
            metas
        )

        return {
            "perfil_financeiro": perfil,
            "probabilidade": round(prob, 4),
            "score_saude": score,
            "resumo_gastos": {k: round(v, 2) for k, v in resumo.items()},
            "recomendacoes": recs,
            "alertas": alertas,
            "explicabilidade": explic,
            "total_gasto": round(total, 2),
            "percentual_renda_gasta": pct_renda,
            "comprometimento_gastos": comprometimento,
            "transacoes_classificadas": trans_class
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro em /simular")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classificar", response_class=CustomJSONResponse)
def classificar(descricao: str, valor: float):
    cat, conf = CategoriaService.classificar(descricao, valor)
    return {
        "descricao": descricao,
        "valor": valor,
        "categoria": cat,
        "confianca": round(conf, 4)
    }

@app.post("/analise-batch-csv", response_class=CustomJSONResponse)
async def analise_batch_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        df = pd.read_csv(StringIO(text))

        required = {"renda_mensal", "nivel_endividamento", "frequencia_poupanca", "descricao", "valor"}
        missing = required - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Colunas faltando: {missing}")

        df.columns = [c.strip().lower() for c in df.columns]
        df["frequencia_poupanca"] = df["frequencia_poupanca"].astype(str).str.strip().str.capitalize()

        grupos = df.groupby(["renda_mensal", "nivel_endividamento", "frequencia_poupanca"])
        resultados = []

        for (renda, endiv, poup), gdf in grupos:
            transacoes = [{"descricao": str(row["descricao"]), "valor": float(row["valor"])} for _, row in gdf.iterrows()]
            req = AnaliseRequest(
                renda_mensal=float(renda),
                nivel_endividamento=float(endiv),
                frequencia_poupanca=str(poup),
                transacoes=[Transacao(**t) for t in transacoes]
            )
            resumo = {cat: 0.0 for cat in CATEGORIAS}
            trans_class = []
            for t in req.transacoes:
                cat, conf = CategoriaService.classificar(t.descricao, t.valor)
                trans_class.append({"descricao": t.descricao, "valor": round(t.valor, 2), "categoria": cat, "confianca": round(conf, 4)})
                resumo[cat] = resumo.get(cat, 0.0) + t.valor

            total = sum(t.valor for t in req.transacoes)
            pct_renda = round(total / req.renda_mensal * 100, 2)
            comprometimento = PerfilService._calcular_comprometimento(req.renda_mensal, total, req.nivel_endividamento)
            perfil, prob, score, explic = PerfilService.analisar(req.renda_mensal, req.nivel_endividamento, req.frequencia_poupanca, resumo, total)
            metas = req.metas or {}
            alertas = AlertaService.gerar(resumo, metas, req.renda_mensal, req.nivel_endividamento, total)
            recs = RecomendacaoService.gerar(perfil, resumo, req.renda_mensal, req.nivel_endividamento, req.frequencia_poupanca, total, metas)

            resultados.append({
                "perfil_financeiro": perfil,
                "probabilidade": round(prob, 4),
                "score_saude": score,
                "resumo_gastos": {k: round(v, 2) for k, v in resumo.items()},
                "recomendacoes": recs,
                "alertas": alertas,
                "explicabilidade": explic,
                "total_gasto": round(total, 2),
                "percentual_renda_gasta": pct_renda,
                "comprometimento_gastos": comprometimento,
                "transacoes_classificadas": trans_class
            })

        return {"processado": len(resultados), "resultados": resultados}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro em /analise-batch-csv")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------
# Endpoints de Histórico e Relatório
# ------------------------------------------------------------------
@app.get("/historico", response_class=CustomJSONResponse)
def listar_historico(limit: int = Query(50, ge=1, le=200)):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM historico ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return {
            "total": len(rows),
            "itens": [
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "renda_mensal": r["renda_mensal"],
                    "nivel_endividamento": r["nivel_endividamento"],
                    "frequencia_poupanca": r["frequencia_poupanca"],
                    "perfil_financeiro": r["perfil_financeiro"],
                    "probabilidade": r["probabilidade"],
                    "score_saude": r["score_saude"],
                    "total_gasto": r["total_gasto"],
                    "percentual_renda_gasta": r["percentual_renda_gasta"],
                    "comprometimento_gastos": r["comprometimento_gastos"],
                    "resumo_gastos": json.loads(r["resumo_gastos"]) if r["resumo_gastos"] else {},
                    "recomendacoes": json.loads(r["recomendacoes"]) if r["recomendacoes"] else [],
                    "alertas": json.loads(r["alertas"]) if r["alertas"] else [],
                    "explicabilidade": json.loads(r["explicabilidade"]) if r["explicabilidade"] else [],
                    "metas": json.loads(r["metas"]) if r["metas"] else {}
                }
                for r in rows
            ]
        }
    except Exception as e:
        logger.exception("Erro em /historico")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/relatorio/{item_id}", response_class=CustomJSONResponse)
def relatorio(item_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM historico WHERE id = ?", (item_id,)).fetchone()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Relatório não encontrado")
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "renda_mensal": row["renda_mensal"],
            "nivel_endividamento": row["nivel_endividamento"],
            "frequencia_poupanca": row["frequencia_poupanca"],
            "perfil_financeiro": row["perfil_financeiro"],
            "probabilidade": row["probabilidade"],
            "score_saude": row["score_saude"],
            "total_gasto": row["total_gasto"],
            "percentual_renda_gasta": row["percentual_renda_gasta"],
            "comprometimento_gastos": row["comprometimento_gastos"],
            "resumo_gastos": json.loads(row["resumo_gastos"]) if row["resumo_gastos"] else {},
            "recomendacoes": json.loads(row["recomendacoes"]) if row["recomendacoes"] else [],
            "alertas": json.loads(row["alertas"]) if row["alertas"] else [],
            "explicabilidade": json.loads(row["explicabilidade"]) if row["explicabilidade"] else [],
            "transacoes_classificadas": json.loads(row["transacoes"]) if row["transacoes"] else [],
            "metas": json.loads(row["metas"]) if row["metas"] else {}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro em /relatorio")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/historico/{item_id}", response_class=CustomJSONResponse)
def deletar_historico(item_id: int):
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("DELETE FROM historico WHERE id = ?", (item_id,))
        conn.commit()
        deleted = cur.rowcount
        conn.close()
        return {"deleted": deleted}
    except Exception as e:
        logger.exception("Erro em DELETE /historico")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/historico/limpar", response_class=CustomJSONResponse)
def limpar_historico():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM historico")
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
        return {"message": "Histórico limpo com sucesso"}
    except Exception as e:
        logger.exception("Erro em POST /historico/limpar")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-models", response_class=CustomJSONResponse)
def reload_models():
    ml.reload()
    return {"models_loaded": ml.loaded, "models_count": len(ml.models)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
