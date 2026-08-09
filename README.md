# 🧠 FinSight AI

<p align="center">
  <img src="https://img.shields.io/badge/version-6.0-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/fastapi-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/oci-F80000?style=flat-square&logo=oracle&logoColor=white" alt="OCI">
</p>

<p align="center">
  <b>Motor de IA que transforma dados brutos de transações em insights estratégicos em tempo real.</b>
</p>

---

## 📋 Sobre o Projeto

**FinSight AI** é um assistente inteligente de saúde financeira desenvolvido durante o **Hackathon ONE G9 Brasil 2026**. O sistema utiliza Machine Learning para classificar transações, calcular scores de saúde financeira e gerar recomendações personalizadas — tudo em tempo real.

> 🏆 **Equipe:** G9-BR-TEAM-20  
> ☁️ **Infraestrutura:** Oracle Cloud Infrastructure (OCI) Always Free  
> 🤖 **IA:** TF-IDF + Regressão Logística + Random Forest

---

## 🎯 Funcionalidades

| Recurso | Descrição |
|---------|-----------|
| 🏷️ **Categorização Automática** | Classifica despesas em 7 categorias usando TF-IDF + ML |
| 📊 **Score de Saúde Financeira** | Escala numérica de 0-1000 com explicações claras |
| 🔮 **Diagnóstico Preditivo** | Identifica perfil: Saudável, Em Observação ou Em Risco |
| 💡 **Recomendações Sob Medida** | Dicas contextualizadas baseadas no perfil e gastos |
| ⚠️ **Alertas Inteligentes** | Notificações quando metas são ultrapassadas |
| 📈 **Dashboard Visual** | Gráficos interativos com histórico e evolução |
| 🌍 **Multi-idioma** | PT / EN / ES |
| 📤 **Análise em Lote** | Upload de CSV para processamento massivo |

---

## 🏗️ Arquitetura

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Usuário   │────▶│   NGINX     │────▶│  Frontend       │
│  (Browser)  │     │  (Porta 80) │     │  (HTML/CSS/JS)  │
└─────────────┘     └─────────────┘     └─────────────────┘
                                               │
                                               ▼
                                        ┌─────────────────┐
                                        │  FastAPI        │
                                        │  (Porta 8000)   │
                                        └─────────────────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
                   │   SQLite    │    │   Modelos   │    │  OCI Object     │
                   │  (Histórico)│    │   ML (.pkl) │    │  Storage        │
                   └─────────────┘    └─────────────┘    └─────────────────┘
```

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **Frontend:** HTML5, Tailwind CSS, Chart.js, Lucide Icons
- **Machine Learning:** scikit-learn, pandas, numpy, joblib
- **Banco de Dados:** SQLite
- **Infraestrutura:** OCI VM.Standard.E2.1.Micro, Ubuntu 24.04 LTS, NGINX
- **Storage:** OCI Object Storage (modelos .pkl)

---

## 🚀 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/No-Country-simulation/G9-BR-TEAM-20.git
cd G9-BR-TEAM-20
```

### 2. Instale as dependências

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Baixe os modelos do OCI Object Storage

> ⚠️ **Importante:** Esse passo só funciona dentro da VM da OCI (autenticação automática via Dynamic Group).

```bash
oci os object get --bucket-name hackathon-one-g9-team-20   --name models_vetorizador_tfidf.pkl --file models/vetorizador_tfidf.pkl
oci os object get --bucket-name hackathon-one-g9-team-20   --name models_modelo_categoria_producao.pkl --file models/modelo_categoria_producao.pkl
oci os object get --bucket-name hackathon-one-g9-team-20   --name models_codificador_categorias.pkl --file models/codificador_categorias.pkl
oci os object get --bucket-name hackathon-one-g9-team-20   --name models_modelo_perfil_producao.pkl --file models/modelo_perfil_producao.pkl
oci os object get --bucket-name hackathon-one-g9-team-20   --name models_codificador_perfil.pkl --file models/codificador_perfil.pkl
```

### 4. Inicie a aplicação

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 5. Acesse

- 🌐 **Frontend:** `http://SEU_IP_OCI`
- 📚 **API Docs (Swagger):** `http://SEU_IP_OCI/docs`
- 💓 **Health Check:** `http://SEU_IP_OCI/health`

---

## 📡 API Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/health` | Verifica status do sistema |
| `POST` | `/analise-financeira` | Análise financeira completa com ML |
| `POST` | `/simular` | Simula cenário sem salvar no histórico |
| `GET` | `/classificar` | Classifica uma única transação |
| `POST` | `/analise-batch-csv` | Análise em lote via CSV |
| `GET` | `/historico` | Lista análises anteriores |
| `GET` | `/relatorio/{id}` | Obtém relatório específico |
| `DELETE` | `/historico/{id}` | Remove análise do histórico |
| `POST` | `/reload-models` | Recarrega modelos sem reiniciar |

### Exemplo de requisição

```bash
curl -X POST http://SEU_IP_OCI/analise-financeira   -H "Content-Type: application/json"   -d '{
    "renda_mensal": 4500,
    "nivel_endividamento": 25,
    "frequencia_poupanca": "Media",
    "transacoes": [
      {"descricao": "Supermercado", "valor": 420},
      {"descricao": "Combustivel", "valor": 300}
    ],
    "metas": {"alimentacao": 600, "transporte": 400}
  }'
```

---

## 📊 Modelos de Machine Learning

| Modelo | Algoritmo | Performance |
|--------|-----------|-------------|
| **Classificação de Categorias** | Regressão Logística Multinomial + TF-IDF | ~86,8% acurácia |
| **Classificação de Perfil** | Random Forest (200 estimadores) | 5.000 perfis sintéticos treinados |

> 💡 **Fallback Heurístico:** Se os modelos não estiverem disponíveis, o sistema ativa regras de negócio inteligentes para garantir que a análise nunca pare.

---

## 👥 Equipe G9-BR-TEAM-20

| Nome | LinkedIn | GitHub | Discord |
|------|----------|--------|---------|
| **Fabio C. Zinetti** | [LinkedIn](https://www.linkedin.com/in/fabiozinetti) | [GitHub](https://github.com/fabinhoz) | @fabinhoz |
| **João P. R. Deodato** | [LinkedIn](https://www.linkedin.com/in/jpdeodato) | [GitHub](https://github.com/jpdeodato) | @jp_deodato |
| **Edson H. F. da Silva** | [LinkedIn](https://www.linkedin.com/in/henriquesilvatech) | [GitHub](https://github.com/86HenriqueSilva) | @henrique.silva2916 |
| **Luciano R. da Silva** | [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/) | @siilverado |
| **Rodrigo M. Veiga** | [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/) | @rodrigoveiga93 |
| **André N. Xavier** | [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/) | @andrepgupgrade |

---

## 🌐 Onde Aplica

- 🏦 **Bancos & Fintechs** — Credit scoring, motores de recomendação, análise de risco
- 🏢 **Bem-estar Corporativo (RH)** — Plataformas de saúde financeira do colaborador
- 🛒 **Super Apps & Varejo** — Carteiras digitais com inteligência preditiva de consumo

---

<p align="center">
  <sub>FinSight AI v6.0 — Hackathon OCI 2026 🚀</sub>
</p>
