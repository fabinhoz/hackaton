#!/bin/bash
# ============================================================
# FinSight AI — SCRIPT DE ATUALIZAÇÃO AUTOMÁTICA v4.0
# G9-BR-TEAM-20 | Hackathon OCI
# Executa backup, instala dependências, atualiza arquivos,
# gerencia SWAP, configura HTTPS (opcional) e reinicia serviços
# ============================================================

set -e

BACKEND_DIR="/home/ubuntu/finsight-ai/backend"
FRONTEND_DIR="/var/www/finsight"
MODEL_DIR="/home/ubuntu/finsight-ai/models"
DATA_DIR="/home/ubuntu/finsight-ai/data"
ROLLBACK_SCRIPT="/home/ubuntu/RESTAURAR_VERSAO_ATUAL.sh"
VENV_DIR="/home/ubuntu/finsight-ai/venv"
DOMAIN=""  # Preencha se tiver domínio para HTTPS

echo "========================================"
echo "  FinSight AI — Atualização v4.0"
echo "  G9-BR-TEAM-20 | Hackathon OCI"
echo "========================================"
echo ""

# --------------------------------------------------
# 0. GERENCIAR SWAP (OOM Prevention)
# --------------------------------------------------
echo "[0/10] Verificando SWAP..."
if ! swapon --show | grep -q "swap"; then
    echo "  SWAP não encontrado. Criando arquivo de 2GB..."
    sudo fallocate -l 2G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
    echo "  SWAP ativado."
else
    echo "  SWAP já ativo."
fi
echo ""

# --------------------------------------------------
# 1. BACKUP OBRIGATÓRIO
# --------------------------------------------------
echo "[1/10] Criando backup de segurança..."
if [ -f "$ROLLBACK_SCRIPT" ]; then
    bash "$ROLLBACK_SCRIPT" backup
else
    echo "  Script de rollback não encontrado. Criando backup manual..."
    BACKUP_DIR="/home/ubuntu/finsight-ai/.backup_rollback/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp "$BACKEND_DIR/main.py" "$BACKUP_DIR/" 2>/dev/null || true
    cp "$FRONTEND_DIR/index.html" "$BACKUP_DIR/" 2>/dev/null || true
    echo "  Backup manual em: $BACKUP_DIR"
fi
echo ""

# --------------------------------------------------
# 2. DEPENDÊNCIAS
# --------------------------------------------------
echo "[2/10] Verificando/instalando dependências..."
if [ ! -d "$VENV_DIR" ]; then
    echo "  Criando virtualenv..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install -q --upgrade pip
pip install -q joblib pandas oci fastapi uvicorn pydantic scikit-learn numpy 2>&1 | grep -v "already satisfied" || true

echo "  Dependências instaladas."
echo ""

# --------------------------------------------------
# 3. ATUALIZAR BACKEND
# --------------------------------------------------
echo "[3/10] Atualizando backend..."
if [ -f "/home/ubuntu/main.py" ]; then
    cp "/home/ubuntu/main.py" "$BACKEND_DIR/main.py"
    echo "  main.py atualizado (de /home/ubuntu/)"
elif [ -f "$BACKEND_DIR/main.py.new" ]; then
    cp "$BACKEND_DIR/main.py.new" "$BACKEND_DIR/main.py"
    echo "  main.py atualizado (de .new)"
else
    echo "  main.py novo não encontrado em /home/ubuntu/ nem em $BACKEND_DIR/main.py.new"
    echo "      Envie o arquivo antes de executar este script."
    exit 1
fi
echo ""

# --------------------------------------------------
# 4. ATUALIZAR FRONTEND
# --------------------------------------------------
echo "[4/10] Atualizando frontend..."
if [ -f "/home/ubuntu/index.html" ]; then
    sudo cp "/home/ubuntu/index.html" "$FRONTEND_DIR/index.html"
    sudo chown www-data:www-data "$FRONTEND_DIR/index.html"
    echo "  index.html atualizado (de /home/ubuntu/)"
elif [ -f "$FRONTEND_DIR/index.html.new" ]; then
    sudo cp "$FRONTEND_DIR/index.html.new" "$FRONTEND_DIR/index.html"
    echo "  index.html atualizado (de .new)"
else
    echo "  index.html novo não encontrado. Pulando frontend."
fi
echo ""

# --------------------------------------------------
# 5. VERIFICAR MODELOS
# --------------------------------------------------
echo "[5/10] Verificando artefatos de modelo..."
MODELOS=(
    "vetorizador_tfidf.pkl"
    "modelo_categoria_producao.pkl"
    "codificador_categorias.pkl"
    "modelo_perfil_producao.pkl"
    "codificador_perfil.pkl"
)

FALTANDO=0
for modelo in "${MODELOS[@]}"; do
    if [ ! -f "$MODEL_DIR/$modelo" ]; then
        echo "  Modelo não encontrado localmente: $modelo"
        FALTANDO=$((FALTANDO + 1))
    else
        echo "  OK: $modelo"
    fi
done

if [ $FALTANDO -gt 0 ]; then
    echo ""
    echo "  $FALTANDO modelo(s) faltando localmente."
    echo "  O backend tentará sincronizar do OCI Object Storage na inicialização."
fi
echo ""

# --------------------------------------------------
# 6. CONFIGURAR SYSTEMD
# --------------------------------------------------
echo "[6/10] Atualizando serviço systemd..."

sudo bash -c "cat > /etc/systemd/system/finsight.service" << 'SYSTEMDEOF'
[Unit]
Description=FinSight AI API v4.0 - G9-BR-TEAM-20
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/finsight-ai/backend
Environment="PATH=/home/ubuntu/finsight-ai/venv/bin"
Environment="API_PORT=8000"
Environment="API_HOST=0.0.0.0"
Environment="MODEL_DIR=/home/ubuntu/finsight-ai/models"
Environment="DATA_DIR=/home/ubuntu/finsight-ai/data"
Environment="OCI_BUCKET_NAME=hackathon-one-g9-team-20"
ExecStart=/home/ubuntu/finsight-ai/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SYSTEMDEOF

echo "  Service systemd atualizado"
echo ""

# --------------------------------------------------
# 7. CONFIGURAR NGINX (HTTP + HTTPS opcional)
# --------------------------------------------------
echo "[7/10] Configurando nginx..."

if [ -n "$DOMAIN" ]; then
    # HTTPS com Let's Encrypt
    sudo apt-get update -qq
    sudo apt-get install -y -qq certbot python3-certbot-nginx

    sudo bash -c "cat > /etc/nginx/sites-available/finsight" << NGINXEOF
server {
    listen 80;
    server_name $DOMAIN;
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}
server {
    listen 443 ssl;
    server_name $DOMAIN;

    ssl_certificate /etc/letsencrypt/live/$DOMAIN/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$DOMAIN/privkey.pem;

    location / {
        root $FRONTEND_DIR;
        try_files \$uri /index.html;
    }
    location /docs {
        proxy_pass http://localhost:8000/docs;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    location /openapi.json {
        proxy_pass http://localhost:8000/openapi.json;
        proxy_set_header Host \$host;
    }
    location /health {
        proxy_pass http://localhost:8000/health;
        proxy_set_header Host \$host;
    }
    location /analise-financeira {
        proxy_pass http://localhost:8000/analise-financeira;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    location /classificar {
        proxy_pass http://localhost:8000/classificar;
        proxy_set_header Host \$host;
    }
    location /historico {
        proxy_pass http://localhost:8000/historico;
        proxy_set_header Host \$host;
    }
    location /reload-models {
        proxy_pass http://localhost:8000/reload-models;
        proxy_set_header Host \$host;
    }
}
NGINXEOF
    sudo ln -sf /etc/nginx/sites-available/finsight /etc/nginx/sites-enabled/finsight
    sudo rm -f /etc/nginx/sites-enabled/default

    if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
        echo "  Solicitando certificado SSL para $DOMAIN..."
        sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "admin@$DOMAIN" || true
    fi
else
    # HTTP apenas
    sudo bash -c "cat > /etc/nginx/sites-available/finsight" << 'NGINXEOF'
server {
    listen 80;
    server_name _;

    location / {
        root /var/www/finsight;
        try_files $uri /index.html;
    }
    location /docs {
        proxy_pass http://localhost:8000/docs;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    location /openapi.json {
        proxy_pass http://localhost:8000/openapi.json;
        proxy_set_header Host $host;
    }
    location /health {
        proxy_pass http://localhost:8000/health;
        proxy_set_header Host $host;
    }
    location /analise-financeira {
        proxy_pass http://localhost:8000/analise-financeira;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    location /classificar {
        proxy_pass http://localhost:8000/classificar;
        proxy_set_header Host $host;
    }
    location /historico {
        proxy_pass http://localhost:8000/historico;
        proxy_set_header Host $host;
    }
    location /reload-models {
        proxy_pass http://localhost:8000/reload-models;
        proxy_set_header Host $host;
    }
}
NGINXEOF
    sudo ln -sf /etc/nginx/sites-available/finsight /etc/nginx/sites-enabled/finsight
    sudo rm -f /etc/nginx/sites-enabled/default
fi

echo "  Nginx configurado"
echo ""

# --------------------------------------------------
# 8. REINICIAR SERVIÇOS
# --------------------------------------------------
echo "[8/10] Reiniciando serviços..."
sudo systemctl daemon-reload
sudo systemctl restart finsight
sudo systemctl reload nginx
sleep 2
echo "  Serviços reiniciados"
echo ""

# --------------------------------------------------
# 9. VERIFICAÇÃO FINAL
# --------------------------------------------------
echo "[9/10] Verificação final..."

HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
if [ "$HEALTH" == "200" ]; then
    echo "  Health check: HTTP 200 OK"
    echo ""
    echo "========================================"
    echo "  ATUALIZAÇÃO v4.0 CONCLUÍDA!"
    echo "========================================"
    echo ""
    IP_PUBLICO=$(curl -s ifconfig.me 2>/dev/null || echo "SEU_IP")
    echo "Acesse: http://$IP_PUBLICO"
    echo "API Docs: http://$IP_PUBLICO/docs"
    echo "Health: http://$IP_PUBLICO/health"
    echo ""
    echo "Se algo estiver errado, execute o rollback:"
    echo "  $ROLLBACK_SCRIPT restore"
else
    echo "  Health check falhou (HTTP $HEALTH)"
    echo ""
    echo "Verifique os logs:"
    echo "  sudo journalctl -u finsight -f"
    echo ""
    echo "Para restaurar a versão anterior:"
    echo "  $ROLLBACK_SCRIPT restore"
    exit 1
fi

# --------------------------------------------------
# 10. STATUS DA MEMÓRIA
# --------------------------------------------------
echo "[10/10] Status de memória:"
free -h
echo ""
echo "========================================"
echo "  FinSight AI v4.0 está no ar!"
echo "========================================"
