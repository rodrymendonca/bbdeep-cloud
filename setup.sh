#!/bin/bash

echo "🚀 Instalando BB Deep Mobile..."

# Atualizar sistema
sudo apt-get update
sudo apt-get install -y wget unzip

# Instalar Chrome
wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt-get update
sudo apt-get install -y google-chrome-stable

# Criar diretório de dados
mkdir -p data

echo "✅ Instalação concluída!"
echo "📝 Execute: streamlit run bbdeep_mobile.py"