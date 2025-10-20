# 🤖 BB Deep Mobile - Scraping Real Bettilt

Sistema avançado de scraping e previsão para o jogo Bac Bo no Bettilt, com machine learning e interface Streamlit.

## 🚀 Funcionalidades

- ✅ **Scraping Real**: Conexão automática ao Bettilt
- ✅ **Login Automático**: Sistema seguro de autenticação
- ✅ **3 Modelos ML**: Random Forest, SVM e LSTM
- ✅ **Previsões em Tempo Real**: Análise probabilística
- ✅ **Interface Mobile**: Design responsivo
- ✅ **Auto-scraping**: Atualização automática a cada X minutos
- ✅ **ChromeDriver Automático**: Instalação e configuração automática

## 📦 Instalação

### Método 1: Streamlit Cloud (Recomendado)
1. Faça fork deste repositório
2. Acesse [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecte seu GitHub e deploy a app

### Método 2: Local
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/bbdeep-mobile.git
cd bbdeep-mobile

# Execute o setup (Linux/Mac)
chmod +x setup.sh
./setup.sh

# Ou instale manualmente
pip install -r requirements.txt
python chromedriver_install.py

# Execute a app
streamlit run bbdeep_mobile.py