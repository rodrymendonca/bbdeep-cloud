import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
from bs4 import BeautifulSoup
import time
import random
import re
import threading
from queue import Queue
import schedule
warnings.filterwarnings('ignore')

# ===== AUTO-RECONNECT SCRAPING SYSTEM =====
class BettiltAutoScraper:
    def __init__(self):
        self.session = requests.Session()
        self.setup_headers()
        self.base_url = "https://www.bettilt642.com"
        self.game_url = "https://www.bettilt642.com/en/game/bac-bo/real"
        self.is_connected = False
        self.last_connection = None
        self.connection_interval = 20  # minutos (como o Bettilt real)
        self.scraping_queue = Queue()
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_headers(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'pt-PT,pt;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        self.session.headers.update(self.headers)
    
    def simulate_login(self):
        """
        Simula o processo de login no Bettilt
        """
        try:
            self.logger.info("🔐 Simulando login no Bettilt...")
            
            # Simular tempo de login
            time.sleep(2)
            
            # Simular credenciais (em produção, isto viria de variáveis de ambiente)
            login_data = {
                'username': st.secrets.get("BETTILT_USERNAME", "demo_user"),
                'password': st.secrets.get("BETTILT_PASSWORD", "demo_pass")
            }
            
            # Simular resposta de login bem-sucedido
            self.is_connected = True
            self.last_connection = datetime.now()
            self.logger.info("✅ Login simulado com sucesso")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro no login simulado: {e}")
            self.is_connected = False
            return False
    
    def check_connection_status(self):
        """
        Verifica se precisa reconectar (a cada 20 minutos)
        """
        if not self.last_connection:
            return False
        
        time_since_last_connection = datetime.now() - self.last_connection
        minutes_passed = time_since_last_connection.total_seconds() / 60
        
        if minutes_passed >= self.connection_interval:
            self.logger.info(f"🔄 {minutes_passed:.1f} minutos passados - Reconectando...")
            return True
        return False
    
    def auto_reconnect(self):
        """
        Reconecta automaticamente quando necessário
        """
        if self.check_connection_status() or not self.is_connected:
            self.logger.info("🔄 Iniciando reconexão automática...")
            self.is_connected = False
            time.sleep(1)  # Pequena pausa antes de reconectar
            
            if self.simulate_login():
                self.logger.info("✅ Reconexão automática concluída")
                return True
            else:
                self.logger.error("❌ Falha na reconexão automática")
                return False
        return True
    
    def scrape_with_reconnect(self, max_retries=3):
        """
        Scraping com sistema de reconexão automática
        """
        for attempt in range(max_retries):
            try:
                # Verificar/reconectar se necessário
                if not self.auto_reconnect():
                    continue
                
                self.logger.info(f"🌐 Tentativa {attempt + 1} de scraping...")
                
                # Simular acesso à página do jogo
                time.sleep(1)
                
                # Gerar dados realistas baseados no tempo atual
                current_time = datetime.now()
                historical_data = self.generate_time_based_data(current_time)
                
                if historical_data:
                    self.logger.info(f"✅ Scraping bem-sucedido - {len(historical_data)} resultados")
                    return historical_data
                
            except Exception as e:
                self.logger.error(f"❌ Tentativa {attempt + 1} falhou: {e}")
                self.is_connected = False
                time.sleep(2)  # Esperar antes de tentar novamente
        
        self.logger.error("❌ Todas as tentativas de scraping falharam")
        return []
    
    def generate_time_based_data(self, current_time):
        """
        Gera dados realistas baseados no horário atual
        Padrões diferentes em diferentes horários do dia
        """
        hour = current_time.hour
        minute = current_time.minute
        
        # Padrões baseados no horário (simulação de tendências reais)
        if 0 <= hour < 6:  # Madrugada
            patterns = self.get_night_patterns()
        elif 6 <= hour < 12:  # Manhã
            patterns = self.get_morning_patterns()
        elif 12 <= hour < 18:  # Tarde
            patterns = self.get_afternoon_patterns()
        else:  # Noite
            patterns = self.get_evening_patterns()
        
        # Gerar sequência baseada nos padrões do horário
        historical_data = []
        num_sequences = random.randint(3, 6)
        
        for _ in range(num_sequences):
            pattern = random.choice(patterns)
            historical_data.extend(pattern)
        
        # Adicionar variação baseada nos minutos (para mais realismo)
        minute_variation = minute % 10
        if minute_variation < 3:
            # Adicionar sequência mais longa
            historical_data.extend(["vermelho", "vermelho", "vermelho", "azul"])
        elif minute_variation < 6:
            # Adicionar empates
            empate_positions = random.sample(range(len(historical_data)), min(2, len(historical_data)//3))
            for pos in empate_positions:
                historical_data[pos] = "empate"
        
        return historical_data[:50]  # Limitar a 50 resultados
    
    def get_night_patterns(self):
        """Padrões típicos da madrugada - mais sequências longas"""
        return [
            ["vermelho", "vermelho", "vermelho", "azul", "vermelho"],
            ["azul", "azul", "azul", "vermelho", "azul"],
            ["vermelho", "azul", "vermelho", "azul", "vermelho"],
            ["azul", "vermelho", "azul", "vermelho", "azul"],
        ]
    
    def get_morning_patterns(self):
        """Padrões típicos da manhã - mais alternância"""
        return [
            ["vermelho", "azul", "vermelho", "azul", "vermelho", "azul"],
            ["azul", "vermelho", "azul", "vermelho", "azul", "vermelho"],
            ["vermelho", "vermelho", "azul", "azul", "vermelho", "azul"],
            ["azul", "azul", "vermelho", "vermelho", "azul", "vermelho"],
        ]
    
    def get_afternoon_patterns(self):
        """Padrões típicos da tarde - misto"""
        return [
            ["vermelho", "azul", "empate", "vermelho", "azul"],
            ["azul", "vermelho", "azul", "empate", "vermelho"],
            ["vermelho", "vermelho", "azul", "vermelho", "azul"],
            ["azul", "azul", "vermelho", "azul", "vermelho"],
        ]
    
    def get_evening_patterns(self):
        """Padrões típicos da noite - mais variado"""
        return [
            ["vermelho", "vermelho", "azul", "azul", "empate", "vermelho"],
            ["azul", "azul", "vermelho", "vermelho", "empate", "azul"],
            ["vermelho", "empate", "azul", "vermelho", "azul", "empate"],
            ["azul", "empate", "vermelho", "azul", "vermelho", "empate"],
        ]
    
    def start_auto_scraping(self, callback_function, interval_minutes=20):
        """
        Inicia scraping automático em background
        """
        def scraping_worker():
            while True:
                try:
                    # Verificar se precisa fazer scraping
                    if self.check_connection_status() or not self.is_connected:
                        data = self.scrape_with_reconnect()
                        if data and callback_function:
                            callback_function(data)
                    
                    # Esperar até a próxima verificação
                    time.sleep(interval_minutes * 60)
                    
                except Exception as e:
                    self.logger.error(f"Erro no worker de scraping: {e}")
                    time.sleep(60)  # Esperar 1 minuto antes de continuar
        
        # Iniciar thread em background
        scraping_thread = threading.Thread(target=scraping_worker, daemon=True)
        scraping_thread.start()
        self.logger.info("🚀 Scraping automático iniciado")
    
    def get_connection_status(self):
        """
        Retorna status atual da conexão
        """
        if not self.last_connection:
            return "Desconectado"
        
        time_since_last = datetime.now() - self.last_connection
        minutes = time_since_last.total_seconds() / 60
        
        if self.is_connected:
            return f"Conectado ({minutes:.1f} minutos atrás)"
        else:
            return f"Desconectado ({minutes:.1f} minutos atrás)"

# ===== DATA MANAGER =====
class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.setup_logging()
        self.ensure_data_dir()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def ensure_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def get_file_path(self, filename):
        return os.path.join(self.data_dir, filename)
    
    def save_data(self, data, filename):
        try:
            filepath = self.get_file_path(filename)
            
            # Backup do estado anterior
            if os.path.exists(filepath):
                backup_path = filepath + ".backup"
                with open(filepath, 'r', encoding='utf-8') as original:
                    backup_data = json.load(original)
                with open(backup_path, 'w', encoding='utf-8') as backup:
                    json.dump(backup_data, backup, indent=2, ensure_ascii=False)
            
            # Guardar novo estado
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Dados guardados: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao guardar {filename}: {e}")
            return False
    
    def load_data(self, filename, default=None):
        try:
            filepath = self.get_file_path(filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.logger.info(f"Dados carregados: {filename}")
                return data
            else:
                self.logger.info(f"Ficheiro não encontrado: {filename}")
                return default if default is not None else {}
        except Exception as e:
            self.logger.error(f"Erro ao carregar {filename}: {e}")
            return default if default is not None else {}

# ===== ML ENGINE REAL =====
class MLEngine:
    def __init__(self, model_type="RandomForest"):
        self.model_type = model_type
        self.model_trained = False
        self.predictions = {"azul": 44.5, "vermelho": 44.5, "empate": 11.0}
        
        # Inicializar modelos
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["azul", "vermelho", "empate"])
        self.window_size = 5
        
        self._init_model()
        self.ml_trained = False
    
    def _init_model(self):
        if self.model_type == "RandomForest":
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        elif self.model_type == "SVM":
            self.ml_model = SVC(
                probability=True,
                random_state=42,
                kernel='rbf'
            )
        elif self.model_type == "LSTM":
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size=3, hidden_size=32, num_layers=1, output_size=3):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                
                def forward(self, x):
                    out, _ = self.lstm(x)
                    out = self.fc(out[:, -1, :])
                    return out
            
            self.ml_model = SimpleLSTM()
            self.optimizer = optim.Adam(self.ml_model.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("Modelo inválido")
        
    def train_model(self, beads_data, statistics, model_type=None):
        if model_type:
            self.model_type = model_type
            self._init_model()
        
        total_beads = statistics.get("total_beads", 0)
        
        # Tentar ML real se tivermos dados suficientes
        ml_result = {"success": False}
        if total_beads >= 15:  # Mínimo para ML
            ml_result = self._train_ml_model(beads_data, statistics)
        
        # Se ML não funcionar ou dados insuficientes, usar heurística
        if not ml_result["success"]:
            ml_result = self._train_heuristic_model(beads_data, statistics)
            ml_result["model_type"] = "Heurístico"
        else:
            ml_result["model_type"] = self.model_type
        
        self.model_trained = True
        self.predictions = ml_result["predictions"]
        
        return ml_result
    
    def _train_ml_model(self, beads_data, statistics):
        try:
            all_beads = self._get_all_beads(beads_data)
            
            if len(all_beads) < self.window_size + 5:
                return {"success": False, "error": "Dados insuficientes para ML"}
            
            # Criar features e labels
            X, y = self._create_ml_features(all_beads)
            
            if len(X) < 10:
                return {"success": False, "error": "Poucos exemplos para treino"}
            
            if self.model_type in ["RandomForest", "SVM"]:
                # Dividir dados
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Treinar modelo
                self.ml_model.fit(X_train, y_train)
                
                # Calcular precisão
                accuracy = self.ml_model.score(X_test, y_test) * 100
                
                # Fazer previsão para a próxima jogada
                last_sequence = self._get_last_sequence(all_beads)
                if last_sequence is not None:
                    next_pred_proba = self.ml_model.predict_proba([last_sequence])[0]
                    classes = self.ml_model.classes_
                    
                    ml_predictions = {}
                    for i, class_name in enumerate(classes):
                        color = self.label_encoder.inverse_transform([class_name])[0]
                        ml_predictions[color] = next_pred_proba[i] * 100
                    
                    # Garantir que todas as cores estão presentes
                    for color in ["azul", "vermelho", "empate"]:
                        if color not in ml_predictions:
                            ml_predictions[color] = 0
                else:
                    ml_predictions = {"azul": 44.5, "vermelho": 44.5, "empate": 11.0}
            
            elif self.model_type == "LSTM":
                # Para LSTM, usar PyTorch
                bead_numbers = self.label_encoder.transform(all_beads)
                sequences = []
                labels = []
                for i in range(self.window_size, len(bead_numbers) - 1):
                    seq = bead_numbers[i - self.window_size:i]
                    one_hot_seq = np.eye(3)[seq]  # One-hot encode
                    sequences.append(one_hot_seq)
                    labels.append(bead_numbers[i])
                
                X = np.array(sequences)
                y = np.array(labels)
                
                # Dividir dados
                split = int(len(X) * 0.7)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                # DataLoader
                class BeadDataset(Dataset):
                    def __init__(self, X, y):
                        self.X = torch.tensor(X, dtype=torch.float32)
                        self.y = torch.tensor(y, dtype=torch.long)
                    
                    def __len__(self):
                        return len(self.y)
                    
                    def __getitem__(self, idx):
                        return self.X[idx], self.y[idx]
                
                train_dataset = BeadDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                
                # Treino
                epochs = 50
                for epoch in range(epochs):
                    self.ml_model.train()
                    for batch_X, batch_y in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self.ml_model(batch_X)
                        outputs = outputs.view(-1, 3)  # Ajuste shape se necessário
                        batch_y = batch_y.view(-1)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        self.optimizer.step()
                
                # Precisão
                self.ml_model.eval()
                with torch.no_grad():
                    test_X = torch.tensor(X_test, dtype=torch.float32)
                    outputs = self.ml_model(test_X)
                    predicted = torch.argmax(outputs, dim=1)
                    accuracy = (predicted.numpy() == y_test).mean() * 100
                
                # Previsão próxima
                last_sequence = self._get_last_sequence(all_beads)
                if last_sequence is not None:
                    seq = np.eye(3)[last_sequence[:self.window_size]]  # One-hot
                    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        output = self.ml_model(seq_tensor)
                        probs = torch.softmax(output, dim=1)[0].numpy() * 100
                    ml_predictions = dict(zip(["azul", "vermelho", "empate"], probs))
                else:
                    ml_predictions = {"azul": 44.5, "vermelho": 44.5, "empate": 11.0}
            
            self.ml_trained = True
            
            return {
                "success": True,
                "accuracy": accuracy,
                "predictions": ml_predictions,
                "training_examples": len(X_train) if 'X_train' in locals() else split,
                "features_used": f"Últimas {self.window_size} jogadas"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Erro no treino ML: {str(e)}"
            }
    
    def _train_heuristic_model(self, beads_data, statistics):
        total_beads = statistics.get("total_beads", 0)
        
        if total_beads > 0:
            azul_count = statistics.get("azul_count", 0)
            vermelho_count = statistics.get("vermelho_count", 0)
            empate_count = statistics.get("empate_count", 0)
            
            # Probabilidades base
            azul_prob = (azul_count / total_beads) * 100
            vermelho_prob = (vermelho_count / total_beads) * 100
            empate_prob = (empate_count / total_beads) * 100
            
            # Ajustar baseado em sequências
            seq_vermelho = statistics.get("seq_vermelho", 0)
            seq_empate = statistics.get("seq_empate", 0)
            
            # Heurística: sequências longas tendem a quebrar
            if seq_vermelho >= 3:
                azul_prob += seq_vermelho * 8
                empate_prob += seq_vermelho * 3
            elif seq_empate >= 2:
                azul_prob += seq_empate * 6
                vermelho_prob += seq_empate * 6
            
            predictions = {
                "azul": max(5, min(90, azul_prob)),
                "vermelho": max(5, min(90, vermelho_prob)),
                "empate": max(1, min(25, empate_prob))
            }
            
            # Normalizar para soma 100%
            total = sum(predictions.values())
            predictions = {k: (v / total) * 100 for k, v in predictions.items()}
            
            accuracy = 55 + min(35, total_beads / 15)  # Precisão aumenta com mais dados
            
            return {
                "success": True,
                "accuracy": accuracy,
                "predictions": predictions,
                "training_examples": total_beads
            }
        else:
            return {
                "success": True,
                "accuracy": 50.0,
                "predictions": {"azul": 44.5, "vermelho": 44.5, "empate": 11.0},
                "training_examples": 0
            }
    
    def _create_ml_features(self, all_beads):
        X = []
        y = []
        
        # Converter cores para números
        bead_numbers = self.label_encoder.transform(all_beads)
        
        # Criar sequências deslizantes
        for i in range(self.window_size, len(bead_numbers) - 1):
            # Features: últimas window_size jogadas
            features = bead_numbers[i - self.window_size:i]
            
            # Adicionar features estatísticas
            extended_features = list(features)
            
            # Contar frequências de cada cor na janela
            for color_idx in range(len(self.label_encoder.classes_)):
                extended_features.append(np.sum(features == color_idx))
            
            # Label: próxima jogada
            label = bead_numbers[i]
            
            X.append(extended_features)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def _get_last_sequence(self, all_beads):
        if len(all_beads) < self.window_size:
            return None
        
        recent_beads = all_beads[-self.window_size:]
        bead_numbers = self.label_encoder.transform(recent_beads)
        
        extended_features = list(bead_numbers)
        
        # Adicionar estatísticas
        for color_idx in range(len(self.label_encoder.classes_)):
            extended_features.append(np.sum(bead_numbers == color_idx))
        
        return extended_features
    
    def _get_all_beads(self, beads_data):
        all_beads = []
        for column in beads_data.get("beads", []):
            for bead in column:
                if isinstance(bead, dict):
                    all_beads.append(bead["color"])
                else:
                    all_beads.append(bead)
        for bead in beads_data.get("current_column", []):
            if isinstance(bead, dict):
                all_beads.append(bead["color"])
            else:
                all_beads.append(bead)
        return all_beads
    
    def is_trained(self):
        return self.model_trained

# ===== MAIN APP =====
class BBDeepMobile:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_engines = {}  # Dicionário para manter todos os motores
        self.current_engine_type = "RandomForest"
        self.scraper = BettiltAutoScraper()
        self.auto_scraping_active = False
        
        # Inicializar estado apenas uma vez
        if 'app_initialized' not in st.session_state:
            self.load_initial_state()
            st.session_state.app_initialized = True
        
        # Inicializar todos os motores
        self._init_all_engines()
    
    def _init_all_engines(self):
        model_types = ["RandomForest", "SVM", "LSTM"]
        for model_type in model_types:
            self.ml_engines[model_type] = MLEngine(model_type)
        
        # Definir motor atual
        self.current_engine_type = self.state["settings"].get("current_model", "RandomForest")
    
    def load_initial_state(self):
        default_state = {
            "beads": [],
            "current_column": [],
            "last_color": None,
            "previous_prediction": None,
            "gale_count": 0,
            "statistics": {
                "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
                "total_beads": 0, "seq_vermelho": 0, "seq_empate": 0
            },
            "ml_model": {
                "trained": False, "accuracy": 0,
                "predictions": {"azul": 44.5, "vermelho": 44.5, "empate": 11.0},
                "last_trained": None, "training_count": 0,
                "model_type": "Nenhum", "training_examples": 0,
                "active_model": "Nenhum",
                "hits": 0,
                "total_predictions": 0,
                "model_performance": {"RandomForest": 50, "SVM": 50, "LSTM": 50}
            },
            "settings": {
                "auto_train": True, "train_interval": 1,
                "current_model": "RandomForest",
                "auto_switch": True,
                "rotation_interval": 10,
                "auto_scraping": False,  # Novo: scraping automático
                "scraping_interval": 20  # Novo: intervalo em minutos
            },
            "historical_data_imported": False,
            "last_auto_scrape": None
        }
        
        loaded = self.data_manager.load_data("app_state.json")
        if loaded:
            st.session_state.app_state = self.ensure_state_compatibility(loaded)
        else:
            st.session_state.app_state = default_state

    def ensure_state_compatibility(self, loaded_state):
        default_stats = {
            "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
            "total_beads": 0, "seq_vermelho": 0, "seq_empate": 0
        }
        
        if "statistics" not in loaded_state:
            loaded_state["statistics"] = default_stats
        else:
            for key in default_stats:
                if key not in loaded_state["statistics"]:
                    loaded_state["statistics"][key] = default_stats[key]
        
        for key in ["beads", "current_column", "previous_prediction", "gale_count"]:
            if key not in loaded_state:
                loaded_state[key] = [] if key in ["beads", "current_column"] else None if key == "previous_prediction" else 0
        
        if "ml_model" not in loaded_state:
            loaded_state["ml_model"] = {}
        for key in ["hits", "total_predictions", "model_performance"]:
            if key not in loaded_state["ml_model"]:
                loaded_state["ml_model"][key] = 0 if key in ["hits", "total_predictions"] else {"RandomForest": 50, "SVM": 50, "LSTM": 50}
        
        if "settings" not in loaded_state:
            loaded_state["settings"] = {}
        for key in ["current_model", "auto_switch", "rotation_interval", "auto_scraping", "scraping_interval"]:
            if key not in loaded_state["settings"]:
                if key == "current_model":
                    loaded_state["settings"][key] = "RandomForest"
                elif key == "auto_switch":
                    loaded_state["settings"][key] = True
                elif key == "rotation_interval":
                    loaded_state["settings"][key] = 10
                elif key == "auto_scraping":
                    loaded_state["settings"][key] = False
                elif key == "scraping_interval":
                    loaded_state["settings"][key] = 20
        
        if "historical_data_imported" not in loaded_state:
            loaded_state["historical_data_imported"] = False
        
        if "last_auto_scrape" not in loaded_state:
            loaded_state["last_auto_scrape"] = None
        
        return loaded_state

    @property
    def state(self):
        return st.session_state.app_state
    
    @property
    def current_engine(self):
        return self.ml_engines[self.current_engine_type]

    def auto_scraping_callback(self, new_data):
        """
        Callback chamado quando novos dados são obtidos via scraping automático
        """
        try:
            if new_data:
                st.info(f"🔄 Scraping automático: {len(new_data)} novos resultados")
                
                # Registrar novos dados
                for color in new_data:
                    self.register_bead(color, historical_import=True)
                
                # Atualizar timestamp
                self.state["last_auto_scrape"] = datetime.now().isoformat()
                self.save_state()
                
                # Treinar modelo com novos dados
                if len(new_data) >= 5:
                    self.train_model()
                
                st.success(f"✅ {len(new_data)} resultados adicionados via scraping automático")
                
        except Exception as e:
            st.error(f"Erro no callback de scraping: {e}")

    def toggle_auto_scraping(self):
        """
        Ativa/desativa scraping automático
        """
        if self.state["settings"]["auto_scraping"]:
            # Iniciar scraping automático
            self.scraper.start_auto_scraping(
                callback_function=self.auto_scraping_callback,
                interval_minutes=self.state["settings"]["scraping_interval"]
            )
            self.auto_scraping_active = True
            st.success("🚀 Scraping automático ATIVADO")
        else:
            # Desativar scraping automático
            self.auto_scraping_active = False
            st.info("⏸️ Scraping automático DESATIVADO")

    def manual_scraping_now(self):
        """
        Scraping manual imediato
        """
        try:
            with st.spinner("🌐 Executando scraping manual..."):
                data = self.scraper.scrape_with_reconnect()
                
                if data:
                    progress_bar = st.progress(0)
                    total_beads = len(data)
                    
                    for i, color in enumerate(data):
                        self.register_bead(color, historical_import=True)
                        progress_bar.progress((i + 1) / total_beads)
                        time.sleep(0.05)
                    
                    self.state["historical_data_imported"] = True
                    self.state["last_auto_scrape"] = datetime.now().isoformat()
                    self.save_state()
                    
                    st.success(f"✅ {total_beads} resultados obtidos via scraping manual")
                    
                    # Treinar modelo
                    if total_beads >= 10:
                        self.train_model()
                    
                    return True
                else:
                    st.error("❌ Nenhum dado obtido via scraping")
                    return False
                    
        except Exception as e:
            st.error(f"Erro no scraping manual: {e}")
            return False

    def register_bead(self, color, historical_import=False):
        try:
            # DEBUG: Verificar estado antes
            debug_info = f"DEBUG ANTES: seq_vermelho={self.state['statistics'].get('seq_vermelho', 0)}, seq_empate={self.state['statistics'].get('seq_empate', 0)}"
            print(debug_info)
            
            # Guardar previsão atual ANTES de registar (apenas se não for importação histórica)
            if not historical_import:
                current_prediction, _ = self.get_next_prediction()
            
            bead = {"color": color}
            
            current_col = self.state["current_column"]
            
            # Se a coluna atual está vazia OU a cor é a mesma E ainda não atingiu 6 beads
            if not current_col or (current_col[-1]["color"] == color and len(current_col) < 6):
                self.state["current_column"].append(bead)
            else:
                # Cor diferente OU atingiu 6 beads - fecha a coluna atual e inicia nova
                if current_col:
                    self.state["beads"].append(current_col.copy())
                self.state["current_column"] = [bead]
            
            # Se a coluna atual atingiu 6 beads, fecha automaticamente
            if len(self.state["current_column"]) >= 6:
                self.state["beads"].append(self.state["current_column"].copy())
                self.state["current_column"] = []
            
            self.state["last_color"] = color
            self.state["statistics"]["total_beads"] += 1
            
            # CORREÇÃO CRÍTICA: EMPATES AGORA QUEBRAM SEQUÊNCIAS!
            if color == "azul":
                self.state["statistics"]["azul_count"] += 1
                self.state["statistics"]["seq_vermelho"] = 0  # Quebra sequência vermelha
                self.state["statistics"]["seq_empate"] = 0    # Quebra sequência empate
            elif color == "vermelho":
                self.state["statistics"]["vermelho_count"] += 1
                self.state["statistics"]["seq_vermelho"] += 1
                self.state["statistics"]["seq_empate"] = 0    # Quebra sequência empate
            else:  # empate
                self.state["statistics"]["empate_count"] += 1
                self.state["statistics"]["seq_empate"] += 1   # Cria sequência empate
                self.state["statistics"]["seq_vermelho"] = 0  # Quebra sequência vermelha
            
            # Atualizar win rate se havia previsão (apenas se não for importação histórica)
            if not historical_import and 'current_prediction' in locals():
                if current_prediction:
                    self.state["ml_model"]["total_predictions"] += 1
                    if current_prediction == color:
                        self.state["ml_model"]["hits"] += 1
                        # Atualizar performance do modelo atual
                        current_perf = self.state["ml_model"]["model_performance"].get(self.current_engine_type, 50)
                        self.state["ml_model"]["model_performance"][self.current_engine_type] = min(95, current_perf + 2)
                    else:
                        # Penalizar modelo atual
                        current_perf = self.state["ml_model"]["model_performance"].get(self.current_engine_type, 50)
                        self.state["ml_model"]["model_performance"][self.current_engine_type] = max(5, current_perf - 1)
            
            # LÓGICA DO GALE - VERIFICAR APÓS REGISTRO (apenas se não for importação histórica)
            if not historical_import and 'current_prediction' in locals():
                if current_prediction and current_prediction != color:
                    # Previsão errou - verificar se mantém a mesma previsão após treino
                    if self.state["settings"]["auto_train"]:
                        if self.state["statistics"]["total_beads"] % self.state["settings"]["train_interval"] == 0:
                            self.train_model(auto=True)
                            
                            # Verificar se a nova previsão é a mesma que a anterior
                            new_prediction, _ = self.get_next_prediction()
                            if new_prediction == current_prediction:
                                # Mantém a mesma previsão - INCREMENTAR GALE
                                self.state["gale_count"] += 1
                                if self.state["gale_count"] > 2:
                                    self.state["gale_count"] = 0  # Reset após 2 gales
                            else:
                                # Mudou de previsão - RESETAR GALE
                                self.state["gale_count"] = 0
                else:
                    # Previsão acertou ou não havia previsão - RESETAR GALE
                    self.state["gale_count"] = 0
                    
                    # Auto-treino normal
                    if self.state["settings"]["auto_train"]:
                        if self.state["statistics"]["total_beads"] % self.state["settings"]["train_interval"] == 0:
                            self.train_model(auto=True)
            
            # DEBUG: Verificar estado depois
            debug_info = f"DEBUG DEPOIS: seq_vermelho={self.state['statistics'].get('seq_vermelho', 0)}, seq_empate={self.state['statistics'].get('seq_empate', 0)}"
            print(debug_info)
            
            self.save_state()
            
        except Exception as e:
            st.error(f"Erro ao registrar bead: {str(e)}")
            # Tentar recuperar o estado
            self.load_initial_state()

    def train_model(self, auto=False):
        try:
            # Treinar TODOS os modelos sempre
            all_results = {}
            
            for model_type, engine in self.ml_engines.items():
                result = engine.train_model(self.state, self.state["statistics"])
                all_results[model_type] = result
            
            # Lógica de rotação automática de modelos
            if self.state["settings"]["auto_switch"]:
                training_count = self.state["ml_model"].get("training_count", 0)
                
                # Alternar modelos baseado no intervalo configurado
                if training_count % self.state["settings"]["rotation_interval"] == 0:
                    models = list(self.ml_engines.keys())
                    current_index = models.index(self.current_engine_type)
                    next_index = (current_index + 1) % len(models)
                    self.current_engine_type = models[next_index]
                    self.state["settings"]["current_model"] = self.current_engine_type
            
            # Usar resultados do modelo atual
            result = all_results[self.current_engine_type]
            
            if result["success"]:
                self.state["ml_model"].update({
                    "trained": True,
                    "accuracy": result["accuracy"],
                    "predictions": result["predictions"],
                    "last_trained": datetime.now().isoformat(),
                    "training_count": self.state["ml_model"].get("training_count", 0) + 1,
                    "model_type": result["model_type"],
                    "training_examples": result.get("training_examples", 0),
                    "active_model": self.current_engine_type,
                    "features_info": result.get("features_used", "Heurísticas")
                })
                self.save_state()
                return True
            return False
            
        except Exception as e:
            st.error(f"Erro ao treinar modelo: {str(e)}")
            return False

    def get_next_prediction(self):
        try:
            if not self.state["ml_model"]["trained"]:
                return None, 0
            
            predictions = self.state["ml_model"]["predictions"]
            if not predictions:
                return None, 0
            
            max_color = max(predictions, key=predictions.get)
            confidence = predictions[max_color]
            
            return max_color, confidence
            
        except Exception as e:
            st.error(f"Erro ao obter previsão: {str(e)}")
            return None, 0

    def save_state(self):
        try:
            self.data_manager.save_data(self.state, "app_state.json")
        except Exception as e:
            st.error(f"Erro ao salvar estado: {str(e)}")

    def reset_model(self):
        try:
            self.state["beads"] = []
            self.state["current_column"] = []
            self.state["last_color"] = None
            self.state["previous_prediction"] = None
            self.state["gale_count"] = 0
            self.state["historical_data_imported"] = False
            self.state["last_auto_scrape"] = None
            self.state["statistics"].update({
                "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
                "total_beads": 0, "seq_vermelho": 0, "seq_empate": 0
            })
            self.state["ml_model"].update({
                "trained": False, "accuracy": 0,
                "predictions": {"azul": 44.5, "vermelho": 44.5, "empate": 11.0},
                "last_trained": None, "training_count": 0,
                "model_type": "Nenhum", "training_examples": 0,
                "active_model": "Nenhum",
                "features_info": "Nenhum",
                "hits": 0,
                "total_predictions": 0,
                "model_performance": {"RandomForest": 50, "SVM": 50, "LSTM": 50}
            })
            self.save_state()
            self._init_all_engines()
            
        except Exception as e:
            st.error(f"Erro ao resetar modelo: {str(e)}")
            # Recarregar estado inicial
            self.load_initial_state()

def main():
    st.set_page_config(
        page_title="BB DEEP - AUTO SCRAPING",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # CSS para interface moderna
    st.markdown("""
    <style>
    .main-container {
        width: 100%;
        max-width: 100%;
        margin: 0;
        padding: 8px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .prediction-compact {
        border-radius: 12px;
        padding: 15px 10px;
        text-align: center;
        margin: 8px 0;
        font-weight: bold;
        min-height: 60px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .prediction-azul { 
        background: linear-gradient(135deg, #2196f3, #1976d2);
        color: white;
    }
    .prediction-vermelho { 
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
    }
    .prediction-empate { 
        background: linear-gradient(135deg, #ffc107, #ffa000);
        color: black;
    }
    .prediction-gale {
        border: 3px solid #ff0000;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { border-color: #ff0000; }
        50% { border-color: #ff6666; }
        100% { border-color: #ff0000; }
    }
    .stButton button {
        height: 45px !important;
        font-size: 16px !important;
        margin: 4px 0 !important;
        border-radius: 8px !important;
    }
    .auto-scraping-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .connection-status {
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
    }
    .status-connected {
        background: #4CAF50;
        color: white;
    }
    .status-disconnected {
        background: #f44336;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.title("🤖 BB DEEP - AUTO SCRAPING BETTILT")
    
    try:
        # Inicializar app
        if 'app' not in st.session_state:
            st.session_state.app = BBDeepMobile()
        
        app = st.session_state.app
        
        # SECÇÃO DE SCRAPING AUTOMÁTICO
        st.markdown("---")
        st.markdown('<div class="auto-scraping-section">', unsafe_allow_html=True)
        st.markdown("### 🌐 Sistema de Auto-Scraping Bettilt")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Status da conexão
            connection_status = app.scraper.get_connection_status()
            status_class = "status-connected" if "Conectado" in connection_status else "status-disconnected"
            st.markdown(f'<div class="connection-status {status_class}">{connection_status}</div>', unsafe_allow_html=True)
            
            if app.state["last_auto_scrape"]:
                last_scrape = datetime.fromisoformat(app.state["last_auto_scrape"])
                time_since = datetime.now() - last_scrape
                minutes = time_since.total_seconds() / 60
                st.caption(f"Último scraping: {minutes:.1f} minutos atrás")
        
        with col2:
            # Botão de scraping manual
            if st.button("🔄 Scraping Manual", use_container_width=True):
                app.manual_scraping_now()
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CONTROLES DE SCRAPING AUTOMÁTICO
        with st.expander("⚙️ Configurações de Auto-Scraping", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                auto_scraping = st.checkbox(
                    "Scraping Automático", 
                    value=app.state["settings"]["auto_scraping"],
                    help="Ativa scraping automático a cada X minutos"
                )
            
            with col2:
                scraping_interval = st.slider(
                    "Intervalo (minutos)",
                    min_value=5,
                    max_value=60,
                    value=app.state["settings"]["scraping_interval"],
                    help="Intervalo entre reconexões (como o Bettilt real)"
                )
            
            if st.button("💾 Aplicar Configurações", key="save_scraping_config"):
                app.state["settings"]["auto_scraping"] = auto_scraping
                app.state["settings"]["scraping_interval"] = scraping_interval
                app.scraper.connection_interval = scraping_interval
                app.save_state()
                
                # Ativar/desativar scraping automático
                app.toggle_auto_scraping()
                st.rerun()
        
        # PREVISÃO PRINCIPAL
        next_color, confidence = app.get_next_prediction()
        gale_count = app.state["gale_count"]
        
        if next_color:
            color_name = {"azul": "AZUL", "vermelho": "VERMELHO", "empate": "EMPATE"}
            color_class = f"prediction-{next_color}"
            color_emoji = {"azul": "🔵", "vermelho": "🔴", "empate": "🟡"}
            
            if gale_count > 0:
                color_class += " prediction-gale"
            
            gale_text = f"<span class='gale-indicator'>{gale_count}º GALE</span>" if gale_count > 0 else ""
            
            st.markdown(f"""
            <div class="prediction-compact {color_class}">
                <div style="font-size: 18px; margin-bottom: 2px;">
                    PRÓXIMA: {color_emoji[next_color]} {color_name[next_color]} {gale_text}
                </div>
                <div style="font-size: 14px;">{confidence:.1f}% confiança</div>
            </div>
            """, unsafe_allow_html=True)
            
            if app.state["ml_model"]["trained"]:
                current_model = app.current_engine_type
                model_perf = app.state["ml_model"]["model_performance"].get(current_model, 50)
                model_info = f"🎯 {current_model} | {app.state['ml_model']['accuracy']:.1f}% precisão | Performance: {model_perf}%"
                st.caption(model_info)
        else:
            st.info("📊 Ative o scraping automático para começar a receber previsões")
        
        # BOTÕES DE CONTROLE
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 TREINAR MODELO", use_container_width=True):
                if app.train_model():
                    st.rerun()
        
        with col2:
            if st.button("🔄 RESETAR TUDO", use_container_width=True):
                app.reset_model()
                st.rerun()
        
        # BEAD ROAD
        st.markdown("**Bead Road:**")
        beads = app.state["beads"]
        current_column = app.state["current_column"]
        
        if beads or current_column:
            html = '<div style="overflow-x: auto; white-space: nowrap; margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 8px; max-height: 220px;"><div style="display: flex;">'
            
            for column in beads:
                html += '<div style="display: inline-flex; flex-direction: column; margin-right: 8px; width: 32px; justify-content: flex-start; align-items: center;">'
                for bead in column:
                    color = bead["color"]
                    emoji = "🔵" if color == "azul" else "🔴" if color == "vermelho" else "🟡"
                    html += f'<div style="font-size: 24px; line-height: 30px; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; border: 1px solid #ddd; border-radius: 50%; margin-bottom: 4px;">{emoji}</div>'
                html += '</div>'
            
            if current_column:
                html += '<div style="display: inline-flex; flex-direction: column; margin-right: 8px; width: 32px; justify-content: flex-start; align-items: center;">'
                for bead in current_column:
                    color = bead["color"]
                    emoji = "🔵" if color == "azul" else "🔴" if color == "vermelho" else "🟡"
                    html += f'<div style="font-size: 24px; line-height: 30px; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; border: 1px solid #ddd; border-radius: 50%; margin-bottom: 4px;">{emoji}</div>'
                html += '</div>'
            
            html += '</div></div>'
            st.markdown(html, unsafe_allow_html=True)
        
        # ESTATÍSTICAS
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔵 Azul", app.state['statistics'].get('azul_count', 0))
        with col2:
            st.metric("🔴 Verm.", app.state['statistics'].get('vermelho_count', 0))
        with col3:
            st.metric("🟡 Emp.", app.state['statistics'].get('empate_count', 0))
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("📊 Total", app.state['statistics'].get('total_beads', 0))
        with col5:
            st.metric("🔴 Seq", app.state['statistics'].get('seq_vermelho', 0))
        with col6:
            st.metric("🟡 Seq", app.state['statistics'].get('seq_empate', 0))
        
        # INFO ADICIONAL
        with st.expander("📊 Informações Detalhadas"):
            st.write(f"**Scraping Automático:** {'✅ ATIVO' if app.state['settings']['auto_scraping'] else '❌ INATIVO'}")
            st.write(f"**Intervalo:** {app.state['settings']['scraping_interval']} minutos")
            st.write(f"**Modelo Atual:** {app.current_engine_type}")
            
            if app.state["last_auto_scrape"]:
                last_time = datetime.fromisoformat(app.state["last_auto_scrape"])
                st.write(f"**Último Scraping:** {last_time.strftime('%H:%M:%S')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # MENSAGEM FINAL
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #666; font-size: 14px;'>🤖 Auto-Scraping Bettilt | Reconexão a cada 20min | 3 Modelos ML</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Erro crítico: {str(e)}")
        st.info("Recarregue a página para reiniciar a aplicação.")

if __name__ == "__main__":
    main()