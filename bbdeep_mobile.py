import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
import warnings
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import io
warnings.filterwarnings('ignore')

# ===== DATA MANAGER COM EXPORTAÇÃO =====
class DataManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.setup_logging()
        self.ensure_data_dir()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bbdeep_logs.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
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
            
            # Backup local
            if os.path.exists(filepath):
                backup_path = filepath + ".backup"
                with open(filepath, 'r', encoding='utf-8') as original:
                    backup_data = json.load(original)
                with open(backup_path, 'w', encoding='utf-8') as backup:
                    json.dump(backup_data, backup, indent=2, ensure_ascii=False)
            
            # Guardar localmente
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
    
    def save_training_data(self, training_data, filename_prefix):
        """Guarda dados de treino separadamente"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_training_{timestamp}.json"
            filepath = self.get_file_path(filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Dados de treino guardados: {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao guardar dados de treino: {e}")
            return False

    # ===== MÉTODOS DE EXPORTAÇÃO =====
    def export_training_history_csv(self):
        """Exporta histórico de treinos para CSV"""
        try:
            # Encontrar todos os ficheiros de treino
            pattern = os.path.join(self.data_dir, "bbdeep_training_*.json")
            training_files = glob.glob(pattern)
            
            if not training_files:
                return None, "Nenhum dado de treino encontrado"
            
            training_data = []
            for file_path in training_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    training_data.append(data)
            
            # Converter para DataFrame
            df = pd.DataFrame(training_data)
            
            # Ordenar por timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            # Converter para CSV
            csv_data = df.to_csv(index=False, encoding='utf-8')
            return csv_data, f"Exportados {len(training_data)} registos de treino"
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar histórico de treinos: {e}")
            return None, f"Erro na exportação: {str(e)}"
    
    def export_logs_csv(self):
        """Exporta logs para CSV"""
        try:
            log_file = 'bbdeep_logs.log'
            if not os.path.exists(log_file):
                return None, "Ficheiro de logs não encontrado"
            
            # Ler logs
            with open(log_file, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            
            # Parse dos logs (formato: timestamp - level - message)
            log_data = []
            for line in log_lines:
                parts = line.strip().split(' - ', 2)
                if len(parts) == 3:
                    timestamp, level, message = parts
                    log_data.append({
                        'timestamp': timestamp,
                        'level': level,
                        'message': message
                    })
            
            if not log_data:
                return None, "Nenhum log válido encontrado"
            
            # Converter para DataFrame e CSV
            df = pd.DataFrame(log_data)
            csv_data = df.to_csv(index=False, encoding='utf-8')
            return csv_data, f"Exportados {len(log_data)} entradas de log"
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar logs: {e}")
            return None, f"Erro na exportação: {str(e)}"
    
    def export_complete_data_json(self):
        """Exporta todos os dados completos em JSON"""
        try:
            # Dados principais
            app_state = self.load_data("app_state.json", {})
            
            # Histórico de treinos
            pattern = os.path.join(self.data_dir, "bbdeep_training_*.json")
            training_files = glob.glob(pattern)
            training_history = []
            
            for file_path in training_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    training_history.append(json.load(f))
            
            # Logs (últimas 1000 linhas)
            log_data = []
            log_file = 'bbdeep_logs.log'
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_lines = f.readlines()[-1000:]  # Últimas 1000 linhas
                    for line in log_lines:
                        parts = line.strip().split(' - ', 2)
                        if len(parts) == 3:
                            timestamp, level, message = parts
                            log_data.append({
                                'timestamp': timestamp,
                                'level': level,
                                'message': message
                            })
            
            # Compilar todos os dados
            complete_data = {
                'export_timestamp': datetime.now().isoformat(),
                'app_state': app_state,
                'training_history': training_history,
                'recent_logs': log_data,
                'summary': {
                    'total_trainings': len(training_history),
                    'total_logs': len(log_data),
                    'beads_count': app_state.get('statistics', {}).get('total_beads', 0)
                }
            }
            
            json_data = json.dumps(complete_data, indent=2, ensure_ascii=False)
            return json_data, f"Dados exportados: {len(training_history)} treinos, {len(log_data)} logs"
            
        except Exception as e:
            self.logger.error(f"Erro ao exportar dados completos: {e}")
            return None, f"Erro na exportação: {str(e)}"

# ===== ML ENGINE (mantém-se igual) =====
class MLEngine:
    def __init__(self, model_type="RandomForest"):
        self.model_type = model_type
        self.model_trained = False
        self.predictions = {"azul": 44.5, "vermelho": 44.5, "empate": 11.0}
        
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
    
    def should_switch_to_heuristic(self, current_win_rate, total_predictions, recent_errors):
        if total_predictions < 10:
            return False
        
        if current_win_rate < 45:
            return True
        
        if len(recent_errors) >= 10:
            error_rate = sum(recent_errors[-10:]) / 10 * 100
            if error_rate > 60:
                return True
            
        return False

    def should_switch_back_to_rf(self, current_win_rate, total_predictions, recent_errors, total_beads, rf_switch_interval, rf_performance_history, min_rf_confidence):
        if total_beads < 25:
            return False
        
        if len(rf_performance_history) >= 5:
            recent_performance = rf_performance_history[-5:]
            improving_trend = self._is_performance_improving(recent_performance)
            if improving_trend and recent_performance[-1] >= min_rf_confidence:
                return True
        
        if len(rf_performance_history) >= 3:
            if all(perf >= min_rf_confidence + 5 for perf in rf_performance_history[-3:]):
                return True
        
        if current_win_rate < 50:
            return False
        
        if len(recent_errors) >= 10:
            error_rate = sum(recent_errors[-10:]) / 10 * 100
            if error_rate > 60:
                return False
        
        if total_predictions % rf_switch_interval == 0:
            return True
            
        return False

    def _is_performance_improving(self, performance_history):
        if len(performance_history) < 2:
            return False
        
        improvements = 0
        for i in range(1, len(performance_history)):
            if performance_history[i] > performance_history[i-1]:
                improvements += 1
        
        return improvements >= len(performance_history) - 1

    def train_model(self, beads_data, statistics, model_type=None, current_win_rate=50, 
                    total_predictions=0, current_model="RandomForest", recent_errors=[],
                    rf_switch_interval=10, rf_performance_history=[], min_rf_confidence=55):
        if model_type:
            self.model_type = model_type
            self._init_model()
        
        total_beads = statistics.get("total_beads", 0)
        
        if current_model == "Heurístico" and self.model_type == "RandomForest":
            if self.should_switch_back_to_rf(current_win_rate, total_predictions, recent_errors, total_beads, rf_switch_interval, rf_performance_history, min_rf_confidence):
                logging.info(f"🔄 Tentando voltar para Random Forest - condições favoráveis")
                use_heuristic = False
            else:
                use_heuristic = True
        else:
            use_heuristic = False
            if self.model_type == "RandomForest" and total_predictions >= 10:
                if self.should_switch_to_heuristic(current_win_rate, total_predictions, recent_errors):
                    use_heuristic = True
                    logging.info("🔁 Alternando para modelo heurístico - RF com desempenho fraco")
        
        if use_heuristic or total_beads < 15:
            ml_result = self._train_heuristic_model(beads_data, statistics)
            ml_result["model_type"] = "Heurístico"
            ml_result["was_switched"] = True
        else:
            ml_result = self._train_ml_model(beads_data, statistics)
            if not ml_result["success"]:
                ml_result = self._train_heuristic_model(beads_data, statistics)
                ml_result["model_type"] = "Heurístico"
                ml_result["was_switched"] = True
            else:
                ml_result["model_type"] = self.model_type
                ml_result["was_switched"] = False
        
        self.model_trained = True
        self.predictions = ml_result["predictions"]
        
        return ml_result

    def _train_ml_model(self, beads_data, statistics):
        try:
            all_beads = self._get_all_beads(beads_data)
            
            if len(all_beads) < self.window_size + 5:
                return {"success": False, "error": "Dados insuficientes para ML"}
            
            X, y = self._create_ml_features(all_beads)
            
            if len(X) < 10:
                return {"success": False, "error": "Poucos exemplos para treino"}
            
            if self.model_type in ["RandomForest"]:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                
                self.ml_model.fit(X_train, y_train)
                accuracy = self.ml_model.score(X_test, y_test) * 100
                
                last_sequence = self._get_last_sequence(all_beads)
                if last_sequence is not None:
                    next_pred_proba = self.ml_model.predict_proba([last_sequence])[0]
                    classes = self.ml_model.classes_
                    
                    ml_predictions = {}
                    for i, class_name in enumerate(classes):
                        color = self.label_encoder.inverse_transform([class_name])[0]
                        ml_predictions[color] = next_pred_proba[i] * 100
                    
                    for color in ["azul", "vermelho", "empate"]:
                        if color not in ml_predictions:
                            ml_predictions[color] = 0
                else:
                    ml_predictions = {"azul": 44.5, "vermelho": 44.5, "empate": 11.0}
            
            elif self.model_type == "LSTM":
                bead_numbers = self.label_encoder.transform(all_beads)
                sequences = []
                labels = []
                for i in range(self.window_size, len(bead_numbers) - 1):
                    seq = bead_numbers[i - self.window_size:i]
                    one_hot_seq = np.eye(3)[seq]
                    sequences.append(one_hot_seq)
                    labels.append(bead_numbers[i])
                
                X = np.array(sequences)
                y = np.array(labels)
                
                split = int(len(X) * 0.7)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
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
                
                epochs = 50
                for epoch in range(epochs):
                    self.ml_model.train()
                    for batch_X, batch_y in train_loader:
                        self.optimizer.zero_grad()
                        outputs = self.ml_model(batch_X)
                        outputs = outputs.view(-1, 3)
                        batch_y = batch_y.view(-1)
                        loss = self.criterion(outputs, batch_y)
                        loss.backward()
                        self.optimizer.step()
                
                self.ml_model.eval()
                with torch.no_grad():
                    test_X = torch.tensor(X_test, dtype=torch.float32)
                    outputs = self.ml_model(test_X)
                    predicted = torch.argmax(outputs, dim=1)
                    accuracy = (predicted.numpy() == y_test).mean() * 100
                
                last_sequence = self._get_last_sequence(all_beads)
                if last_sequence is not None:
                    seq = np.eye(3)[last_sequence[:self.window_size]]
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
            
            azul_prob = (azul_count / total_beads) * 100
            vermelho_prob = (vermelho_count / total_beads) * 100
            empate_prob = (empate_count / total_beads) * 100
            
            seq_vermelho = statistics.get("seq_vermelho", 0)
            seq_empate = statistics.get("seq_empate", 0)
            
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
            
            total = sum(predictions.values())
            predictions = {k: (v / total) * 100 for k, v in predictions.items()}
            
            accuracy = 55 + min(35, total_beads / 15)
            
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
        
        bead_numbers = self.label_encoder.transform(all_beads)
        
        for i in range(self.window_size, len(bead_numbers) - 1):
            features = bead_numbers[i - self.window_size:i]
            extended_features = list(features)
            
            for color_idx in range(len(self.label_encoder.classes_)):
                extended_features.append(np.sum(features == color_idx))
            
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

# ===== MAIN APP ATUALIZADA =====
class BBDeepMobile:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_engine = None
        
        if 'app_initialized' not in st.session_state:
            self.load_initial_state()
            st.session_state.app_initialized = True
        
        self.ml_engine = MLEngine(self.state["settings"].get("ml_model_type", "RandomForest"))

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
                "recent_errors": [],
                "rf_performance_history": []
            },
            "settings": {
                "auto_train": True, 
                "train_interval": 1,
                "ml_model_type": "RandomForest",
                "auto_switch": True,
                "max_gale": 1,
                "rf_switch_interval": 10,
                "min_rf_confidence": 55
            }
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
        for key in ["hits", "total_predictions", "recent_errors", "rf_performance_history"]:
            if key not in loaded_state["ml_model"]:
                loaded_state["ml_model"][key] = 0 if key in ["hits", "total_predictions"] else []
        
        if "settings" not in loaded_state:
            loaded_state["settings"] = {}
        for key in ["ml_model_type", "auto_switch", "max_gale", "rf_switch_interval", "min_rf_confidence"]:
            if key not in loaded_state["settings"]:
                if key == "ml_model_type":
                    loaded_state["settings"][key] = "RandomForest"
                elif key == "auto_switch":
                    loaded_state["settings"][key] = True
                elif key == "max_gale":
                    loaded_state["settings"][key] = 1
                elif key == "rf_switch_interval":
                    loaded_state["settings"][key] = 10
                elif key == "min_rf_confidence":
                    loaded_state["settings"][key] = 55
        
        return loaded_state

    @property
    def state(self):
        return st.session_state.app_state

    def register_bead(self, color):
        current_prediction, _ = self.get_next_prediction()
        
        bead = {"color": color}
        current_col = self.state["current_column"]
        
        if not current_col or (current_col[-1]["color"] == color and len(current_col) < 6):
            self.state["current_column"].append(bead)
        else:
            if current_col:
                self.state["beads"].append(current_col.copy())
            self.state["current_column"] = [bead]
        
        if len(self.state["current_column"]) >= 6:
            self.state["beads"].append(self.state["current_column"].copy())
            self.state["current_column"] = []
        
        self.state["last_color"] = color
        self.state["statistics"]["total_beads"] += 1
        
        if color == "azul":
            self.state["statistics"]["azul_count"] += 1
            self.state["statistics"]["seq_vermelho"] = 0
            self.state["statistics"]["seq_empate"] = 0
        elif color == "vermelho":
            self.state["statistics"]["vermelho_count"] += 1
            self.state["statistics"]["seq_vermelho"] += 1
            self.state["statistics"]["seq_empate"] = 0
        else:
            self.state["statistics"]["empate_count"] += 1
            self.state["statistics"]["seq_empate"] += 1
            self.state["statistics"]["seq_vermelho"] = 0
        
        if current_prediction:
            self.state["ml_model"]["total_predictions"] += 1
            was_correct = current_prediction == color
            self.state["ml_model"]["recent_errors"].append(0 if was_correct else 1)
            
            if len(self.state["ml_model"]["recent_errors"]) > 20:
                self.state["ml_model"]["recent_errors"] = self.state["ml_model"]["recent_errors"][-20:]
            
            if was_correct:
                self.state["ml_model"]["hits"] += 1
        
        max_gale = self.state["settings"]["max_gale"]
        if current_prediction and current_prediction != color:
            if self.state["settings"]["auto_train"]:
                if self.state["statistics"]["total_beads"] % self.state["settings"]["train_interval"] == 0:
                    win_rate = (self.state["ml_model"]["hits"] / self.state["ml_model"]["total_predictions"] * 100) if self.state["ml_model"]["total_predictions"] > 0 else 50
                    
                    self.train_model(
                        auto=True, 
                        current_win_rate=win_rate,
                        total_predictions=self.state["ml_model"]["total_predictions"]
                    )
                    
                    new_prediction, _ = self.get_next_prediction()
                    if new_prediction == current_prediction and self.state["gale_count"] < max_gale:
                        self.state["gale_count"] += 1
                    else:
                        self.state["gale_count"] = 0
            else:
                self.state["gale_count"] = 0
        else:
            self.state["gale_count"] = 0
            
            if self.state["settings"]["auto_train"]:
                if self.state["statistics"]["total_beads"] % self.state["settings"]["train_interval"] == 0:
                    win_rate = (self.state["ml_model"]["hits"] / self.state["ml_model"]["total_predictions"] * 100) if self.state["ml_model"]["total_predictions"] > 0 else 50
                    self.train_model(
                        auto=True, 
                        current_win_rate=win_rate,
                        total_predictions=self.state["ml_model"]["total_predictions"]
                    )
        
        # Guardar dados de treino quando há treino
        if self.state["settings"]["auto_train"] and self.state["statistics"]["total_beads"] % self.state["settings"]["train_interval"] == 0:
            training_data = {
                "timestamp": datetime.now().isoformat(),
                "beads_count": self.state["statistics"]["total_beads"],
                "model_type": self.state["ml_model"].get("model_type", "Nenhum"),
                "accuracy": self.state["ml_model"].get("accuracy", 0),
                "predictions": self.state["ml_model"].get("predictions", {}),
                "win_rate": win_rate if 'win_rate' in locals() else 0
            }
            self.data_manager.save_training_data(training_data, "bbdeep")
        
        self.save_state()

    def train_model(self, auto=False, current_win_rate=50, total_predictions=0):
        if self.state["settings"]["auto_switch"]:
            active_model = self.state["ml_model"]["active_model"]
            recent_errors = self.state["ml_model"].get("recent_errors", [])
            total_beads = self.state["statistics"]["total_beads"]
            rf_switch_interval = self.state["settings"]["rf_switch_interval"]
            rf_performance_history = self.state["ml_model"].get("rf_performance_history", [])
            min_rf_confidence = self.state["settings"]["min_rf_confidence"]
            
            rf_engine = MLEngine("RandomForest")
            rf_result = rf_engine.train_model(
                self.state, 
                self.state["statistics"],
                current_win_rate=current_win_rate,
                total_predictions=total_predictions,
                current_model=active_model,
                recent_errors=recent_errors,
                rf_switch_interval=rf_switch_interval,
                rf_performance_history=rf_performance_history,
                min_rf_confidence=min_rf_confidence
            )
            
            if rf_result["success"] and not rf_result.get("was_switched", False):
                rf_accuracy = rf_result["accuracy"]
                self.state["ml_model"]["rf_performance_history"].append(rf_accuracy)
                if len(self.state["ml_model"]["rf_performance_history"]) > 10:
                    self.state["ml_model"]["rf_performance_history"] = self.state["ml_model"]["rf_performance_history"][-10:]
            
            if active_model == "Heurístico":
                should_switch = rf_engine.should_switch_back_to_rf(
                    current_win_rate, 
                    total_predictions, 
                    recent_errors, 
                    total_beads, 
                    rf_switch_interval,
                    self.state["ml_model"]["rf_performance_history"],
                    min_rf_confidence
                )
                
                if should_switch and rf_result["success"] and not rf_result.get("was_switched", False):
                    result = rf_result
                    self.ml_engine = rf_engine
                    self.state["settings"]["ml_model_type"] = "RandomForest"
                    self.state["ml_model"]["active_model"] = "RandomForest"
                    logging.info("🎯 VOLTANDO para Random Forest - performance melhorou!")
                else:
                    heuristic_engine = MLEngine("RandomForest")
                    result = heuristic_engine._train_heuristic_model(self.state, self.state["statistics"])
                    result["model_type"] = "Heurístico"
                    result["was_switched"] = True
                    self.ml_engine = heuristic_engine
                    self.state["ml_model"]["active_model"] = "Heurístico"
            else:
                if rf_result.get("was_switched", False) or not rf_result["success"]:
                    heuristic_engine = MLEngine("RandomForest")
                    result = heuristic_engine._train_heuristic_model(self.state, self.state["statistics"])
                    result["model_type"] = "Heurístico" 
                    result["was_switched"] = True
                    self.ml_engine = heuristic_engine
                    self.state["ml_model"]["active_model"] = "Heurístico"
                    logging.info("🔁 MUDANDO para Heurístico - RF com desempenho fraco")
                else:
                    result = rf_result
                    self.ml_engine = rf_engine
                    self.state["ml_model"]["active_model"] = "RandomForest"
        else:
            result = self.ml_engine.train_model(
                self.state, 
                self.state["statistics"],
                current_win_rate=current_win_rate,
                total_predictions=total_predictions
            )
        
        if result["success"]:
            self.state["ml_model"].update({
                "trained": True,
                "accuracy": result["accuracy"],
                "predictions": result["predictions"],
                "last_trained": datetime.now().isoformat(),
                "training_count": self.state["ml_model"].get("training_count", 0) + 1,
                "model_type": result["model_type"],
                "training_examples": result.get("training_examples", 0),
                "active_model": result["model_type"],
                "features_info": result.get("features_used", "Heurísticas"),
                "was_switched": result.get("was_switched", False)
            })
            self.save_state()
            return True
        return False

    def get_next_prediction(self):
        if not self.state["ml_model"]["trained"]:
            return None, 0
        
        predictions = self.state["ml_model"]["predictions"]
        if not predictions:
            return None, 0
        
        max_color = max(predictions, key=predictions.get)
        confidence = predictions[max_color]
        
        return max_color, confidence

    def save_state(self):
        self.data_manager.save_data(self.state, "app_state.json")

    def reset_model(self):
        self.state["beads"] = []
        self.state["current_column"] = []
        self.state["last_color"] = None
        self.state["previous_prediction"] = None
        self.state["gale_count"] = 0
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
            "recent_errors": [],
            "rf_performance_history": []
        })
        self.save_state()
        self.ml_engine = MLEngine(self.state["settings"]["ml_model_type"])

def main():
    st.set_page_config(
        page_title="BB DEEP Mobile + GALE INTELIGENTE",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # CSS
    st.markdown("""
    <style>
    .main-container { width: 100%; max-width: 100%; margin: 0; padding: 8px; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .prediction-compact { border-radius: 12px; padding: 15px 10px; text-align: center; margin: 8px 0; font-weight: bold; min-height: 60px; display: flex; flex-direction: column; justify-content: center; }
    .prediction-azul { background: linear-gradient(135deg, #2196f3, #1976d2); color: white; }
    .prediction-vermelho { background: linear-gradient(135deg, #f44336, #d32f2f); color: white; }
    .prediction-empate { background: linear-gradient(135deg, #ffc107, #ffa000); color: black; }
    .prediction-gale { border: 3px solid #ff0000; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { border-color: #ff0000; } 50% { border-color: #ff6666; } 100% { border-color: #ff0000; } }
    .stButton button { height: 45px !important; font-size: 16px !important; margin: 4px 0 !important; border-radius: 8px !important; }
    h1 { font-size: 20px !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 16px !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 14px !important; margin-bottom: 0.25rem !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display:none;}
    div[data-testid="stVerticalBlock"] > div { padding: 0.25rem 0; }
    .gale-indicator { background: linear-gradient(135deg, #ff0000, #cc0000); color: white; padding: 3px 8px; border-radius: 10px; font-size: 12px; margin-left: 5px; }
    .model-heuristic { background: linear-gradient(135deg, #ff9800, #f57c00) !important; }
    .bead-road-container { overflow-x: auto; white-space: nowrap; margin: 10px 0; padding: 10px; background-color: #f0f0f0; border-radius: 8px; max-height: 220px; }
    .bead-column { display: inline-flex; flex-direction: column; margin-right: 8px; width: 32px; justify-content: flex-start; align-items: center; }
    .bead { font-size: 24px; line-height: 30px; width: 30px; height: 30px; display: flex; justify-content: center; align-items: center; border: 1px solid #ddd; border-radius: 50%; margin-bottom: 4px; }
    .export-section { background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.title("🤖 BB DEEP - ML INTELIGENTE + 1 GALE")
    
    if 'app' not in st.session_state:
        st.session_state.app = BBDeepMobile()
    
    app = st.session_state.app
    
    # PREVISÃO
    next_color, confidence = app.get_next_prediction()
    gale_count = app.state["gale_count"]
    active_model = app.state["ml_model"]["active_model"]
    
    if next_color:
        color_name = {"azul": "AZUL", "vermelho": "VERMELHO", "empate": "EMPATE"}
        color_class = f"prediction-{next_color}"
        color_emoji = {"azul": "🔵", "vermelho": "🔴", "empate": "🟡"}
        
        if gale_count > 0:
            color_class += " prediction-gale"
        
        if active_model == "Heurístico":
            color_class += " model-heuristic"
        
        gale_text = f"<span class='gale-indicator'>{gale_count}º GALE</span>" if gale_count > 0 else ""
        model_indicator = "🔄 HEUR" if active_model == "Heurístico" else "🤖 ML"
        
        st.markdown(f"""
        <div class="prediction-compact {color_class}">
            <div style="font-size: 18px; margin-bottom: 2px;">
                {model_indicator} | PRÓXIMA: {color_emoji[next_color]} {color_name[next_color]} {gale_text}
            </div>
            <div style="font-size: 14px;">{confidence:.1f}% confiança</div>
        </div>
        """, unsafe_allow_html=True)
        
        if app.state["ml_model"]["trained"]:
            was_switched = app.state["ml_model"].get("was_switched", False)
            switch_info = " 🔁 (Alternado)" if was_switched else ""
            model_info = f"🎯 {app.state['ml_model']['model_type']} | {app.state['ml_model']['accuracy']:.1f}% precisão{switch_info}"
            st.caption(model_info)
            
            if active_model == "Heurístico":
                win_rate = (app.state["ml_model"]["hits"] / app.state["ml_model"]["total_predictions"] * 100) if app.state["ml_model"]["total_predictions"] > 0 else 0
                interval = app.state["settings"]["rf_switch_interval"]
                current_count = app.state["ml_model"]["total_predictions"] % interval
                next_switch = interval - current_count
                
                rf_history = app.state["ml_model"].get("rf_performance_history", [])
                if rf_history:
                    current_rf_perf = rf_history[-1]
                    avg_rf_perf = sum(rf_history) / len(rf_history)
                    
                    st.caption(f"🔄 Volta ao RF em: {next_switch} jogadas | RF Actual: {current_rf_perf:.1f}% (média: {avg_rf_perf:.1f}%)")
                    
                    if current_rf_perf >= 60:
                        st.success(f"✅ RF a melhorar - {current_rf_perf:.1f}% de precisão")
                    elif current_rf_perf >= 55:
                        st.info(f"📈 RF estável - {current_rf_perf:.1f}% de precisão")
                    else:
                        st.warning(f"⚠️ RF precisa melhorar - {current_rf_perf:.1f}% de precisão")
    else:
        st.info("📊 Registe beads e treine o modelo")
    
    # BOTÃO DE TREINO
    if st.button("🎯 TREINAR MODELO ML", use_container_width=True, key="train_ml_main"):
        if app.train_model():
            st.rerun()
    
    # BOTÕES DE REGISTO
    st.markdown("**Registar:**")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        if st.button("🔵 AZUL", use_container_width=True, key="btn_azul"):
            app.register_bead('azul')
            st.rerun()
    
    with btn_col2:
        if st.button("🔴 VERM.", use_container_width=True, key="btn_vermelho"):
            app.register_bead('vermelho')
            st.rerun()
    
    with btn_col3:
        if st.button("🟡 EMPATE", use_container_width=True, key="btn_empate"):
            app.register_bead('empate')
            st.rerun()
    
    # BEAD ROAD
    st.markdown("**Bead Road:**")
    beads = app.state["beads"]
    current_column = app.state["current_column"]
    
    if beads or current_column:
        html = '<div class="bead-road-container"><div style="display: flex;">'
        
        for column in beads:
            html += '<div class="bead-column">'
            for bead in column:
                color = bead["color"]
                emoji = "🔵" if color == "azul" else "🔴" if color == "vermelho" else "🟡"
                html += f'<div class="bead">{emoji}</div>'
            html += '</div>'
        
        if current_column:
            html += '<div class="bead-column">'
            for bead in current_column:
                color = bead["color"]
                emoji = "🔵" if color == "azul" else "🔴" if color == "vermelho" else "🟡"
                html += f'<div class="bead">{emoji}</div>'
            html += '</div>'
        
        html += '</div></div>'
        html += """
        <script>
        var container = parent.document.querySelector('.bead-road-container');
        if (container) { container.scrollLeft = container.scrollWidth; }
        </script>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.caption("Sem beads registados ainda.")
    
    # ESTATÍSTICAS
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔵 Azul", app.state['statistics']['azul_count'], delta=None)
    with col2:
        st.metric("🔴 Verm.", app.state['statistics']['vermelho_count'], delta=None)
    with col3:
        st.metric("🟡 Emp.", app.state['statistics']['empate_count'], delta=None)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("📊 Total", app.state['statistics']['total_beads'], delta=None)
    with col5:
        st.metric("🔴 Seq", app.state['statistics']['seq_vermelho'], delta=None)
    with col6:
        st.metric("🟡 Seq", app.state['statistics']['seq_empate'], delta=None)
    
    if gale_count > 0:
        st.markdown("---")
        st.warning(f"🎯 **EM {gale_count}º GALE** - Mantendo previsão após erro")
    
    # PROBABILIDADES
    if app.state["ml_model"]["trained"]:
        st.markdown("---")
        st.markdown("**Probabilidades ML:**")
        
        pred = app.state["ml_model"]["predictions"]
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown(f"🔵 {pred['azul']:.1f}%")
            st.progress(pred['azul']/100)
        with col_p2:
            st.markdown(f"🔴 {pred['vermelho']:.1f}%")
            st.progress(pred['vermelho']/100)
        with col_p3:
            st.markdown(f"🟡 {pred['empate']:.1f}%")
            st.progress(pred['empate']/100)
    
    # CONTROLES
    st.markdown("---")
    
    if st.button("🔄 RESETAR MODELO", use_container_width=True, key="reset_model"):
        app.reset_model()
        st.rerun()
    
    with st.popover("⚙️ Configurações", use_container_width=True):
        auto_train = st.checkbox("Auto-treino", value=app.state["settings"]["auto_train"], key="auto_train")
        train_interval = st.slider("Intervalo auto-treino:", 1, 20, app.state["settings"]["train_interval"], key="train_interval")
        
        rf_switch_interval = st.slider(
            "Tentar voltar ao RF a cada (jogadas):", 
            min_value=5, 
            max_value=20, 
            value=app.state["settings"]["rf_switch_interval"],
            help="Número de jogadas no heurístico antes de tentar voltar ao Random Forest"
        )
        
        min_rf_confidence = st.slider(
            "Confiança mínima para voltar ao RF (%):", 
            min_value=50, 
            max_value=70, 
            value=app.state["settings"]["min_rf_confidence"],
            help="Precisão mínima que o RF precisa ter para voltar a ser usado automaticamente"
        )
        
        ml_model_type = st.selectbox("Tipo de Modelo ML", ["RandomForest", "LSTM"], 
                                   index=0 if app.state["settings"]["ml_model_type"] == "RandomForest" else 1, 
                                   key="ml_model_type")
        
        auto_switch = st.checkbox("Auto-Switch Inteligente (RF ↔ Heurístico)", 
                                value=app.state["settings"]["auto_switch"], 
                                key="auto_switch")
        
        max_gale = st.selectbox("Máximo Gale", [1], index=0)
        
        if st.button("💾 Aplicar", key="save_config"):
            app.state["settings"]["auto_train"] = auto_train
            app.state["settings"]["train_interval"] = train_interval
            app.state["settings"]["ml_model_type"] = ml_model_type
            app.state["settings"]["auto_switch"] = auto_switch
            app.state["settings"]["max_gale"] = max_gale
            app.state["settings"]["rf_switch_interval"] = rf_switch_interval
            app.state["settings"]["min_rf_confidence"] = min_rf_confidence
            
            app.ml_engine = MLEngine(ml_model_type)
            app.save_state()
            st.rerun()
        
        # ===== SECÇÃO DE EXPORTAÇÃO =====
        st.markdown("---")
        st.markdown("### 📤 Exportar Dados")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📊 Exportar Treinos (CSV)", use_container_width=True):
                csv_data, message = app.data_manager.export_training_history_csv()
                if csv_data:
                    st.download_button(
                        label="⬇️ Descarregar CSV",
                        data=csv_data,
                        file_name=f"bbdeep_training_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.success(message)
                else:
                    st.error(message)
            
            if st.button("📝 Exportar Logs (CSV)", use_container_width=True):
                csv_data, message = app.data_manager.export_logs_csv()
                if csv_data:
                    st.download_button(
                        label="⬇️ Descarregar CSV",
                        data=csv_data,
                        file_name=f"bbdeep_logs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.success(message)
                else:
                    st.error(message)
        
        with col_export2:
            if st.button("📁 Exportar Todos Dados (JSON)", use_container_width=True):
                json_data, message = app.data_manager.export_complete_data_json()
                if json_data:
                    st.download_button(
                        label="⬇️ Descarregar JSON",
                        data=json_data,
                        file_name=f"bbdeep_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    st.success(message)
                else:
                    st.error(message)
            
            if st.button("🔄 Limpar Dados Antigos", use_container_width=True):
                # Implementar limpeza de dados antigos se necessário
                st.info("Funcionalidade de limpeza em desenvolvimento")
    
    with st.popover("📊 Info ML", use_container_width=True):
        st.write(f"**Modelo Ativo:** {app.state['ml_model']['active_model']}")
        st.write(f"**Precisão:** {app.state['ml_model']['accuracy']:.1f}%")
        win_rate = (app.state['ml_model']['hits'] / app.state['ml_model']['total_predictions'] * 100) if app.state['ml_model']['total_predictions'] > 0 else 0
        st.write(f"**Win Rate Real:** {win_rate:.1f}% ({app.state['ml_model']['hits']}/{app.state['ml_model']['total_predictions']})")
        st.write(f"**Exemplos treino:** {app.state['ml_model']['training_examples']}")
        st.write(f"**Total treinos:** {app.state['ml_model']['training_count']}")
        st.write(f"**Gale atual:** {app.state['gale_count']}")
        
        rf_history = app.state["ml_model"].get("rf_performance_history", [])
        if rf_history:
            st.write(f"**Performance RF (histórico):** {', '.join([f'{p:.1f}%' for p in rf_history[-5:]])}")
        
        if app.state['current_column']:
            st.write("**Últimas jogadas:**")
            last_beads = ""
            for bead in app.state['current_column'][-5:]:
                symbol = bead['color'][0].upper()
                last_beads += symbol + " "
            st.write(last_beads)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666; font-size: 14px;'>🤖 ML Inteligente + 1 GALE | Exportação CSV/JSON | feito com ❤️</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()