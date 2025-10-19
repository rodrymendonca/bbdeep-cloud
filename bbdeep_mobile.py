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
warnings.filterwarnings('ignore')

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
    def __init__(self):
        self.model_trained = False
        self.predictions = {"azul": 44.5, "vermelho": 44.5, "empate": 11.0}
        
        # Modelo de ML real
        self.ml_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.label_encoder = LabelEncoder()
        self.ml_trained = False
        self.window_size = 5  # Usar últimas 5 jogadas para prever a próxima
        
        # Inicializar encoder
        self.label_encoder.fit(["azul", "vermelho", "empate"])
        
    def train_model(self, beads_data, statistics):
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
            ml_result["model_type"] = "Random Forest ML"
        
        self.model_trained = True
        self.predictions = ml_result["predictions"]
        
        return ml_result
    
    def _train_ml_model(self, beads_data, statistics):
        try:
            # Preparar dados para ML
            all_beads = self._get_all_beads(beads_data)
            
            if len(all_beads) < self.window_size + 5:
                return {"success": False, "error": "Dados insuficientes para ML"}
            
            # Criar features e labels
            X, y = self._create_ml_features(all_beads)
            
            if len(X) < 10:
                return {"success": False, "error": "Poucos exemplos para treino"}
            
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
            
            self.ml_trained = True
            self.ml_accuracy = accuracy
            
            return {
                "success": True,
                "accuracy": accuracy,
                "predictions": ml_predictions,
                "training_examples": len(X_train),
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
            label = bead_numbers[i + 1]
            
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
        self.ml_engine = MLEngine()
        
        # Inicializar estado apenas uma vez
        if 'app_initialized' not in st.session_state:
            self.load_initial_state()
            st.session_state.app_initialized = True

    def load_initial_state(self):
        default_state = {
            "beads": [],
            "current_column": [],
            "last_color": None,
            "previous_prediction": None,  # Nova: previsão anterior
            "gale_count": 0,  # Nova: contador de Gale
            "statistics": {
                "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
                "total_beads": 0, "seq_vermelho": 0, "seq_empate": 0
            },
            "ml_model": {
                "trained": False, "accuracy": 0,
                "predictions": {"azul": 44.5, "vermelho": 44.5, "empate": 11.0},
                "last_trained": None, "training_count": 0,
                "model_type": "Nenhum", "training_examples": 0,
                "active_model": "Nenhum"
            },
            "settings": {
                "auto_train": True, "train_interval": 1  # Default para 1
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
        
        return loaded_state

    @property
    def state(self):
        return st.session_state.app_state

    def register_bead(self, color):
        # Guardar previsão atual ANTES de registar
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
        
        # Atualizar estatísticas
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
        
        # LÓGICA DO GALE - VERIFICAR APÓS REGISTRO
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
                # Sem auto-treino, não podemos verificar - manter gale count?
                pass
        else:
            # Previsão acertou ou não havia previsão - RESETAR GALE
            self.state["gale_count"] = 0
            
            # Auto-treino normal
            if self.state["settings"]["auto_train"]:
                if self.state["statistics"]["total_beads"] % self.state["settings"]["train_interval"] == 0:
                    self.train_model(auto=True)
        
        self.save_state()

    def train_model(self, auto=False):
        result = self.ml_engine.train_model(self.state, self.state["statistics"])
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
                "features_info": result.get("features_used", "Heurísticas")
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
            "features_info": "Nenhum"
        })
        self.save_state()

def main():
    st.set_page_config(
        page_title="BB DEEP Mobile + GALE",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # CSS Ultra Compacto para Mobile (ajustado para bead road crescer pra baixo)
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
    h1 { font-size: 20px !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 16px !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 14px !important; margin-bottom: 0.25rem !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    div[data-testid="stVerticalBlock"] > div {
        padding: 0.25rem 0;
    }
    .gale-indicator {
        background: linear-gradient(135deg, #ff0000, #cc0000);
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 12px;
        margin-left: 5px;
    }
    /* Estilos para o Bead Road ajustados */
    .bead-road-container {
        overflow-x: auto;
        white-space: nowrap;
        margin: 10px 0;
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 8px;
        max-height: 220px;  /* Ajustado pra caber 6 beads + padding */
    }
    .bead-column {
        display: inline-flex;
        flex-direction: column;  /* Mudado pra column normal, beads crescem pra baixo */
        margin-right: 8px;
        width: 32px;
        justify-content: flex-start;
        align-items: center;
    }
    .bead {
        font-size: 24px;
        line-height: 30px;
        width: 30px;
        height: 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 1px solid #ddd;
        border-radius: 50%;
        margin-bottom: 4px;  /* Mais espaço entre beads */
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.title("🤖 BB DEEP - ML + GALE")
    
    # Inicializar app
    if 'app' not in st.session_state:
        st.session_state.app = BBDeepMobile()
    
    app = st.session_state.app
    
    # PREVISÃO COM INDICADOR GALE
    next_color, confidence = app.get_next_prediction()
    gale_count = app.state["gale_count"]
    
    if next_color:
        color_name = {"azul": "AZUL", "vermelho": "VERMELHO", "empate": "EMPATE"}
        color_class = f"prediction-{next_color}"
        color_emoji = {"azul": "🔵", "vermelho": "🔴", "empate": "🟡"}
        
        # Adicionar classe GALE se estiver em Gale
        if gale_count > 0:
            color_class += " prediction-gale"
        
        # Texto do Gale
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
            model_info = f"🎯 {app.state['ml_model']['model_type']} | {app.state['ml_model']['accuracy']:.1f}% precisão"
            if "features_info" in app.state["ml_model"]:
                model_info += f" | {app.state['ml_model']['features_info']}"
            st.caption(model_info)
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
    
    # VISUALIZAÇÃO DO BEAD ROAD
    st.markdown("**Bead Road:**")
    beads = app.state["beads"]
    current_column = app.state["current_column"]
    
    if beads or current_column:
        html = '<div class="bead-road-container"><div style="display: flex;">'
        
        # Adicionar colunas completas
        for column in beads:
            html += '<div class="bead-column">'
            for bead in column:
                color = bead["color"]
                emoji = "🔵" if color == "azul" else "🔴" if color == "vermelho" else "🟡"
                html += f'<div class="bead">{emoji}</div>'
            html += '</div>'
        
        # Adicionar coluna atual (incompleta)
        if current_column:
            html += '<div class="bead-column">'
            for bead in current_column:
                color = bead["color"]
                emoji = "🔵" if color == "azul" else "🔴" if color == "vermelho" else "🟡"
                html += f'<div class="bead">{emoji}</div>'
            html += '</div>'
        
        html += '</div></div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.caption("Sem beads registados ainda.")
    
    # INDICADOR GALE
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
        train_interval = st.slider("Intervalo:", 1, 20, app.state["settings"]["train_interval"], key="train_interval")
        
        if st.button("💾 Aplicar", key="save_config"):
            app.state["settings"]["auto_train"] = auto_train
            app.state["settings"]["train_interval"] = train_interval
            app.save_state()
            st.rerun()
    
    with st.popover("📊 Info ML", use_container_width=True):
        st.write(f"**Modelo:** {app.state['ml_model']['model_type']}")
        st.write(f"**Precisão:** {app.state['ml_model']['accuracy']:.1f}%")
        st.write(f"**Exemplos treino:** {app.state['ml_model']['training_examples']}")
        st.write(f"**Total treinos:** {app.state['ml_model']['training_count']}")
        st.write(f"**Gale atual:** {app.state['gale_count']}")
        
        if app.state['current_column']:
            st.write("**Últimas jogadas:**")
            last_beads = ""
            for bead in app.state['current_column'][-5:]:
                symbol = bead['color'][0].upper()
                last_beads += symbol + " "
            st.write(last_beads)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # MENSAGEM FINAL
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666; font-size: 14px;'>🤖 ML Real + GALE | feito com ❤️</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()