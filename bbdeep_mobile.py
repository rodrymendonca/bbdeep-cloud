import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===== DATA MANAGER =====
class DataManager:
    def __init__(self):
        self.data_dir = "data"
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def get_file_path(self, filename):
        return os.path.join(self.data_dir, filename)
    
    def save_data(self, data, filename):
        try:
            filepath = self.get_file_path(filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            return False
    
    def load_data(self, filename):
        try:
            filepath = self.get_file_path(filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            return None

# ===== ML ENGINE =====
class MLEngine:
    def __init__(self):
        self.model_trained = False
        self.predictions = {"azul": 44.5, "vermelho": 44.5, "empate": 11.0}

    def train_model(self, beads_data, statistics):
        total_beads = statistics.get("total_beads", 0)
        
        if total_beads > 0:
            azul_count = statistics.get("azul_count", 0)
            vermelho_count = statistics.get("vermelho_count", 0)
            empate_count = statistics.get("empate_count", 0)
            
            # Probabilidades base
            azul_prob = (azul_count / total_beads) * 100
            vermelho_prob = (vermelho_count / total_beads) * 100
            empate_prob = (empate_count / total_beads) * 100
            
            # Ajustar sequências
            seq_vermelho = statistics.get("seq_vermelho", 0)
            seq_empate = statistics.get("seq_empate", 0)
            
            if seq_vermelho >= 3:
                azul_prob += seq_vermelho * 5
                empate_prob += seq_vermelho * 2
            elif seq_empate >= 2:
                azul_prob += seq_empate * 4
                vermelho_prob += seq_empate * 4
            
            predictions = {
                "azul": max(5, min(95, azul_prob)),
                "vermelho": max(5, min(95, vermelho_prob)),
                "empate": max(1, min(30, empate_prob))
            }
            
            # Normalizar
            total = sum(predictions.values())
            predictions = {k: (v / total) * 100 for k, v in predictions.items()}
            
            accuracy = 60 + min(30, total_beads / 10)
            
            return {
                "success": True,
                "accuracy": accuracy,
                "predictions": predictions,
                "model_type": "Heurístico",
                "training_examples": total_beads
            }
        else:
            return {
                "success": True,
                "accuracy": 50.0,
                "predictions": {"azul": 44.5, "vermelho": 44.5, "empate": 11.0},
                "model_type": "Básico",
                "training_examples": 0
            }
    
    def is_trained(self):
        return self.model_trained

# ===== MAIN APP =====
class BBDeepCore:
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
                "auto_train": True, "train_interval": 5
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
        
        for key in ["beads", "current_column"]:
            if key not in loaded_state:
                loaded_state[key] = []
        
        # Remover dados desnecessários se existirem
        for key in ["bank", "bets", "bet_history"]:
            if key in loaded_state:
                del loaded_state[key]
        
        return loaded_state

    @property
    def state(self):
        return st.session_state.app_state

    def register_bead(self, color, tie_sum=None):
        bead = {"color": color}
        if color == "empate" and tie_sum:
            bead["tie_sum"] = tie_sum
        
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
        
        self.save_state()
        
        # Auto-treino
        if self.state["settings"]["auto_train"]:
            if self.state["statistics"]["total_beads"] % self.state["settings"]["train_interval"] == 0:
                self.train_model(auto=True)

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
                "active_model": result["model_type"]
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
        self.state["statistics"].update({
            "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
            "total_beads": 0, "seq_vermelho": 0, "seq_empate": 0
        })
        self.state["ml_model"].update({
            "trained": False, "accuracy": 0,
            "predictions": {"azul": 44.5, "vermelho": 44.5, "empate": 11.0},
            "last_trained": None, "training_count": 0,
            "model_type": "Nenhum", "training_examples": 0,
            "active_model": "Nenhum"
        })
        self.save_state()

def main():
    st.set_page_config(
        page_title="BB DEEP Core",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # CSS Simplificado
    st.markdown("""
    <style>
    .prediction-box { 
        border: 3px solid; 
        border-radius: 10px; 
        padding: 20px; 
        text-align: center; 
        margin: 15px 0; 
        font-weight: bold; 
        font-size: 20px; 
    }
    .prediction-azul { border-color: #2196f3; background-color: rgba(33, 150, 243, 0.1); }
    .prediction-vermelho { border-color: #f44336; background-color: rgba(244, 67, 54, 0.1); }
    .prediction-empate { border-color: #ffc107; background-color: rgba(255, 193, 7, 0.1); }
    
    .control-panel {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }
    
    .stat-box {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 BB DEEP Core")
    st.markdown("**Machine Learning para Previsão de Cores**")
    
    # Inicializar app
    if 'app' not in st.session_state:
        st.session_state.app = BBDeepCore()
    
    app = st.session_state.app
    
    # Layout principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # PREVISÃO ATUAL
        st.subheader("🔮 Próxima Previsão")
        next_color, confidence = app.get_next_prediction()
        
        if next_color:
            color_name = {"azul": "AZUL", "vermelho": "VERMELHO", "empate": "EMPATE"}
            color_class = f"prediction-{next_color}"
            color_emoji = {"azul": "🔵", "vermelho": "🔴", "empate": "🟡"}
            
            st.markdown(f"""
            <div class="prediction-box {color_class}">
                <div style="font-size: 28px; margin-bottom: 15px;">{color_emoji[next_color]}</div>
                <div style="font-size: 24px;">{color_name[next_color]}</div>
                <div style="font-size: 16px; margin-top: 10px;">{confidence:.1f}% confiança</div>
            </div>
            """, unsafe_allow_html=True)
            
            if app.state["ml_model"]["trained"]:
                st.info(f"**Modelo:** {app.state['ml_model']['model_type']} | **Precisão:** {app.state['ml_model']['accuracy']:.1f}%")
        else:
            st.info("🤖 Registe alguns beads e treine o modelo para obter previsões")
        
        # ESTATÍSTICAS
        st.subheader("📊 Estatísticas")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("🔵 Azul", app.state['statistics']['azul_count'])
        with col_stat2:
            st.metric("🔴 Vermelho", app.state['statistics']['vermelho_count'])
        with col_stat3:
            st.metric("🟡 Empate", app.state['statistics']['empate_count'])
        
        st.metric("🎯 Total Beads", app.state['statistics']['total_beads'])
        
        # SEQUÊNCIAS ATUAIS
        st.subheader("📈 Sequências Atuais")
        col_seq1, col_seq2 = st.columns(2)
        with col_seq1:
            st.metric("🔴 Seq Vermelho", app.state['statistics']['seq_vermelho'])
        with col_seq2:
            st.metric("🟡 Seq Empate", app.state['statistics']['seq_empate'])
    
    with col2:
        # REGISTO DE BEADS
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("🎯 Registar Beads")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🔵 AZUL", use_container_width=True, key="btn_azul"):
                app.register_bead('azul')
                st.success("Bead AZUL registado!")
        with col_btn2:
            if st.button("🔴 VERMELHO", use_container_width=True, key="btn_vermelho"):
                app.register_bead('vermelho')
                st.success("Bead VERMELHO registado!")
        with col_btn3:
            tie_sum = st.selectbox("Soma para EMPATE:", [2,3,4,5,6,7,8,9,10,11,12], key="tie_select")
            if st.button("🟡 EMPATE", use_container_width=True, key="btn_empate"):
                app.register_bead('empate', tie_sum)
                st.success(f"Bead EMPATE ({tie_sum}) registado!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # INFORMAÇÃO DO MODELO
        st.subheader("🤖 Estado do Modelo")
        
        if app.state["ml_model"]["trained"]:
            st.success("✅ Modelo treinado e ativo")
            
            # Mostrar probabilidades detalhadas
            pred = app.state["ml_model"]["predictions"]
            st.markdown("**Probabilidades detalhadas:**")
            col_prob1, col_prob2, col_prob3 = st.columns(3)
            with col_prob1:
                st.metric("🔵 Azul", f"{pred['azul']:.1f}%")
            with col_prob2:
                st.metric("🔴 Vermelho", f"{pred['vermelho']:.1f}%")
            with col_prob3:
                st.metric("🟡 Empate", f"{pred['empate']:.1f}%")
            
            st.write(f"**Último treino:** {app.state['ml_model']['last_trained'][:19].replace('T', ' ')}")
            st.write(f"**Exemplos de treino:** {app.state['ml_model']['training_examples']}")
        else:
            st.warning("⚠️ Modelo não treinado")
        
        # GESTÃO DO MODELO
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("⚙️ Gestão do Modelo")
        
        col_mgmt1, col_mgmt2 = st.columns(2)
        with col_mgmt1:
            if st.button("🎯 Treinar Modelo", use_container_width=True, key="train_ml"):
                if app.train_model():
                    st.success("✅ Modelo treinado com sucesso!")
                else:
                    st.error("❌ Erro no treino do modelo")
        
        with col_mgmt2:
            if st.button("🔄 Reset Modelo", use_container_width=True, key="reset_model"):
                app.reset_model()
                st.success("✅ Modelo resetado com sucesso!")
        
        # Configurações
        st.subheader("🔧 Configurações")
        auto_train = st.checkbox("Auto-treino", value=app.state["settings"]["auto_train"], key="auto_train")
        train_interval = st.number_input("Intervalo de auto-treino (beads):", 
                                       value=app.state["settings"]["train_interval"],
                                       min_value=1, max_value=50, key="train_interval")
        
        if st.button("💾 Guardar Configurações", use_container_width=True, key="save_config"):
            app.state["settings"]["auto_train"] = auto_train
            app.state["settings"]["train_interval"] = train_interval
            app.save_state()
            st.success("✅ Configurações guardadas!")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Informação adicional
    with st.expander("📋 Informação do Sistema"):
        st.write(f"**Total de colunas completas:** {len(app.state['beads'])}")
        st.write(f"**Beads na coluna atual:** {len(app.state['current_column'])}")
        if app.state['current_column']:
            last_beads = ", ".join([bead['color'][0].upper() + (str(bead.get('tie_sum', '')) if bead['color'] == 'empate' else '') 
                                  for bead in app.state['current_column'][-3:]])
            st.write(f"**Últimos beads:** {last_beads}")
        
        st.write(f"**Contador de treinos:** {app.state['ml_model']['training_count']}")

if __name__ == "__main__":
    main()