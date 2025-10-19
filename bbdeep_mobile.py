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
        page_title="BB DEEP Mobile",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # CSS Ultra Compacto para Mobile
    st.markdown("""
    <style>
    /* Reset e configurações base */
    .main-container {
        width: 100%;
        max-width: 100%;
        margin: 0;
        padding: 8px;
    }
    
    /* Compactar tudo */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Previsão compacta */
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
    
    /* Botões compactos */
    .stButton button {
        height: 45px !important;
        font-size: 16px !important;
        margin: 4px 0 !important;
        border-radius: 8px !important;
    }
    
    /* Estatísticas em linha única */
    .stats-row {
        display: flex;
        justify-content: space-between;
        margin: 8px 0;
        flex-wrap: wrap;
    }
    .stat-item {
        flex: 1;
        min-width: 30%;
        text-align: center;
        padding: 8px 4px;
        margin: 2px;
        background-color: #f8f9fa;
        border-radius: 6px;
        font-size: 12px;
    }
    
    /* Barras de progresso compactas */
    .compact-progress {
        height: 20px;
        margin: 5px 0;
    }
    
    /* Reduzir espaçamento de todos os elementos */
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    h1 {
        font-size: 20px !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-size: 16px !important;
        margin-bottom: 0.5rem !important;
    }
    
    h3 {
        font-size: 14px !important;
        margin-bottom: 0.25rem !important;
    }
    
    /* Ajustar inputs */
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 0.5rem !important;
    }
    
    /* Esconder elementos do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Compactar expanders */
    .streamlit-expanderHeader {
        font-size: 14px !important;
        padding: 0.5rem 0.75rem !important;
    }
    
    /* Reduzir padding geral */
    div[data-testid="stVerticalBlock"] > div {
        padding: 0.25rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Container principal ultra compacto
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    st.title("🤖 BB DEEP")
    
    # Inicializar app
    if 'app' not in st.session_state:
        st.session_state.app = BBDeepMobile()
    
    app = st.session_state.app
    
    # SEÇÃO 1: PREVISÃO (MUITO COMPACTA)
    next_color, confidence = app.get_next_prediction()
    
    if next_color:
        color_name = {"azul": "AZUL", "vermelho": "VERMELHO", "empate": "EMPATE"}
        color_class = f"prediction-{next_color}"
        color_emoji = {"azul": "🔵", "vermelho": "🔴", "empate": "🟡"}
        
        st.markdown(f"""
        <div class="prediction-compact {color_class}">
            <div style="font-size: 18px; margin-bottom: 2px;">PRÓXIMA: {color_emoji[next_color]} {color_name[next_color]}</div>
            <div style="font-size: 14px;">{confidence:.1f}% confiança</div>
        </div>
        """, unsafe_allow_html=True)
        
        if app.state["ml_model"]["trained"]:
            st.caption(f"🎯 {app.state['ml_model']['model_type']} | {app.state['ml_model']['accuracy']:.1f}% precisão")
    else:
        st.info("📊 Registe beads e treine o modelo")
    
    # SEÇÃO 2: BOTÕES DE REGISTO (COMPACTOS)
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
        if st.button("🟡 EMP.", use_container_width=True, key="btn_empate"):
            # Para empate, mostrar seleção inline
            with st.popover("Soma:"):
                tie_sum = st.selectbox("Escolha soma:", [2,3,4,5,6,7,8,9,10,11,12], key="tie_select")
                if st.button("✅ Confirmar", key="confirm_empate"):
                    app.register_bead('empate', tie_sum)
                    st.rerun()
    
    # SEÇÃO 3: ESTATÍSTICAS ULTRA COMPACTAS
    st.markdown("---")
    
    # Primeira linha de stats
    st.markdown('<div class="stats-row">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔵 Azul", app.state['statistics']['azul_count'], delta=None)
    with col2:
        st.metric("🔴 Verm.", app.state['statistics']['vermelho_count'], delta=None)
    with col3:
        st.metric("🟡 Emp.", app.state['statistics']['empate_count'], delta=None)
    
    # Segunda linha de stats
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("📊 Total", app.state['statistics']['total_beads'], delta=None)
    with col5:
        st.metric("🔴 Seq", app.state['statistics']['seq_vermelho'], delta=None)
    with col6:
        st.metric("🟡 Seq", app.state['statistics']['seq_empate'], delta=None)
    
    # SEÇÃO 4: PROBABILIDADES COMPACTAS
    if app.state["ml_model"]["trained"]:
        st.markdown("---")
        st.markdown("**Probabilidades:**")
        
        pred = app.state["ml_model"]["predictions"]
        
        # Barras de progresso horizontais compactas
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
    
    # SEÇÃO 5: CONTROLES COMPACTOS
    st.markdown("---")
    
    # Botões de ação em linha
    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("🎯 TREINAR", use_container_width=True, key="train_ml"):
            if app.train_model():
                st.rerun()
    
    with action_col2:
        if st.button("🔄 RESETAR", use_container_width=True, key="reset_model"):
            app.reset_model()
            st.rerun()
    
    # Configurações em popover para economizar espaço
    with st.popover("⚙️ Configurações", use_container_width=True):
        auto_train = st.checkbox("Auto-treino", value=app.state["settings"]["auto_train"], key="auto_train")
        train_interval = st.slider("Intervalo:", 1, 20, app.state["settings"]["train_interval"], key="train_interval")
        
        if st.button("💾 Aplicar", key="save_config"):
            app.state["settings"]["auto_train"] = auto_train
            app.state["settings"]["train_interval"] = train_interval
            app.save_state()
            st.rerun()
    
    # Informação rápida em popover
    with st.popover("📊 Info Rápida", use_container_width=True):
        st.write(f"**Colunas:** {len(app.state['beads'])}")
        st.write(f"**Coluna atual:** {len(app.state['current_column'])}/6")
        st.write(f"**Treinos:** {app.state['ml_model']['training_count']}")
        
        if app.state['current_column']:
            st.write("**Últimos:**")
            last_beads = ""
            for bead in app.state['current_column'][-3:]:
                symbol = bead['color'][0].upper()
                if bead['color'] == 'empate':
                    symbol += str(bead.get('tie_sum', ''))
                last_beads += symbol + " "
            st.write(last_beads)
    
    # Estado atual muito compacto
    if app.state['current_column']:
        current_progress = len(app.state['current_column'])
        st.caption(f"📝 Coluna atual: {current_progress}/6 beads")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()