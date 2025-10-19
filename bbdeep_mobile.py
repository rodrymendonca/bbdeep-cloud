import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
import os

# Adicionar path para importar módulos
sys.path.append(os.path.dirname(__file__))

try:
    from data_manager import DataManager
    from ml_engine import MLEngine
except ImportError:
    # Para debug local
    from data_manager import DataManager
    from ml_engine import MLEngine

class BBDeepMobile:
    def __init__(self):
        self.data_manager = DataManager()
        self.ml_engine = MLEngine()
        
        # Estado da aplicação
        if 'app_state' not in st.session_state:
            self.load_initial_state()
        
        self.state = st.session_state.app_state
    
    def load_initial_state(self):
        default_state = {
            "beads": [],
            "current_column": [],
            "last_color": None,
            "bank": 100.0,
            "bets": [],
            "bet_history": [],
            "statistics": {
                "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
                "total_beads": 0, "bets_count": 0, "bets_won": 0, "bets_lost": 0,
                "max_win": 0, "max_win_percent": 0, "seq_vermelho": 0, "seq_empate": 0,
                "profit": 0, "total_wagered": 0, "total_won": 0,
                "gale_suggestions": 0, "gale_accepted": 0, "gale_won": 0, "gale_lost": 0
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
            },
            "gale_active": False,
            "current_gale_streak": 0,
            "last_gale_color": None
        }
        
        # Tentar carregar estado guardado
        loaded = self.data_manager.load_data("app_state.json")
        if loaded:
            st.session_state.app_state = self.ensure_state_compatibility(loaded)
        else:
            st.session_state.app_state = default_state
    
    def ensure_state_compatibility(self, loaded_state):
        # (Manter a mesma função do código original)
        default_stats = {
            "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
            "total_beads": 0, "bets_count": 0, "bets_won": 0, "bets_lost": 0,
            "max_win": 0, "max_win_percent": 0, "seq_vermelho": 0, "seq_empate": 0,
            "profit": 0, "total_wagered": 0, "total_won": 0,
            "gale_suggestions": 0, "gale_accepted": 0, "gale_won": 0, "gale_lost": 0
        }
        
        if "statistics" in loaded_state:
            for key, default_value in default_stats.items():
                if key not in loaded_state["statistics"]:
                    loaded_state["statistics"][key] = default_value
        
        return loaded_state

    def register_bead(self, color, tie_sum=None):
        bead = {"color": color}
        if color == "empate" and tie_sum:
            bead["tie_sum"] = tie_sum
        
        # Lógica de registo (similar à original)
        if self.state["last_color"] == bead["color"] and len(self.state["current_column"]) < 6:
            self.state["current_column"].append(bead)
        else:
            if self.state["current_column"]:
                self.state["beads"].append(self.state["current_column"])
            self.state["current_column"] = [bead]
        
        self.state["last_color"] = bead["color"]
        self.state["statistics"]["total_beads"] += 1
        
        # Atualizar estatísticas
        if bead["color"] == "azul":
            self.state["statistics"]["azul_count"] += 1
            self.state["statistics"]["seq_vermelho"] = 0
            self.state["statistics"]["seq_empate"] = 0
        elif bead["color"] == "vermelho":
            self.state["statistics"]["vermelho_count"] += 1
            self.state["statistics"]["seq_vermelho"] += 1
            self.state["statistics"]["seq_empate"] = 0
        else:
            self.state["statistics"]["empate_count"] += 1
            self.state["statistics"]["seq_empate"] += 1
            self.state["statistics"]["seq_vermelho"] = 0
        
        self.resolve_bets(bead)
        self.save_state()
        
        # Auto-train se configurado
        if self.state["settings"]["auto_train"]:
            if self.state["statistics"]["total_beads"] % self.state["settings"]["train_interval"] == 0:
                self.train_model(auto=True)

    def place_bet(self, color, amount):
        if amount <= 0 or amount > self.state["bank"]:
            st.error("Valor de aposta inválido")
            return False
        
        bet = {
            "id": datetime.now().timestamp(),
            "color": color,
            "amount": amount,
            "placed_at": datetime.now().isoformat(),
            "gale_count": 0
        }
        
        self.state["bets"].append(bet)
        self.state["bank"] -= amount
        self.save_state()
        return True

    def resolve_bets(self, bead):
        # (Implementar lógica de resolução de apostas similar à original)
        color = bead["color"]
        tie_sum = bead.get("tie_sum")
        
        ODDS = {'azul': 2.0, 'vermelho': 2.0, 'empate': 5.0}
        TIE_PAYOUTS = {2: 88, 12: 88, 3: 25, 11: 25, 4: 10, 10: 10, 5: 6, 9: 6, 6: 4, 7: 4, 8: 4}
        
        for bet in list(self.state["bets"]):
            # Lógica de payout...
            pass
        
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
                "training_examples": result.get("training_examples", 0)
            })
            self.save_state()

    def save_state(self):
        self.data_manager.save_data(self.state, "app_state.json")

def main():
    st.set_page_config(
        page_title="BB DEEP Mobile",
        page_icon="🎰",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # CSS para mobile
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .main > div {
            padding: 0.5rem;
        }
        .stButton button {
            width: 100%;
            margin: 2px 0;
            font-size: 14px;
        }
        .stMetric {
            padding: 0.5rem;
        }
    }
    .bead-azul { color: #2196f3; font-weight: bold; }
    .bead-vermelho { color: #f44336; font-weight: bold; }
    .bead-empate { color: #ffc107; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎰 BB DEEP - Mobile")
    
    # Inicializar app
    app = BBDeepMobile()
    
    # Layout para mobile
    tab1, tab2, tab3 = st.tabs(["🎯 Beads & Apostas", "📊 Estatísticas", "🤖 ML & Config"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Registar Beads")
            
            # Botões grandes para mobile
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("🔵 AZUL", use_container_width=True, key="btn_azul"):
                    app.register_bead('azul')
                    st.rerun()
            with btn_col2:
                if st.button("🔴 VERMELHO", use_container_width=True, key="btn_vermelho"):
                    app.register_bead('vermelho')
                    st.rerun()
            with btn_col3:
                if st.button("🟡 EMPATE", use_container_width=True, key="btn_empate"):
                    # Para empate, pedir a soma
                    tie_sum = st.selectbox("Soma do empate:", [2,3,4,5,6,7,8,9,10,11,12], key="tie_sum")
                    if st.button("Confirmar Empate", use_container_width=True, key="confirm_empate"):
                        app.register_bead('empate', tie_sum)
                        st.rerun()
            
            # Display beads atual
            st.subheader("Beads Atuais")
            if app.state["current_column"]:
                beads_display = " | ".join([f"<span class='bead-{bead["color"]}'>{bead['color'][0].upper()}</span>" for bead in app.state["current_column"]])
                st.markdown(beads_display, unsafe_allow_html=True)
            else:
                st.info("Nenhum bead registado")
        
        with col2:
            st.subheader("Fazer Aposta")
            
            bet_amount = st.number_input("Valor:", value=5.0, min_value=0.1, step=0.5, key="bet_amount")
            
            bet_col1, bet_col2, bet_col3 = st.columns(3)
            with bet_col1:
                if st.button("Apostar 🔵", use_container_width=True, key="bet_azul"):
                    if app.place_bet('azul', bet_amount):
                        st.success(f"Aposta de {bet_amount} em AZUL colocada!")
                        st.rerun()
            with bet_col2:
                if st.button("Apostar 🔴", use_container_width=True, key="bet_vermelho"):
                    if app.place_bet('vermelho', bet_amount):
                        st.success(f"Aposta de {bet_amount} em VERMELHO colocada!")
                        st.rerun()
            with bet_col3:
                if st.button("Apostar 🟡", use_container_width=True, key="bet_empate"):
                    if app.place_bet('empate', bet_amount):
                        st.success(f"Aposta de {bet_amount} em EMPATE colocada!")
                        st.rerun()
            
            # Apostas ativas
            st.subheader("Apostas Ativas")
            if app.state["bets"]:
                for bet in app.state["bets"]:
                    st.write(f"• {bet['color']}: {bet['amount']:.1f}")
            else:
                st.info("Nenhuma aposta ativa")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Banca")
            st.metric("Saldo", f"{app.state['bank']:.1f}x")
            st.metric("Lucro", f"{app.state['statistics']['profit']:+.1f}x")
            st.metric("Total Apostado", f"{app.state['statistics']['total_wagered']:.1f}x")
            
            st.subheader("Apostas")
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.metric("Total", app.state['statistics']['bets_count'])
            with col1b:
                st.metric("Ganhas", app.state['statistics']['bets_won'])
            with col1c:
                st.metric("Perdidas", app.state['statistics']['bets_lost'])
        
        with col2:
            st.subheader("Beads")
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("🔵 Azul", app.state['statistics']['azul_count'])
            with col2b:
                st.metric("🔴 Vermelho", app.state['statistics']['vermelho_count'])
            with col2c:
                st.metric("🟡 Empate", app.state['statistics']['empate_count'])
            
            st.metric("Total Beads", app.state['statistics']['total_beads'])
            st.metric("Seq. Vermelho", app.state['statistics']['seq_vermelho'])
            st.metric("Seq. Empate", app.state['statistics']['seq_empate'])
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Machine Learning")
            
            if st.button("🎯 Treinar Modelo", use_container_width=True):
                app.train_model(auto=False)
                st.rerun()
            
            if app.state["ml_model"]["trained"]:
                st.success(f"Modelo: {app.state['ml_model']['model_type']}")
                st.metric("Precisão", f"{app.state['ml_model']['accuracy']:.1f}%")
                
                st.subheader("Previsões Atuais")
                pred = app.state["ml_model"]["predictions"]
                col1a, col1b, col1c = st.columns(3)
                with col1a:
                    st.metric("🔵 Azul", f"{pred['azul']:.1f}%")
                with col1b:
                    st.metric("🔴 Vermelho", f"{pred['vermelho']:.1f}%")
                with col1c:
                    st.metric("🟡 Empate", f"{pred['empate']:.1f}%")
            else:
                st.warning("Modelo não treinado")
        
        with col2:
            st.subheader("Configurações")
            
            auto_train = st.checkbox("Auto-treino", value=app.state["settings"]["auto_train"])
            train_interval = st.number_input("Intervalo de treino (beads)", 
                                           value=app.state["settings"]["train_interval"],
                                           min_value=1, max_value=50)
            
            if st.button("💾 Guardar Configurações", use_container_width=True):
                app.state["settings"]["auto_train"] = auto_train
                app.state["settings"]["train_interval"] = train_interval
                app.save_state()
                st.success("Configurações guardadas!")
            
            st.subheader("Gestão de Dados")
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("🔄 Reset Beads", use_container_width=True):
                    app.state["beads"] = []
                    app.state["current_column"] = []
                    app.state["last_color"] = None
                    app.save_state()
                    st.rerun()
            with col2b:
                if st.button("🗑️ Reset Apostas", use_container_width=True):
                    app.state["bets"] = []
                    app.state["bet_history"] = []
                    app.save_state()
                    st.rerun()

if __name__ == "__main__":
    main()