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
        
        # Sistema de odds
        self.ODDS = {
            'azul': 2.0,
            'vermelho': 2.0,
            'empate': 5.0
        }
        self.TIE_PAYOUTS = {
            2: 88, 12: 88,
            3: 25, 11: 25,
            4: 10, 10: 10,
            5: 6, 9: 6,
            6: 4, 7: 4, 8: 4
        }
        
        # Configurações do Gale
        self.max_gale_streak = 3
        self.gale_suggestion_threshold = 45
        
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
            st.success("✅ Estado anterior carregado!")
        else:
            st.session_state.app_state = default_state
            st.info("🆕 Novo estado criado")
    
    def ensure_state_compatibility(self, loaded_state):
        default_stats = {
            "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
            "total_beads": 0, "bets_count": 0, "bets_won": 0, "bets_lost": 0,
            "max_win": 0, "max_win_percent": 0, "seq_vermelho": 0, "seq_empate": 0,
            "profit": 0, "total_wagered": 0, "total_won": 0,
            "gale_suggestions": 0, "gale_accepted": 0, "gale_won": 0, "gale_lost": 0
        }
        
        if "statistics" not in loaded_state:
            loaded_state["statistics"] = default_stats
        else:
            for key, default_value in default_stats.items():
                if key not in loaded_state["statistics"]:
                    loaded_state["statistics"][key] = default_value
        
        # Garantir outras chaves necessárias
        for key in ["beads", "current_column", "last_color", "bank", "bets", "bet_history"]:
            if key not in loaded_state:
                loaded_state[key] = [] if key.endswith('s') else None
        
        if "ml_model" not in loaded_state:
            loaded_state["ml_model"] = {
                "trained": False, "accuracy": 0,
                "predictions": {"azul": 44.5, "vermelho": 44.5, "empate": 11.0},
                "last_trained": None, "training_count": 0,
                "model_type": "Nenhum", "training_examples": 0,
                "active_model": "Nenhum"
            }
        
        if "settings" not in loaded_state:
            loaded_state["settings"] = {"auto_train": True, "train_interval": 5}
        
        return loaded_state

    def register_bead(self, color, tie_sum=None):
        bead = {"color": color}
        if color == "empate" and tie_sum:
            bead["tie_sum"] = tie_sum
        
        # Lógica de registo
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
        color = bead["color"]
        tie_sum = bead.get("tie_sum")
        resolved_bets = []
        
        for bet in list(self.state["bets"]):
            self.state["statistics"]["bets_count"] += 1
            self.state["statistics"]["total_wagered"] += bet["amount"]
            
            if bet["color"] == color:
                # Aposta ganhou
                if color != "empate":
                    payout = bet["amount"] * self.ODDS[color]
                else:
                    payout_factor = (self.TIE_PAYOUTS.get(tie_sum, 4) / 100) + 1
                    payout = bet["amount"] * payout_factor
                
                profit = payout - bet["amount"]
                self.state["bank"] += payout
                self.state["statistics"]["profit"] += profit
                self.state["statistics"]["total_won"] += payout
                self.state["statistics"]["bets_won"] += 1
                
                if profit > self.state["statistics"]["max_win"]:
                    self.state["statistics"]["max_win"] = profit
                    self.state["statistics"]["max_win_percent"] = (profit / bet["amount"]) * 100
                
                bet["outcome"] = "won"
                bet["payout"] = payout
                
            elif color == "empate" and bet["color"] in ["azul", "vermelho"]:
                # Empate - devolve aposta
                self.state["bank"] += bet["amount"]
                bet["outcome"] = "push"
                bet["payout"] = bet["amount"]
            else:
                # Aposta perdeu
                self.state["statistics"]["bets_lost"] += 1
                self.state["statistics"]["profit"] -= bet["amount"]
                bet["outcome"] = "lost"
                bet["payout"] = 0
            
            bet["resolved_at"] = datetime.now().isoformat()
            resolved_bets.append(bet)
        
        # Mover apostas resolvidas para histórico
        self.state["bet_history"].extend(resolved_bets)
        self.state["bets"] = [b for b in self.state["bets"] if b not in resolved_bets]
        
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
                "active_model": result["model_type"]
            })
            self.save_state()
            return True
        return False

    def save_state(self):
        success = self.data_manager.save_data(self.state, "app_state.json")
        if success:
            st.success("💾 Estado guardado!")
        else:
            st.error("❌ Erro ao guardar estado")

    def reset_beads(self):
        self.state["beads"] = []
        self.state["current_column"] = []
        self.state["last_color"] = None
        self.state["statistics"].update({
            "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
            "total_beads": 0, "seq_vermelho": 0, "seq_empate": 0
        })
        self.save_state()

    def reset_bets(self):
        total_active = sum(bet["amount"] for bet in self.state["bets"])
        self.state["bank"] += total_active
        
        self.state["statistics"].update({
            "bets_count": 0, "bets_won": 0, "bets_lost": 0,
            "max_win": 0, "max_win_percent": 0, "profit": 0,
            "total_wagered": 0, "total_won": 0
        })
        
        self.state["bets"] = []
        self.state["bet_history"] = []
        self.save_state()

    def reset_all(self):
        default_state = {
            "beads": [], "current_column": [], "last_color": None,
            "bank": 100.0, "bets": [], "bet_history": [],
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
            "settings": self.state["settings"],
            "gale_active": False, "current_gale_streak": 0, "last_gale_color": None
        }
        
        self.state.update(default_state)
        self.save_state()

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
    .bead-display {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin: 10px 0;
    }
    .bead {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
    }
    .bead-azul { background-color: #2196f3; }
    .bead-vermelho { background-color: #f44336; }
    .bead-empate { background-color: #ffc107; color: black; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎰 BB DEEP - Mobile")
    
    # Inicializar app
    app = BBDeepMobile()
    
    # Layout para mobile
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Beads & Apostas", "📊 Estatísticas", "🤖 ML & Config", "💾 Dados"])
    
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
                    tie_sum = st.selectbox("Soma do empate:", [2,3,4,5,6,7,8,9,10,11,12], key="tie_sum")
                    if st.button("✅ Confirmar Empate", use_container_width=True, key="confirm_empate"):
                        app.register_bead('empate', tie_sum)
                        st.rerun()
            
            # Display beads atual
            st.subheader("Coluna Atual")
            if app.state["current_column"]:
                beads_html = '<div class="bead-display">'
                for bead in app.state["current_column"]:
                    color_class = f"bead-{bead['color']}"
                    letter = bead['color'][0].upper()
                    beads_html += f'<div class="bead {color_class}">{letter}</div>'
                beads_html += '</div>'
                st.markdown(beads_html, unsafe_allow_html=True)
            else:
                st.info("Nenhum bead registado")
            
            # Histórico de colunas
            if app.state["beads"]:
                st.subheader("Histórico de Colunas")
                for i, column in enumerate(reversed(app.state["beads"][-5:])):  # Últimas 5 colunas
                    with st.expander(f"Coluna {len(app.state['beads'])-i}"):
                        beads_html = '<div class="bead-display">'
                        for bead in column:
                            color_class = f"bead-{bead['color']}"
                            letter = bead['color'][0].upper()
                            beads_html += f'<div class="bead {color_class}">{letter}</div>'
                        beads_html += '</div>'
                        st.markdown(beads_html, unsafe_allow_html=True)
        
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
                    color_emoji = "🔵" if bet['color'] == 'azul' else "🔴" if bet['color'] == 'vermelho' else "🟡"
                    st.write(f"{color_emoji} {bet['color']}: {bet['amount']:.1f}x")
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
            
            if app.state['statistics']['bets_count'] > 0:
                win_rate = (app.state['statistics']['bets_won'] / app.state['statistics']['bets_count']) * 100
                st.metric("Taxa de Vitória", f"{win_rate:.1f}%")
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Machine Learning")
            
            if st.button("🎯 Treinar Modelo", use_container_width=True):
                if app.train_model(auto=False):
                    st.success("Modelo treinado com sucesso!")
                else:
                    st.error("Erro ao treinar modelo")
                st.rerun()
            
            if app.state["ml_model"]["trained"]:
                st.success(f"✅ {app.state['ml_model']['model_type']}")
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
                
                # Próxima previsão
                max_color = max(pred, key=pred.get)
                st.info(f"🎯 Próxima previsão: **{max_color.upper()}** ({pred[max_color]:.1f}%)")
            else:
                st.warning("🤖 Modelo não treinado")
        
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
    
    with tab4:
        st.subheader("Gestão de Dados")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Reset Beads", use_container_width=True):
                app.reset_beads()
                st.success("Beads resetados!")
                st.rerun()
        with col2:
            if st.button("💰 Reset Apostas", use_container_width=True):
                app.reset_bets()
                st.success("Apostas resetadas!")
                st.rerun()
        with col3:
            if st.button("💥 Reset Tudo", use_container_width=True):
                if st.checkbox("Confirmar reset completo"):
                    app.reset_all()
                    st.success("Reset completo realizado!")
                    st.rerun()
        
        st.subheader("Backup & Restauro")
        if st.button("💾 Backup Manual", use_container_width=True):
            app.save_state()
        
        # Info do estado atual
        st.subheader("Informação do Estado")
        st.write(f"**Total de Beads:** {app.state['statistics']['total_beads']}")
        st.write(f"**Colunas guardadas:** {len(app.state['beads'])}")
        st.write(f"**Apostas no histórico:** {len(app.state['bet_history'])}")
        st.write(f"**Último treino ML:** {app.state['ml_model'].get('last_trained', 'Nunca')}")

if __name__ == "__main__":
    main()