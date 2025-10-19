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
        
        self.ODDS = {'azul': 2.0, 'vermelho': 2.0, 'empate': 5.0}
        self.TIE_PAYOUTS = {2: 88, 12: 88, 3: 25, 11: 25, 4: 10, 10: 10, 5: 6, 9: 6, 6: 4, 7: 4, 8: 4}
        
        # Inicializar estado apenas uma vez
        if 'app_initialized' not in st.session_state:
            self.load_initial_state()
            st.session_state.app_initialized = True

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
                "profit": 0, "total_wagered": 0, "total_won": 0
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
            "total_beads": 0, "bets_count": 0, "bets_won": 0, "bets_lost": 0,
            "max_win": 0, "max_win_percent": 0, "seq_vermelho": 0, "seq_empate": 0,
            "profit": 0, "total_wagered": 0, "total_won": 0
        }
        
        if "statistics" not in loaded_state:
            loaded_state["statistics"] = default_stats
        else:
            for key in default_stats:
                if key not in loaded_state["statistics"]:
                    loaded_state["statistics"][key] = default_stats[key]
        
        for key in ["beads", "current_column", "bets", "bet_history"]:
            if key not in loaded_state:
                loaded_state[key] = []
        
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
        
        self.resolve_bets(bead)
        self.save_state()
        
        # Auto-treino
        if self.state["settings"]["auto_train"]:
            if self.state["statistics"]["total_beads"] % self.state["settings"]["train_interval"] == 0:
                self.train_model(auto=True)

    def place_bet(self, color, amount):
        if amount <= 0 or amount > self.state["bank"]:
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
        
        for bet in list(self.state["bets"]):
            self.state["statistics"]["bets_count"] += 1
            self.state["statistics"]["total_wagered"] += bet["amount"]
            
            if bet["color"] == color:
                if color != "empate":
                    payout = bet["amount"] * self.ODDS[color]
                else:
                    payout_multiplier = (self.TIE_PAYOUTS.get(tie_sum, 4) / 100) + 1
                    payout = bet["amount"] * payout_multiplier
                
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
                self.state["bank"] += bet["amount"]
                bet["outcome"] = "push"
                bet["payout"] = bet["amount"]
            else:
                self.state["statistics"]["bets_lost"] += 1
                self.state["statistics"]["profit"] -= bet["amount"]
                bet["outcome"] = "lost"
                bet["payout"] = 0
            
            bet["resolved_at"] = datetime.now().isoformat()
            self.state["bet_history"].append(bet)
        
        self.state["bets"] = []

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
        for bet in self.state["bets"]:
            self.state["bank"] += bet["amount"]
        
        self.state["bets"] = []
        self.state["bet_history"] = []
        self.state["statistics"].update({
            "bets_count": 0, "bets_won": 0, "bets_lost": 0,
            "max_win": 0, "max_win_percent": 0, "profit": 0,
            "total_wagered": 0, "total_won": 0
        })
        self.save_state()

    def reset_all(self):
        default_state = {
            "beads": [], "current_column": [], "last_color": None,
            "bank": 100.0, "bets": [], "bet_history": [],
            "statistics": {
                "azul_count": 0, "vermelho_count": 0, "empate_count": 0,
                "total_beads": 0, "bets_count": 0, "bets_won": 0, "bets_lost": 0,
                "max_win": 0, "max_win_percent": 0, "seq_vermelho": 0, "seq_empate": 0,
                "profit": 0, "total_wagered": 0, "total_won": 0
            },
            "ml_model": {
                "trained": False, "accuracy": 0,
                "predictions": {"azul": 44.5, "vermelho": 44.5, "empate": 11.0},
                "last_trained": None, "training_count": 0,
                "model_type": "Nenhum", "training_examples": 0,
                "active_model": "Nenhum"
            },
            "settings": self.state["settings"]
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
    
    # CSS
    st.markdown("""
    <style>
    .main-board {
        background-color: #1a1a1a;
        border: 2px solid #444;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .board-header {
        font-size: 18px;
        font-weight: bold;
        color: white;
        margin-bottom: 15px;
        text-align: center;
    }
    .beads-grid {
        display: grid;
        grid-template-columns: repeat(10, 1fr);
        gap: 5px;
        margin: 10px 0;
    }
    .bead-column {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 3px;
    }
    .bead {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        border: 2px solid white;
        font-size: 14px;
    }
    .bead-azul { background-color: #2196f3; }
    .bead-vermelho { background-color: #f44336; }
    .bead-empate { background-color: #ffc107; color: black; }
    .column-number {
        font-size: 12px;
        color: #888;
        margin-top: 5px;
    }
    .current-column {
        border: 2px dashed #fff;
        padding: 5px;
        border-radius: 8px;
    }
    
    .prediction-box { 
        border: 3px solid; 
        border-radius: 10px; 
        padding: 15px; 
        text-align: center; 
        margin: 10px 0; 
        font-weight: bold; 
        font-size: 18px; 
    }
    .prediction-azul { border-color: #2196f3; background-color: rgba(33, 150, 243, 0.1); }
    .prediction-vermelho { border-color: #f44336; background-color: rgba(244, 67, 54, 0.1); }
    .prediction-empate { border-color: #ffc107; background-color: rgba(255, 193, 7, 0.1); }
    
    .control-panel {
        background-color: #2a2a2a;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎰 BB DEEP - Mobile")
    
    # Inicializar app UMA VEZ
    if 'app' not in st.session_state:
        st.session_state.app = BBDeepMobile()
    
    app = st.session_state.app
    
    # Layout principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # QUADRO PRINCIPAL DE BEADS (como na app original)
        st.markdown('<div class="main-board">', unsafe_allow_html=True)
        st.markdown('<div class="board-header">🎯 BEADS VERTICAIS</div>', unsafe_allow_html=True)
        
        # Criar grid de colunas
        all_columns = app.state["beads"] + [app.state["current_column"]] if app.state["current_column"] else app.state["beads"]
        
        if all_columns:
            # Mostrar máximo de 10 colunas
            display_columns = all_columns[-10:]
            
            st.markdown('<div class="beads-grid">', unsafe_allow_html=True)
            
            for i, column in enumerate(display_columns):
                is_current = (i == len(display_columns) - 1 and column == app.state["current_column"])
                column_class = "bead-column current-column" if is_current else "bead-column"
                
                st.markdown(f'<div class="{column_class}">', unsafe_allow_html=True)
                
                # Adicionar beads vazios para completar 6 posições
                for row in range(6):
                    if row < len(column):
                        bead = column[row]
                        color_class = f"bead-{bead['color']}"
                        if bead['color'] == 'empate':
                            display_text = str(bead.get('tie_sum', 'E'))
                        else:
                            display_text = bead['color'][0].upper()
                        st.markdown(f'<div class="bead {color_class}">{display_text}</div>', unsafe_allow_html=True)
                    else:
                        # Bead vazio
                        st.markdown('<div class="bead" style="background-color: #333; border: 2px dashed #666;"></div>', unsafe_allow_html=True)
                
                # Número da coluna
                col_number = len(app.state["beads"]) + (1 if app.state["current_column"] else 0) - len(display_columns) + i + 1
                status = " (Atual)" if is_current else ""
                st.markdown(f'<div class="column-number">Coluna {col_number}{status}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Estatísticas do quadro
            current_progress = len(app.state["current_column"])
            st.write(f"**Progresso da coluna atual:** {current_progress}/6 beads")
            st.write(f"**Total de colunas:** {len(app.state['beads'])}")
            
        else:
            st.info("Nenhum bead registado ainda. Use os botões abaixo para começar.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CONTROLES DE BEADS
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("🎯 Registar Beads")
        
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            if st.button("🔵 AZUL", use_container_width=True, key="btn_azul"):
                app.register_bead('azul')
        with col1b:
            if st.button("🔴 VERMELHO", use_container_width=True, key="btn_vermelho"):
                app.register_bead('vermelho')
        with col1c:
            tie_sum = st.selectbox("Soma EMPATE:", [2,3,4,5,6,7,8,9,10,11,12], key="tie_select")
            if st.button("🟡 EMPATE", use_container_width=True, key="btn_empate"):
                app.register_bead('empate', tie_sum)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # PREVISÃO
        st.subheader("🔮 Próxima Previsão")
        next_color, confidence = app.get_next_prediction()
        
        if next_color:
            color_name = {"azul": "AZUL", "vermelho": "VERMELHO", "empate": "EMPATE"}
            color_class = f"prediction-{next_color}"
            color_emoji = {"azul": "🔵", "vermelho": "🔴", "empate": "🟡"}
            
            st.markdown(f"""
            <div class="prediction-box {color_class}">
                <div style="font-size: 24px; margin-bottom: 10px;">{color_emoji[next_color]}</div>
                <div>{color_name[next_color]}</div>
                <div style="font-size: 14px; margin-top: 5px;">{confidence:.1f}% confiança</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🤖 Treine o modelo ML para obter previsões")
        
        # APOSTAS
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("💰 Fazer Aposta")
        
        bet_amount = st.number_input("Valor da aposta:", value=5.0, min_value=0.1, step=0.5, key="bet_amount")
        
        col2a, col2b, col2c = st.columns(3)
        with col2a:
            if st.button("Apostar 🔵", use_container_width=True, key="bet_azul"):
                if app.place_bet('azul', bet_amount):
                    st.success(f"✅ Aposta de {bet_amount} em AZUL!")
        with col2b:
            if st.button("Apostar 🔴", use_container_width=True, key="bet_vermelho"):
                if app.place_bet('vermelho', bet_amount):
                    st.success(f"✅ Aposta de {bet_amount} em VERMELHO!")
        with col2c:
            if st.button("Apostar 🟡", use_container_width=True, key="bet_empate"):
                if app.place_bet('empate', bet_amount):
                    st.success(f"✅ Aposta de {bet_amount} em EMPATE!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # APOSTAS ATIVAS
        st.subheader("📋 Apostas Ativas")
        if app.state["bets"]:
            for bet in app.state["bets"]:
                emoji = "🔵" if bet['color'] == 'azul' else "🔴" if bet['color'] == 'vermelho' else "🟡"
                st.write(f"{emoji} **{bet['color'].upper()}**: {bet['amount']:.1f}x")
        else:
            st.info("Nenhuma aposta ativa")
        
        # ESTATÍSTICAS RÁPIDAS
        st.subheader("📊 Estatísticas")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("💰 Banca", f"{app.state['bank']:.1f}x")
            st.metric("🔵 Azul", app.state['statistics']['azul_count'])
            st.metric("🔴 Vermelho", app.state['statistics']['vermelho_count'])
        with col_stat2:
            st.metric("📈 Lucro", f"{app.state['statistics']['profit']:+.1f}x")
            st.metric("🟡 Empate", app.state['statistics']['empate_count'])
            st.metric("🎯 Total", app.state['statistics']['total_beads'])

    # BOTÕES DE GESTÃO
    st.markdown("---")
    st.subheader("⚙️ Gestão")
    
    col_manage1, col_manage2, col_manage3, col_manage4 = st.columns(4)
    
    with col_manage1:
        if st.button("🎯 Treinar Modelo", use_container_width=True):
            if app.train_model():
                st.success("✅ Modelo treinado!")
    
    with col_manage2:
        if st.button("🔄 Reset Beads", use_container_width=True):
            app.reset_beads()
            st.success("✅ Beads resetados!")
    
    with col_manage3:
        if st.button("💰 Reset Apostas", use_container_width=True):
            app.reset_bets()
            st.success("✅ Apostas resetadas!")
    
    with col_manage4:
        if st.button("💥 Reset Tudo", use_container_width=True):
            app.reset_all()
            st.success("✅ Reset completo!")

if __name__ == "__main__":
    main()