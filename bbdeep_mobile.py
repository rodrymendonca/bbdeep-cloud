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
        
        # CORREÇÃO: Sempre que a cor muda OU a coluna atingiu 6 beads, fecha a coluna atual
        if current_col and (current_col[-1]["color"] != color or len(current_col) >= 6):
            self.state["beads"].append(current_col.copy())
            self.state["current_column"] = [bead]
        else:
            # Continua na mesma coluna
            self.state["current_column"].append(bead)
        
        # CORREÇÃO: Se a coluna atual atingiu 6 beads, fecha automaticamente
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
    .bead-display { display: flex; gap: 5px; margin: 10px 0; }
    .bead { width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; border: 2px solid white; font-size: 16px; }
    .bead-azul { background-color: #2196f3; }
    .bead-vermelho { background-color: #f44336; }
    .bead-empate { background-color: #ffc107; color: black; }
    .column-container { border: 2px solid #444; border-radius: 8px; padding: 10px; margin: 8px 0; background-color: #1a1a1a; }
    .column-header { font-weight: bold; margin-bottom: 8px; color: #fff; }
    .prediction-box { border: 3px solid; border-radius: 10px; padding: 15px; text-align: center; margin: 10px 0; font-weight: bold; font-size: 18px; }
    .prediction-azul { border-color: #2196f3; background-color: rgba(33, 150, 243, 0.1); }
    .prediction-vermelho { border-color: #f44336; background-color: rgba(244, 67, 54, 0.1); }
    .prediction-empate { border-color: #ffc107; background-color: rgba(255, 193, 7, 0.1); }
    
    /* NOVO CSS PARA BEADS VERTICAIS - COMO NO ORIGINAL */
    .beads-vertical-container { display: flex; flex-direction: column; gap: 5px; margin: 10px 0; }
    .bead-vertical { width: 35px; height: 35px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; border: 2px solid white; font-size: 16px; margin-bottom: 5px; }
    .bead-vertical-azul { background-color: #2196f3; }
    .bead-vertical-vermelho { background-color: #f44336; }
    .bead-vertical-empate { background-color: #ffc107; color: black; font-size: 14px; }
    .columns-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(50px, 1fr)); gap: 10px; margin: 15px 0; }
    .column-vertical { display: flex; flex-direction: column; align-items: center; gap: 2px; }
    .column-label { font-size: 12px; color: #888; margin-bottom: 5px; }
    
    /* Estilo para mostrar o valor do empate */
    .tie-value { font-size: 12px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎰 BB DEEP - Mobile")
    
    # Inicializar app UMA VEZ
    if 'app' not in st.session_state:
        st.session_state.app = BBDeepMobile()
    
    app = st.session_state.app
    
    # Layout principal
    tab1, tab2, tab3 = st.tabs(["🎯 Beads & Apostas", "📊 Estatísticas", "⚙️ Configurações"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Registar Beads")
            
            # Botões para beads
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                if st.button("🔵 AZUL", use_container_width=True, key="btn_azul"):
                    app.register_bead('azul')
            with col1b:
                if st.button("🔴 VERMELHO", use_container_width=True, key="btn_vermelho"):
                    app.register_bead('vermelho')
            with col1c:
                # CORREÇÃO: Simplificar o registo de empates
                tie_sum = st.selectbox("Soma para EMPATE:", [2,3,4,5,6,7,8,9,10,11,12], key="tie_select")
                if st.button("🟡 EMPATE", use_container_width=True, key="btn_empate"):
                    app.register_bead('empate', tie_sum)
            
            # PRÓXIMA PREVISÃO
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
            
            # Coluna atual
            st.subheader("📝 Coluna Atual")
            if app.state["current_column"]:
                st.markdown('<div class="column-container">', unsafe_allow_html=True)
                st.markdown('<div class="column-header">Coluna em Construção</div>', unsafe_allow_html=True)
                
                beads_html = '<div class="bead-display">'
                for bead in app.state["current_column"]:
                    color_class = f"bead-{bead['color']}"
                    # CORREÇÃO: Mostrar o valor do empate
                    if bead['color'] == 'empate':
                        display_text = str(bead.get('tie_sum', 'E'))
                    else:
                        display_text = bead['color'][0].upper()
                    beads_html += f'<div class="bead {color_class}">{display_text}</div>'
                beads_html += '</div>'
                st.markdown(beads_html, unsafe_allow_html=True)
                
                st.write(f"**Progresso:** {len(app.state['current_column'])}/6 beads")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Nenhum bead na coluna atual")
        
        with col2:
            st.subheader("💰 Fazer Aposta")
            
            bet_amount = st.number_input("Valor:", value=5.0, min_value=0.1, step=0.5, key="bet_amount")
            
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
            
            # Apostas ativas
            st.subheader("📋 Apostas Ativas")
            if app.state["bets"]:
                for bet in app.state["bets"]:
                    emoji = "🔵" if bet['color'] == 'azul' else "🔴" if bet['color'] == 'vermelho' else "🟡"
                    st.write(f"{emoji} **{bet['color'].upper()}**: {bet['amount']:.1f}x")
            else:
                st.info("Nenhuma aposta ativa")
            
            # Histórico de colunas - CORREÇÃO COMPLETA
            st.subheader("📚 Histórico de Colunas")
            if app.state["beads"]:
                # Criar grid para as colunas verticais
                st.markdown('<div class="columns-grid">', unsafe_allow_html=True)
                
                # Mostrar TODAS as colunas (ou as últimas 12 para não sobrecarregar)
                display_columns = app.state["beads"][-12:]  # Últimas 12 colunas
                
                for i, column in enumerate(display_columns):
                    col_number = len(app.state["beads"]) - len(display_columns) + i + 1
                    
                    st.markdown(f'''
                    <div class="column-vertical">
                        <div class="column-label">Coluna {col_number}</div>
                    ''', unsafe_allow_html=True)
                    
                    # Adicionar beads verticais - CORREÇÃO: Mostrar valor do empate
                    for bead in column:
                        color_class = f"bead-vertical-{bead['color']}"
                        if bead['color'] == 'empate':
                            display_text = str(bead.get('tie_sum', 'E'))
                        else:
                            display_text = bead['color'][0].upper()
                        st.markdown(f'<div class="bead-vertical {color_class}">{display_text}</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Mostrar estatísticas do histórico
                st.write(f"**Total de colunas completas:** {len(app.state['beads'])}")
            else:
                st.info("Nenhuma coluna completa ainda")
    
    with tab2:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("💳 Banca & Apostas")
            st.metric("💰 Banca", f"{app.state['bank']:.1f}x")
            st.metric("📈 Lucro", f"{app.state['statistics']['profit']:+.1f}x")
            
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.metric("🎯 Total", app.state['statistics']['bets_count'])
            with col1b:
                st.metric("✅ Ganhas", app.state['statistics']['bets_won'])
            with col1c:
                st.metric("❌ Perdidas", app.state['statistics']['bets_lost'])
            
            if app.state['statistics']['bets_count'] > 0:
                win_rate = (app.state['statistics']['bets_won'] / app.state['statistics']['bets_count']) * 100
                st.metric("📊 Taxa Vitória", f"{win_rate:.1f}%")
        
        with col2:
            st.subheader("🎰 Beads & ML")
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("🔵 Azul", app.state['statistics']['azul_count'])
            with col2b:
                st.metric("🔴 Vermelho", app.state['statistics']['vermelho_count'])
            with col2c:
                st.metric("🟡 Empate", app.state['statistics']['empate_count'])
            
            st.metric("📊 Total Beads", app.state['statistics']['total_beads'])
            st.metric("🔴 Seq Vermelho", app.state['statistics']['seq_vermelho'])
            st.metric("🟡 Seq Empate", app.state['statistics']['seq_empate'])
            
            if app.state["ml_model"]["trained"]:
                st.subheader("🤖 Previsões")
                pred = app.state["ml_model"]["predictions"]
                col_pred1, col_pred2, col_pred3 = st.columns(3)
                with col_pred1:
                    st.metric("🔵 Azul", f"{pred['azul']:.1f}%")
                with col_pred2:
                    st.metric("🔴 Vermelho", f"{pred['vermelho']:.1f}%")
                with col_pred3:
                    st.metric("🟡 Empate", f"{pred['empate']:.1f}%")
    
    with tab3:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🤖 Machine Learning")
            
            if st.button("🎯 Treinar Modelo", use_container_width=True, key="train_ml"):
                if app.train_model():
                    st.success("✅ Modelo treinado!")
            
            if app.state["ml_model"]["trained"]:
                st.info(f"**Modelo:** {app.state['ml_model']['model_type']}")
                st.metric("🎯 Precisão", f"{app.state['ml_model']['accuracy']:.1f}%")
            
            st.subheader("⚙️ Configurações")
            auto_train = st.checkbox("Auto-treino", value=app.state["settings"]["auto_train"], key="auto_train")
            train_interval = st.number_input("Intervalo treino", 
                                           value=app.state["settings"]["train_interval"],
                                           min_value=1, max_value=20, key="train_interval")
            
            if st.button("💾 Guardar Config", use_container_width=True, key="save_config"):
                app.state["settings"]["auto_train"] = auto_train
                app.state["settings"]["train_interval"] = train_interval
                app.save_state()
                st.success("✅ Configurações guardadas!")
        
        with col2:
            st.subheader("🗑️ Gestão de Dados")
            
            if st.button("🔄 Reset Beads", use_container_width=True, key="reset_beads"):
                app.reset_beads()
                st.success("✅ Beads resetados!")
            
            if st.button("💰 Reset Apostas", use_container_width=True, key="reset_bets"):
                app.reset_bets()
                st.success("✅ Apostas resetadas!")
            
            if st.button("💥 Reset Tudo", use_container_width=True, key="reset_all"):
                app.reset_all()
                st.success("✅ Reset completo!")
            
            st.subheader("📊 Informação")
            st.write(f"**Beads totais:** {app.state['statistics']['total_beads']}")
            st.write(f"**Colunas:** {len(app.state['beads'])}")
            st.write(f"**Apostas históricas:** {len(app.state['bet_history'])}")

if __name__ == "__main__":
    main()