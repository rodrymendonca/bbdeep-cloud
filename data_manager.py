import json
import os
import streamlit as st
from datetime import datetime
import logging

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
            
            self.logger.info(f"Dados guardados: {filename} - Beads: {len(data.get('beads', []))}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao guardar {filename}: {e}")
            # Tentar usar backup se existir
            try:
                backup_path = self.get_file_path(filename) + ".backup"
                if os.path.exists(backup_path):
                    with open(backup_path, 'r', encoding='utf-8') as backup:
                        backup_data = json.load(backup)
                    with open(self.get_file_path(filename), 'w', encoding='utf-8') as f:
                        json.dump(backup_data, f, indent=2, ensure_ascii=False)
                    self.logger.info("Restaurado a partir do backup")
            except:
                pass
            return False
    
    def load_data(self, filename, default=None):
        try:
            filepath = self.get_file_path(filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.logger.info(f"Dados carregados: {filename} - Beads: {len(data.get('beads', []))}")
                return data
            else:
                self.logger.info(f"Ficheiro não encontrado: {filename}")
                return default if default is not None else {}
        except Exception as e:
            self.logger.error(f"Erro ao carregar {filename}: {e}")
            # Tentar carregar backup
            try:
                backup_path = self.get_file_path(filename) + ".backup"
                if os.path.exists(backup_path):
                    with open(backup_path, 'r', encoding='utf-8') as backup:
                        data = json.load(backup)
                    self.logger.info("Carregado a partir do backup")
                    return data
            except:
                pass
            return default if default is not None else {}
    
    def export_data(self, data, export_path=None):
        try:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"bbdeep_export_{timestamp}.json"
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Export criado: {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao criar export: {e}")
            return False
    
    def import_data(self, import_path):
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Dados importados: {import_path}")
            return data
        except Exception as e:
            self.logger.error(f"Erro ao importar dados: {e}")
            return None

    def get_all_saves(self):
        """Lista todos os ficheiros de save disponíveis"""
        try:
            files = os.listdir(self.data_dir)
            saves = [f for f in files if f.endswith('.json') and 'app_state' in f]
            return saves
        except:
            return []