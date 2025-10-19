import json
import os
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
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bbdeep.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def ensure_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.logger.info(f"Pasta de dados criada: {self.data_dir}")
    
    def get_file_path(self, filename):
        return os.path.join(self.data_dir, filename)
    
    def save_data(self, data, filename):
        try:
            filepath = self.get_file_path(filename)
            
            self.logger.info(f"Tentando guardar {filename} - Tamanho beads: {len(data.get('beads', []))}, Total beads: {data.get('statistics', {}).get('total_beads', 0)}")
            
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
                
                self.logger.info(f"Dados carregados: {filename} - "
                               f"Beads: {len(data.get('beads', []))}, "
                               f"Total: {data.get('statistics', {}).get('total_beads', 0)}")
                return data
            else:
                self.logger.info(f"Ficheiro não encontrado: {filename}")
                return default if default is not None else {}
        except Exception as e:
            self.logger.error(f"Erro ao carregar {filename}: {e}")
            return default if default is not None else {}
    
    def export_data(self, data, export_path=None):
        try:
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"bbdeep_backup_{timestamp}.json"
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Backup criado: {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao criar backup: {e}")
            return False
    
    def import_data(self, import_path):
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Backup importado: {import_path} - "
                           f"Beads: {len(data.get('beads', []))}, "
                           f"Total: {data.get('statistics', {}).get('total_beads', 0)}")
            return data
        except Exception as e:
            self.logger.error(f"Erro ao importar backup: {e}")
            return None