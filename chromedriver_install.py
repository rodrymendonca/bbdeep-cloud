#!/usr/bin/env python3
"""
Script para instalar ChromeDriver automaticamente
"""

import os
import sys
from webdriver_manager.chrome import ChromeDriverManager

def install_chromedriver():
    print("🔧 Instalando ChromeDriver...")
    try:
        # Instalar ChromeDriver
        driver_path = ChromeDriverManager().install()
        print(f"✅ ChromeDriver instalado em: {driver_path}")
        return True
    except Exception as e:
        print(f"❌ Erro ao instalar ChromeDriver: {e}")
        return False

if __name__ == "__main__":
    install_chromedriver()