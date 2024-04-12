import tkinter
import customtkinter
import numpy as np
#import joblib
import pandas as pd
#from CTkMessagebox import CTkMessagebox
import datetime
import locale
#from sklearn.metrics import mean_absolute_percentage_error
#import pickle
import sys
sys.path.append('./')
import Yamakawa_Adaptativo as YA
import save_database as sd
#import openpyxl
from PIL import Image
from mpmath import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings("ignore")
import glob



customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class TTemp:
    def setXit(self, xit):
        self.xit = xit

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Sistema de Previsão de Vazão de Gás Natural")
        self.iconbitmap('Imagens/logo.ico')
        self.geometry(f"{1050}x{650}")
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
        # Variáveis
        self.yhat = [0]
        self.inputs = {}
        self.inputs11 = {}
        self.inputs24 = {}
        self.var = tkinter.StringVar()
        self.database_24_path = 'BD/dataset_24.db'
        self.database_7_path = 'BD/database_7.db'
        self.database_residuos_path = 'BD/residuos.db'
        self.hoje = datetime.date.today()
        self.amanha = datetime.date.today() + datetime.timedelta(days=1)
        self.ontem = datetime.date.today() - datetime.timedelta(days=1)
        self.anteontem = datetime.date.today() - datetime.timedelta(days=2)
        self.model = YA.Adaptativo()
        self.l = 10
        self.forecast = []

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Previsão",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.open_24h_window, text="24 horas")
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.open_7dias_window, text="7 dias", state="normal")
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        
        # create tabviews
        self.tabview_7 = customtkinter.CTkTabview(self, width=830, height=550)
        self.tabview_24 = customtkinter.CTkTabview(self, width=830, height=550)
        
        # CAIXA DE MENSAGEM
        self.textbox_msg = customtkinter.CTkTextbox(self, width=830, height=9)
        self.textbox_msg.place(x=190, y=611)
        
        # Altera cor da interface
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        
        
        #Cria tabelas no banco de dados
        sd.execute("CREATE TABLE IF NOT EXISTS relatorio24(ponto TEXT, data DATE, vazao24_real FLOAT, \
                    vazao11_real FLOAT, vazao24_estimado FLOAT)", self.database_24_path)
            
        sd.execute("CREATE TABLE IF NOT EXISTS relatorio7_pontual(ponto TEXT, data DATE, vazao24_1 FLOAT, \
                    vazao24_2 FLOAT, vazao24_3 FLOAT, vazao24_4 FLOAT, vazao24_5 FLOAT, vazao24_6 FLOAT, \
                    vazao24_7 FLOAT)", self.database_7_path)
            
        sd.execute("CREATE TABLE IF NOT EXISTS relatorio7_intervalsup(ponto TEXT, data DATE, vazao24_1 FLOAT, \
                    vazao24_2 FLOAT, vazao24_3 FLOAT, vazao24_4 FLOAT, vazao24_5 FLOAT, vazao24_6 FLOAT, \
                    vazao24_7 FLOAT)", self.database_7_path)
            
        sd.execute("CREATE TABLE IF NOT EXISTS relatorio7_intervalinf(ponto TEXT, data DATE, vazao24_1 FLOAT, \
                    vazao24_2 FLOAT, vazao24_3 FLOAT, vazao24_4 FLOAT, vazao24_5 FLOAT, vazao24_6 FLOAT, \
                    vazao24_7 FLOAT)", self.database_7_path)
        
###################################################################ABA 24 HORAS     
    def open_24h_window(self):

        self.tabview_7.destroy()
        
        self.textbox_msg.delete("0.0","end")
        
                
        # ABA DE PREVISÃO 24
        self.tabview_24 = customtkinter.CTkTabview(self, width=830, height=550)
        self.tabview_24.grid(row=0, column=1, padx=(10, 0), pady=(10, 0))
        text_tab = "Previsão da vazão acumulada diária"
        self.tabview_24.add(text_tab)
        
              
        ####################################################### ABA DE HORAS
        #################### Data anterior
        self.label_data_anterior = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Dia anterior",
                                                  anchor="w", font=customtkinter.CTkFont(size=12))
        self.label_data_anterior.grid(row=0, column=2, padx=(0, 0), pady=(0, 0))
        
        self.label_ontem = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text= self.ontem.strftime('%d/%m/%Y'),
                                                 anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_ontem.grid(row=1, column=2, padx=(0, 0), pady=(0, 0))
        
        
        ################### Data atual
        self.label_data_atual = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Dia atual",
                                                  anchor="w", font=customtkinter.CTkFont(size=12))
        self.label_data_atual.grid(row=0, column=4, padx=(0, 0), pady=(0, 0))
        
        
        self.label_hoje = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text= self.hoje.strftime('%d/%m/%Y'),
                                                 anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_hoje.grid(row=1, column=4, padx=(0, 0), pady=(0, 0))
        
        
        ######################################################### ABA BARBACENA
        self.label_barbacena0 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Barbacena",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_barbacena0.grid(row=3, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada 24 do dia anterior
        self.label_barbacena1 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     24h:",
                                                       anchor="w")
        self.label_barbacena1.grid(row=3, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")

        self.entry_barbacena1 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_barbacena1.grid(row=3, column=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada até 11h e faz a previsão
        self.label_barbacena2 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     11h:",
                                                       anchor="w")
        self.label_barbacena2.grid(row=3, column=3, padx=(5, 5), pady=(5, 5), sticky="nsew")

        self.entry_barbacena2 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_barbacena2.grid(row=3, column=4, padx=(5, 5), pady=(5, 5), sticky="nsew")

        ###apresenta o yhat e salva o csv "estimado"
        self.label_barbacena3 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Previsão 24h:",
                                                       anchor="w")
        self.label_barbacena3.grid(row=3, column=5, padx=(5, 5), pady=(5, 5), sticky="nsew")

        self.label_barbacena4 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="                     ",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_barbacena4.grid(row=3, column=6, padx=(5, 5), pady=(5, 5), sticky="nsew")


        ############################################################# ABA BETIM
        self.label_betim0 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Betim",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_betim0.grid(row=4, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada 24 do dia anterior
        self.label_betim1 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     24h:",
                                                       anchor="w")
        self.label_betim1.grid(row=4, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_betim1 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_betim1.grid(row=4, column=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada até 11h e faz a previsão
        self.label_betim2 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     11h:",
                                                       anchor="w")
        self.label_betim2.grid(row=4, column=3, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_betim2 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_betim2.grid(row=4, column=4, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###apresenta o yhat e salva o csv "estimado"
        self.label_betim3 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Previsão 24h:",
                                                       anchor="w")
        self.label_betim3.grid(row=4, column=5, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.label_betim4 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="                     ",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_betim4.grid(row=4, column=6, padx=(5, 5), pady=(5, 5), sticky="nsew")


        ######################################################## ABA BRUMADINHO
        self.label_brumadinho0 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Brumadinho",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_brumadinho0.grid(row=5, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada 24 do dia anterior
        self.label_brumadinho1 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     24h:",
                                                       anchor="w")
        self.label_brumadinho1.grid(row=5, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_brumadinho1 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_brumadinho1.grid(row=5, column=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada até 11h e faz a previsão
        self.label_brumadinho2 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     11h:",
                                                       anchor="w")
        self.label_brumadinho2.grid(row=5, column=3, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_brumadinho2 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_brumadinho2.grid(row=5, column=4, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###apresenta o yhat e salva o csv "estimado"
        self.label_brumadinho3 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Previsão 24h:",
                                                       anchor="w")
        self.label_brumadinho3.grid(row=5, column=5, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.label_brumadinho4 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="                     ",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_brumadinho4.grid(row=5, column=6, padx=(5, 5), pady=(5, 5), sticky="nsew")


        ######################################################### ABA JACUTINGA
        self.label_jacutinga0 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Jacutinga",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_jacutinga0.grid(row=6, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada 24 do dia anterior
        self.label_jacutinga1 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     24h:",
                                                       anchor="w")
        self.label_jacutinga1.grid(row=6, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_jacutinga1 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_jacutinga1.grid(row=6, column=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada até 11h e faz a previsão
        self.label_jacutinga2 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     11h:",
                                                       anchor="w")
        self.label_jacutinga2.grid(row=6, column=3, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_jacutinga2 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_jacutinga2.grid(row=6, column=4, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###apresenta o yhat e salva o csv "estimado"
        self.label_jacutinga3 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Previsão 24h:",
                                                       anchor="w")
        self.label_jacutinga3.grid(row=6, column=5, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.label_jacutinga4 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="                     ",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_jacutinga4.grid(row=6, column=6, padx=(5, 5), pady=(5, 5), sticky="nsew")


        ###################################################### ABA JUIZ DE FORA
        self.label_juizdefora0 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Juiz de Fora",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_juizdefora0.grid(row=7, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada 24 do dia anterior
        self.label_juizdefora1 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     24h:",
                                                       anchor="w")
        self.label_juizdefora1.grid(row=7, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_juizdefora1 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_juizdefora1.grid(row=7, column=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada até 11h e faz a previsão
        self.label_juizdefora2 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     11h:",
                                                       anchor="w")
        self.label_juizdefora2.grid(row=7, column=3, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_juizdefora2 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_juizdefora2.grid(row=7, column=4, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###apresenta o yhat e salva o csv "estimado"
        self.label_juizdefora3 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Previsão 24h:",
                                                       anchor="w")
        self.label_juizdefora3.grid(row=7, column=5, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.label_juizdefora4 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="                     ",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_juizdefora4.grid(row=7, column=6, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        

        ########################################################## ABA SAO BRAS
        self.label_saobras0 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="São Brás",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_saobras0.grid(row=8, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada 24 do dia anterior
        self.label_saobras1 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     24h:",
                                                       anchor="w")
        self.label_saobras1.grid(row=8, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_saobras1 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_saobras1.grid(row=8, column=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###insere vazao acumulada até 11h e faz a previsão
        self.label_saobras2 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="     11h:",
                                                       anchor="w")
        self.label_saobras2.grid(row=8, column=3, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.entry_saobras2 = customtkinter.CTkEntry(self.tabview_24.tab(text_tab), placeholder_text="Exemplo: 2050.32", width=150)
        self.entry_saobras2.grid(row=8, column=4, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        ###apresenta o yhat e salva o csv "estimado"
        self.label_saobras3 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="Previsão 24h:",
                                                       anchor="w")
        self.label_saobras3.grid(row=8, column=5, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        self.label_saobras4 = customtkinter.CTkLabel(self.tabview_24.tab(text_tab), text="                     ",
                                                       anchor="w", font=customtkinter.CTkFont(size=12, weight="bold"))
        self.label_saobras4.grid(row=8, column=6, padx=(5, 5), pady=(5, 5), sticky="nsew")
        
        

        ##############################################################   BOTÕES
       
        button_image = customtkinter.CTkImage(Image.open("Imagens/edit.png"), size=(20, 20))
        self.button_preencher = customtkinter.CTkButton(self.tabview_24.tab(text_tab),
                                                       border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                       text="Preencher",
                                                       image = button_image,
                                                       hover_color = "#55e2e9",
                                                       command=lambda: self.get_preencher())
        self.button_preencher.place(x=130, y=300)
        
        
        # button_image = customtkinter.CTkImage(Image.open("Imagens/prever.png"), size=(20, 20))
        # self.button_prever = customtkinter.CTkButton(self.tabview_24.tab(text_tab),
        #                                                border_width=0, text_color=("gray10", "#DCE4EE"), width=130,
        #                                                text="Prever",
        #                                                image = button_image,
        #                                                state="normal",
        #                                                hover_color = "#55e2e9",
        #                                                command=lambda: self.predict(text_tab))
        # self.button_prever.place(x=200, y=300)
        # #self.button_prever.grid(row=8, column=5, padx=(5, 5), pady=(5, 5))
        
        
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/save.png"), size=(20, 20))
        self.button_salvar = customtkinter.CTkButton(self.tabview_24.tab(text_tab),
                                                        border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                        text="Prever e Salvar",
                                                        image = button_image,
                                                        state="normal",
                                                        hover_color = "#55e2e9",
                                                        command=lambda: self.predict_24(text_tab))
        self.button_salvar.place(x=290, y=300)

        
        button_image = customtkinter.CTkImage(Image.open("Imagens/download.png"), size=(20, 20))
        self.button_download = customtkinter.CTkButton(self.tabview_24.tab(text_tab),
                                                       border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                       text = "Download",
                                                       image = button_image,
                                                       state="normal",
                                                       hover_color = "#55e2e9",
                                                       command=lambda: self.download_24())
        self.button_download.place(x=450, y=300)

        
        


###################################################################ABA 7 DIAS
    def open_7dias_window(self):
        
        # Removing all labels inside frame
        self.tabview_24.destroy()
        
        self.textbox_msg.delete("0.0","end")
        
        
        # create tabview
        self.tabview_7 = customtkinter.CTkTabview(self, width=830, height=550)
        self.tabview_7.grid(row=0, column=1, padx=(10, 0), pady=(10, 0))
        self.tabview_7.add("Barbacena")
        self.tabview_7.add("Betim")
        self.tabview_7.add("Brumadinho")
        self.tabview_7.add("Jacutinga")
        self.tabview_7.add("Juiz de Fora")
        self.tabview_7.add("São Brás")
        self.tabview_7.tab("Barbacena").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview_7.tab("Betim").grid_columnconfigure(0, weight=1)
        self.tabview_7.tab("Brumadinho").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview_7.tab("Jacutinga").grid_columnconfigure(0, weight=1)
        self.tabview_7.tab("Juiz de Fora").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview_7.tab("São Brás").grid_columnconfigure(0, weight=1)
        
        ###PUXA DATA DO DIA ATUAL E O SÉTIMO DIA
        setimo_dia = self.hoje + datetime.timedelta(days=7)
        

        # ABA BARBACENA
        
        ### ÁREA COM BOTÃO DE PEVISÃO
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/save.png"), size=(20, 20))
        self.button_prever_barbacena = customtkinter.CTkButton(self.tabview_7.tab("Barbacena"),
                                                       border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                       text="Prever e Salvar",
                                                       image = button_image,
                                                       state="normal",
                                                       hover_color = "#55e2e9",
                                                       command=lambda: self.predict_7('Barbacena')
                                                       )
        self.button_prever_barbacena.place(x=10, y=20)
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/download.png"), size=(20, 20))
        self.button_download_barbacena = customtkinter.CTkButton(self.tabview_7.tab("Barbacena"),
                                                        border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                        text = "Download",
                                                        image = button_image,
                                                        state="normal",
                                                        hover_color = "#55e2e9",
                                                        command=lambda: self.download_7("Barbacena"))
        self.button_download_barbacena.place(x=180, y=20)
        

        # ABA BETIM
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/save.png"), size=(20, 20))
        self.button_prever_betim = customtkinter.CTkButton(self.tabview_7.tab("Betim"),
                                                       border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                       text="Prever e Salvar",
                                                       image = button_image,
                                                       state="normal",
                                                       hover_color = "#55e2e9",
                                                       command=lambda: self.predict_7('Betim')
                                                       )
        self.button_prever_betim.place(x=10, y=20)
        
        
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/download.png"), size=(20, 20))
        self.button_download_betim = customtkinter.CTkButton(self.tabview_7.tab("Betim"),
                                                        border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                        text = "Download",
                                                        image = button_image,
                                                        state="normal",
                                                        hover_color = "#55e2e9",
                                                        command=lambda: self.download_7("Betim"))
        self.button_download_betim.place(x=180, y=20)
        

       

        # ABA BRUMADINHO
        
        ### ÁREA COM BOTÃO DE PREVISÃO
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/save.png"), size=(20, 20))
        self.button_prever_brumadinho = customtkinter.CTkButton(self.tabview_7.tab("Brumadinho"),
                                                       border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                       text="Prever e Salvar",
                                                       image = button_image,
                                                       state="normal",
                                                       hover_color = "#55e2e9",
                                                       command=lambda: self.predict_7('Brumadinho')
                                                       )
        self.button_prever_brumadinho.place(x=10, y=20)
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/download.png"), size=(20, 20))
        self.button_download_brumadinho = customtkinter.CTkButton(self.tabview_7.tab("Brumadinho"),
                                                        border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                        text = "Download",
                                                        image = button_image,
                                                        state="normal",
                                                        hover_color = "#55e2e9",
                                                        command=lambda: self.download_7("Brumadinho"))
        self.button_download_brumadinho.place(x=180, y=20)
        

        # ABA JACUTINGA
        
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/save.png"), size=(20, 20))
        self.button_prever_jacutinga = customtkinter.CTkButton(self.tabview_7.tab("Jacutinga"),
                                                       border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                       text="Prever e Salvar",
                                                       image = button_image,
                                                       state="normal",
                                                       hover_color = "#55e2e9",
                                                       command=lambda: self.predict_7('Jacutinga')
                                                       )
        self.button_prever_jacutinga.place(x=10, y=20)
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/download.png"), size=(20, 20))
        self.button_download_jacutinga = customtkinter.CTkButton(self.tabview_7.tab("Jacutinga"),
                                                        border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                        text = "Download",
                                                        image = button_image,
                                                        state="normal",
                                                        hover_color = "#55e2e9",
                                                        command=lambda: self.download_7("Jacutinga"))
        self.button_download_jacutinga.place(x=180, y=20)
        

        # ABA JUIZ DE FORA
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/save.png"), size=(20, 20))
        self.button_prever_juizdefora = customtkinter.CTkButton(self.tabview_7.tab("Juiz de Fora"),
                                                       border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                       text="Prever e Salvar",
                                                       image = button_image,
                                                       state="normal",
                                                       hover_color = "#55e2e9",
                                                       command=lambda: self.predict_7('JuizdeFora')
                                                       )
        self.button_prever_juizdefora.place(x=10, y=20)
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/download.png"), size=(20, 20))
        self.button_download_juizdefora = customtkinter.CTkButton(self.tabview_7.tab("Juiz de Fora"),
                                                        border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                        text = "Download",
                                                        image = button_image,
                                                        state="normal",
                                                        hover_color = "#55e2e9",
                                                        command=lambda: self.download_7("JuizdeFora"))
        self.button_download_juizdefora.place(x=180, y=20)
        
        # ABA SAO BRAS
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/save.png"), size=(20, 20))
        self.button_prever_saobras = customtkinter.CTkButton(self.tabview_7.tab("São Brás"),
                                                       border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                       text="Prever e Salvar",
                                                       image = button_image,
                                                       state="normal",
                                                       hover_color = "#55e2e9",
                                                       command=lambda: self.predict_7('SaoBras')
                                                       )
        self.button_prever_saobras.place(x=10, y=20)
        
        button_image = customtkinter.CTkImage(Image.open("Imagens/download.png"), size=(20, 20))
        self.button_download_saobras = customtkinter.CTkButton(self.tabview_7.tab("São Brás"),
                                                        border_width=0, text_color=("gray10", "#DCE4EE"), width=150,
                                                        text = "Download",
                                                        image = button_image,
                                                        state="normal",
                                                        hover_color = "#55e2e9",
                                                        command=lambda: self.download_7("SaoBras"))
        self.button_download_saobras.place(x=180, y=20)
        
        
        

############################################################################################################################### 

        
    def get_preencher(self):
        
        self.clear_entry()
        self.textbox_msg.delete("0.0","end")
        
        # wb = openpyxl.load_workbook('Planilhas/entrada.xlsx')
        # sheet = wb.active
        
        # tag = 'B'
        # self.entry_barbacena1.insert(0,sheet[tag+'2'].value)
        # self.entry_betim1.insert(0,sheet[tag+'3'].value)
        # self.entry_brumadinho1.insert(0,sheet[tag+'4'].value)
        # self.entry_jacutinga1.insert(0,sheet[tag+'5'].value)
        # self.entry_juizdefora1.insert(0,sheet[tag+'6'].value)
        # self.entry_saobras1.insert(0,sheet[tag+'7'].value)
        # tag = 'C'
        # self.entry_barbacena2.insert(0,sheet[tag+'2'].value)
        # self.entry_betim2.insert(0,sheet[tag+'3'].value)
        # self.entry_brumadinho2.insert(0,sheet[tag+'4'].value)
        # self.entry_jacutinga2.insert(0,sheet[tag+'5'].value)
        # self.entry_juizdefora2.insert(0,sheet[tag+'6'].value)
        # self.entry_saobras2.insert(0,sheet[tag+'7'].value)
        
        
        
        file11 = glob.glob('Planilhas/*11h.csv')[0]
        file24 = glob.glob('Planilhas/*24h.csv')[0]
        
        df11 = pd.read_csv(file11, names = ['data', 'tag', 'valor'], sep=';')
        
        df11['valor'] = df11['valor'].str.replace(',','.')
        df11['valor'] = df11['valor'].astype(float)
        
        for row in df11.index.values:
            if df11['tag'][row] == 'FQID_01_0127':
                idx_betim = row
            elif df11['tag'][row] == 'FQID_02_0002':
                idx_juizdefora = row
            elif df11['tag'][row] == 'FQID_03_0010':
                idx_saobras = row
            elif df11['tag'][row] == 'FQID_04_0006':
                idx_barbacena = row
            elif df11['tag'][row] == 'FQID_06_0006':
                idx_jacutinga = row
            elif df11['tag'][row] == 'FQID_07_0003':
                idx_brumadinho = row
        
        self.entry_barbacena1.insert(0,df11['valor'][idx_barbacena])
        self.entry_betim1.insert(0,df11['valor'][idx_betim])
        self.entry_brumadinho1.insert(0,df11['valor'][idx_brumadinho])
        self.entry_jacutinga1.insert(0,df11['valor'][idx_jacutinga])
        self.entry_juizdefora1.insert(0,df11['valor'][idx_juizdefora])
        self.entry_saobras1.insert(0,df11['valor'][idx_saobras])
        
        df24 = pd.read_csv(file24, names = ['data', 'tag', 'valor'], sep=';')
        
        df24['valor'] = df24['valor'].str.replace(',','.')
        df24['valor'] = df24['valor'].astype(float)
        
        for row in df24.index.values:
            if df24['tag'][row] == 'FQID_01_0127':
                idx_betim = row
            elif df24['tag'][row] == 'FQID_02_0002':
                idx_juizdefora = row
            elif df24['tag'][row] == 'FQID_03_0010':
                idx_saobras = row
            elif df24['tag'][row] == 'FQID_04_0006':
                idx_barbacena = row
            elif df24['tag'][row] == 'FQID_06_0006':
                idx_jacutinga = row
            elif df24['tag'][row] == 'FQID_07_0003':
                idx_brumadinho = row
        
        self.entry_barbacena2.insert(0,df24['valor'][idx_barbacena])
        self.entry_betim2.insert(0,df24['valor'][idx_betim])
        self.entry_brumadinho2.insert(0,df24['valor'][idx_brumadinho])
        self.entry_jacutinga2.insert(0,df24['valor'][idx_jacutinga])
        self.entry_juizdefora2.insert(0,df24['valor'][idx_juizdefora])
        self.entry_saobras2.insert(0,df24['valor'][idx_saobras])
        
            
    def download_24(self):
        
        self.textbox_msg.delete("0.0","end")
    
        for ponto in ['Barbacena', 'Betim', 'Brumadinho', 'Jacutinga', 'JuizdeFora', 'SaoBras']:
            df = pd.DataFrame(sd.executep("SELECT * FROM relatorio24 WHERE ponto = ?", (ponto,), self.database_24_path))
            df.columns = ['identificador', 'data', 'Fech_24_real', 'Acc_11_real', 'Fech_24_estimado']
            
            if ponto == 'Barbacena':
                identificador = '04_0006'
            elif ponto == 'Betim':
                identificador = '01_0127'
            elif ponto == 'Brumadinho':
                identificador = '07_0003'
            elif ponto == 'Jacutinga':
                identificador = '06_0006'
            elif ponto == 'JuizdeFora':
                identificador = '02_0002'
            elif ponto == 'SaoBras':
                identificador = '03_0010'
            
            
            data = self.hoje.strftime("%Y/%m/%d")
            data = data.replace('/','_')
                
            df.to_csv('Downloads_24horas/'+data+'_'+identificador+'.csv', index=True, sep=';')
            
        self.textbox_msg.insert("0.0", "*A base de dados foi salva na pasta Downloads.")
        self.textbox_msg.configure(text_color='green')
    
    
              
                
    def predict_24(self, text_tab):
        
        self.textbox_msg.delete("0.0","end")
        

        check_data = sd.executep("SELECT * FROM relatorio24 WHERE data = ?", (self.hoje,), self.database_24_path)  

            
        if check_data:
            self.textbox_msg.delete("0.0","end")
            self.textbox_msg.insert("0.0", "*Já existem registros salvos para esta data.")
            self.textbox_msg.configure(text_color='red')
            
        else:
        
            self.inputs24 = {'Barbacena': self.entry_barbacena1.get(),
                      'Betim': self.entry_betim1.get(),
                      'Brumadinho': self.entry_brumadinho1.get(),
                      'Jacutinga': self.entry_jacutinga1.get(),
                      'JuizdeFora': self.entry_juizdefora1.get(),
                      'SaoBras': self.entry_saobras1.get()
                      }
            self.inputs11 = {'Barbacena': self.entry_barbacena2.get(),
                      'Betim': self.entry_betim2.get(),
                      'Brumadinho': self.entry_brumadinho2.get(),
                      'Jacutinga': self.entry_jacutinga2.get(),
                      'JuizdeFora': self.entry_juizdefora2.get(),
                      'SaoBras': self.entry_saobras2.get()
                      }
            self.inputs = {'Barbacena': 0,
                      'Betim': 0,
                      'Brumadinho': 0,
                      'Jacutinga': 0,
                      'JuizdeFora': 0,
                      'SaoBras': 0
                      }
            
            
            any_empty_inputs24 = bool(len(['' for x in self.inputs24.values() if not x]))
            any_empty_inputs11 = bool(len(['' for x in self.inputs11.values() if not x]))
            
            
            if any_empty_inputs24 == False or any_empty_inputs11 == False:
                
                day = self.hoje.weekday()
                
                
                for ponto in ['Barbacena', 'Betim', 'Brumadinho', 'Jacutinga', 'JuizdeFora', 'SaoBras']:
                    
                    #vazao11 de hoje
                    vazao11_hoje = locale.atof(self.inputs11[ponto])
                    vazao11_hoje = np.array([float(vazao11_hoje)]).reshape((-1, 1))[0][0]
                    
                    
                    if self.hoje.weekday() == 5 and ponto == 'Betim':
                        #Se hoje é sábado, então weekday = 6, value11_ontem e vazao24_ontem são de domingo passado
                        
                        domingo_passado = self.hoje - datetime.timedelta(days=6)
                        
                        day2 = domingo_passado.weekday()
                        
                        #vazao11 de ontem e vazao24 de ontem
                        try:
                            value11_ontem = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,domingo_passado), self.database_24_path)[0][0]
                            vazao24_ontem = sd.executep("SELECT vazao24_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,domingo_passado), self.database_24_path)[0][0]
                        except:
                            value11_ontem = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,domingo_passado), self.database_24_path)
                            vazao24_ontem = sd.executep("SELECT vazao24_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,domingo_passado), self.database_24_path)
                    elif self.hoje.weekday() == 0 and ponto == 'Betim':
                        #Se hoje é sábado, então weekday = 6, value11_ontem e vazao24_ontem são de domingo passado
                        
                        sexta_passado = self.hoje - datetime.timedelta(days=3)
                        
                        day2 = sexta_passado.weekday()
                        
                        #vazao11 de ontem e vazao24 de ontem
                        try:
                            value11_ontem = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,sexta_passado), self.database_24_path)[0][0]
                            vazao24_ontem = sd.executep("SELECT vazao24_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,sexta_passado), self.database_24_path)[0][0]
                        except:
                            value11_ontem = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,sexta_passado), self.database_24_path)
                            vazao24_ontem = sd.executep("SELECT vazao24_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,sexta_passado), self.database_24_path)
                                                          
                    else:
                        #Se não é sábado, value11_ontem e vazao24_ontem são do dia anterior
                        
                        day2 = self.ontem.weekday()
                                                
                        #vazao24 de ontem
                        vazao24_ontem = locale.atof(self.inputs24[ponto])
                        vazao24_ontem = np.array([float(vazao24_ontem)]).reshape((-1, 1))[0][0]
                        
                        #vazao11 de ontem
                        try:
                            value11_ontem = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,self.ontem), self.database_24_path)[0][0]
                        except:
                            value11_ontem = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                            and data = ?", (ponto,self.ontem), self.database_24_path)
                        
                    if value11_ontem: #verifica se value11_ontem e value11_anteontem estão vazios
                        #Aqui faz adaptação 11-11
                        self.model.predict(ponto, float(vazao11_hoje), float(np.nan), float(value11_ontem), day2)
                        #Aqui faz adaptação 11-24
                        self.model.predict(ponto, float(value11_ontem), float(vazao24_ontem), float(np.nan), day2)
                    
                    #Aqui faz previsão
                    a = self.model.predict(ponto, float(vazao11_hoje), float(np.nan), float(np.nan), day)              
                    
                    
                    self.yhat = np.round(np.float64(a).reshape(1,1)[0][0], 2)
                        
                    y = str(locale.currency(self.yhat, grouping=True, symbol=False))
                    y_print = self.replace(y)
                    
                    self.inputs[ponto] = y_print
                    
                    self.print_estimado(text_tab)
                
                #Salva no banco de dados
                self.save_24(text_tab)
                    
            else:
                self.textbox_msg.insert("0.0", "*Preencha todos os campos.")
                self.textbox_msg.configure(text_color='red')
            

    def print_estimado(self, text_tab):
        
        self.label_barbacena4.configure(text=self.inputs["Barbacena"])
        
        self.label_betim4.configure(text=self.inputs["Betim"])
        
        self.label_brumadinho4.configure(text=self.inputs["Brumadinho"])

        self.label_juizdefora4.configure(text=self.inputs["JuizdeFora"])

        self.label_jacutinga4.configure(text=self.inputs["Jacutinga"])
        
        self.label_saobras4.configure(text=self.inputs["SaoBras"])


    def return_lr(self, file):
        for x in file: 
            if x[0].value == None:
               lr = x
               return lr[0].row
            else:
               lr = x 
        return lr[0].row
    
    
    def save_24(self, text_tab):

        
        self.inputs24 = {'Barbacena': self.entry_barbacena1.get(),
                  'Betim': self.entry_betim1.get(),
                  'Brumadinho': self.entry_brumadinho1.get(),
                  'Jacutinga': self.entry_jacutinga1.get(),
                  'JuizdeFora': self.entry_juizdefora1.get(),
                  'SaoBras': self.entry_saobras1.get()
                  }
        self.inputs11 = {'Barbacena': self.entry_barbacena2.get(),
                  'Betim': self.entry_betim2.get(),
                  'Brumadinho': self.entry_brumadinho2.get(),
                  'Jacutinga': self.entry_jacutinga2.get(),
                  'JuizdeFora': self.entry_juizdefora2.get(),
                  'SaoBras': self.entry_saobras2.get()
                  }
        
        self.textbox_msg.delete("0.0","end")
        
        query = sd.executep("SELECT * FROM relatorio24 WHERE data = ?", (self.ontem,), self.database_24_path)        
        
        for ponto in ['Barbacena', 'Betim', 'Brumadinho', 'Jacutinga', 'JuizdeFora', 'SaoBras']:

            data = self.inputs[ponto].replace('.', '')
            data1 = data.replace(' ', '.')
            vazao_estimado = float(locale.atof(data1))
            
            vazao11 = self.inputs11[ponto]
            
            vazao_24 = self.inputs24[ponto]
                        
            if query:
                sd.executep("UPDATE relatorio24 SET vazao24_real = ? \
                            WHERE data = ? and ponto = ?", (vazao_24, self.ontem, ponto), self.database_24_path)
            else:
                sd.executep("INSERT INTO relatorio24 VALUES(?, ?, ?, ?, ?)", (ponto, self.ontem, vazao_24, None, None), self.database_24_path)
            
            sd.executep("INSERT INTO relatorio24 VALUES(?, ?, ?, ?, ?)", (ponto, self.hoje, None, vazao11, vazao_estimado), self.database_24_path)

        self.textbox_msg.insert("0.0", "*Previsão para o dia "+(self.hoje).strftime('%d/%m/%Y')+" salva com sucesso.")
        self.textbox_msg.configure(text_color='green')
        


    
###############################################################################################################################   


    def plot_previsao(self, ponto, previsao_intervalar, previsao_pontual):
        #previsao_intervalar, previsao_pontual = self.predict_7(ponto)

        sup, inf = [], []
        for row in range(len(previsao_intervalar)):
            sup.append(previsao_intervalar[row][1])
            inf.append(previsao_intervalar[row][0])
            
        fig, ax = plt.subplots(1,1, figsize=(8,4))
                
        fig.patch.set_facecolor('gainsboro')
        ax.set_facecolor('gainsboro')
        
        ax.fill_between(range(1,len(previsao_pontual)+1), inf, sup, facecolor='dodgerblue', alpha = 0.5, label='PREVISÃO INTERVALAR')
        ax.plot(range(1,len(previsao_pontual)+1),previsao_pontual, 'black', marker='o', ms = 5, linewidth=1, label='PREVISÃO PONTUAL')
        
        #plt.ylabel("Vazão",fontsize=12)
        plt.xlabel("Horizontes de previsão",fontsize=10)
        plt.tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.rcParams['axes.spines.right'] = False
        ax.spines['left'].set_visible(False)
        
        
        labels = [self.hoje.strftime('%d/%m'),
                  (self.hoje + datetime.timedelta(days=1)).strftime('%d/%m'), 
                  (self.hoje + datetime.timedelta(days=2)).strftime('%d/%m'), 
                  (self.hoje + datetime.timedelta(days=3)).strftime('%d/%m'),
                  (self.hoje + datetime.timedelta(days=4)).strftime('%d/%m'),
                  (self.hoje + datetime.timedelta(days=5)).strftime('%d/%m'), 
                  (self.hoje + datetime.timedelta(days=6)).strftime('%d/%m'),
                  (self.hoje + datetime.timedelta(days=7)).strftime('%d/%m')]
        
        ax.set_xticklabels(labels)
        
        plt.yticks([])

        
        #Plot previsões pontuais
        for x,y in zip(range(1,len(previsao_pontual)+1),previsao_pontual):
        
            label = locale.currency(y, grouping=True, symbol=None)
                    
            ax.annotate(self.replace(label), # this is the text
                         (x,y), # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         fontsize=10,
                         weight="bold",
                         ha='center')
        
        #Plot previsões intervalares superioes
        for x,y in zip(range(1,len(previsao_pontual)+1),sup):
        
            label = locale.currency(y, grouping=True, symbol=None)
        
            ax.annotate(self.replace(label), # this is the text
                         (x,y), # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         fontsize=8,
                         ha='center')
        #ax.legend(loc='lower left')
        
        #Plot previsões intervalares inferiores
        for x,y in zip(range(1,len(previsao_pontual)+1),inf):
        
            label = locale.currency(y, grouping=True, symbol=None)
        
            ax.annotate(self.replace(label), # this is the text
                         (x,y), # these are the coordinates to position the label
                         textcoords="offset points", # how to position the text
                         xytext=(0,0), # distance from text to points (x,y)
                         fontsize=8,
                         ha='center',
                         va = "top")
        #ax.legend(facecolor='gainsboro')
        #fig.subplots_adjust(left=0.1, bottom=0.01)
        if ponto == "JuizdeFora":
            ponto = 'Juiz de Fora'
        if ponto == "SaoBras":
            ponto = 'São Brás'
        canvas = FigureCanvasTkAgg(fig,master=self.tabview_7.tab(ponto))
        #canvas.draw()
        canvas.get_tk_widget().place(relx=0, rely=0.15)

    def predict_7(self, ponto):
        
        self.textbox_msg.delete("0.0","end")
         
        try:
            vazao11_hoje = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                and data = ?", (ponto,self.hoje), self.database_24_path)[0][0]
            
            vazao24_ontem = sd.executep("SELECT vazao24_real FROM relatorio24 WHERE ponto = ? \
                                            and data = ?", (ponto,self.ontem), self.database_24_path)[0][0]
        except:
            vazao11_hoje = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                and data = ?", (ponto,self.hoje), self.database_24_path)
            
            vazao24_ontem = sd.executep("SELECT vazao24_real FROM relatorio24 WHERE ponto = ? \
                                            and data = ?", (ponto,self.ontem), self.database_24_path)
        
        if vazao11_hoje:
            
            previsao_intervalar = []
            previsao_pontual = []
            
            for KMais in range(1,8):
                
                dia = self.hoje.weekday()
                
                if self.hoje.weekday() == 5:
                    ontem = self.hoje - datetime.timedelta(days=6)
                elif self.hoje.weekday() == 0:
                    ontem = self.hoje - datetime.timedelta(days=3)
                else:
                    ontem = self.hoje - datetime.timedelta(days=1)
                
                dia2 = self.hoje.weekday()
                
                for k in range(0, KMais+1):
                    if dia2 == 5:
                        anterior = self.hoje - datetime.timedelta(days=6)
                        dia2 = 6
                    elif dia2 == 0:
                        anterior = self.hoje - datetime.timedelta(days=3)
                        dia2 = 4
                    else:
                        anterior = self.hoje - datetime.timedelta(days=1)
                        dia2 = dia2 - 1
                
                
                #ontem_KMais = self.ontem - datetime.timedelta(days=KMais)
                    
                try:
                    vazao11_anterior = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                        and data = ?", (ponto,anterior), self.database_24_path)[0][0]
                    vazao24_ontem = sd.executep("SELECT vazao24_real FROM relatorio24 WHERE ponto = ? \
                                                    and data = ?", (ponto,self.ontem), self.database_24_path)[0][0]
                except:
                    vazao11_anterior = sd.executep("SELECT vazao11_real FROM relatorio24 WHERE ponto = ? \
                                                        and data = ?", (ponto,anterior), self.database_24_path)
                    vazao24_ontem = sd.executep("SELECT vazao24_real FROM relatorio24 WHERE ponto = ? \
                                                     and data = ?", (ponto,self.ontem), self.database_24_path)
                
                if vazao11_anterior:
                    if anterior.weekday() >= 0 and ontem.weekday() >= 0:
                        self.model.predict_KMais(ponto, vazao11_anterior, vazao24_ontem, KMais, dia2)
                    
                #Prever de hoje em diante
                inf, sup = self.model.predict_KMais(ponto, vazao11_hoje, float(np.nan), KMais, dia)
                
                pontual = np.mean([inf, sup])
                
                sup = np.round(np.float64(sup).reshape(1,1)[0][0], 2)
                
                inf = np.round(np.float64(inf).reshape(1,1)[0][0], 2)
                
                previsao_intervalar.append([inf,sup])
                
                previsao_pontual.append(np.round(np.float64(pontual).reshape(1,1)[0][0], 2))
            
                
            #Salva no banco de dados
            self.save_7(ponto, previsao_pontual, previsao_intervalar)
            
            self.plot_previsao(ponto, previsao_intervalar, previsao_pontual)
        
        else:
            self.textbox_msg.insert("0.0", "*Não existem dados de hoje para realizar as previsões. Favor executar a estimativa 24 horas antes.")
            self.textbox_msg.configure(text_color='red')
            

                    
            
    def save_7(self, ponto, previsao_pontual, previsao_intervalar):

        self.textbox_msg.delete("0.0","end")
        
        
        sd.executep("INSERT INTO relatorio7_pontual VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                    (ponto, self.hoje, previsao_pontual[0], previsao_pontual[1], previsao_pontual[2], 
                     previsao_pontual[3], previsao_pontual[4], previsao_pontual[5], previsao_pontual[6]), 
                    self.database_7_path)
        
        sd.executep("INSERT INTO relatorio7_intervalsup VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                    (ponto, self.hoje, previsao_intervalar[0][1], previsao_intervalar[1][1], previsao_intervalar[2][1], 
                     previsao_intervalar[3][1], previsao_intervalar[4][1], previsao_intervalar[5][1], previsao_intervalar[6][1]), 
                    self.database_7_path)
        
        sd.executep("INSERT INTO relatorio7_intervalinf VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                    (ponto, self.hoje, previsao_intervalar[0][0], previsao_intervalar[1][0], previsao_intervalar[2][0], 
                     previsao_intervalar[3][0], previsao_intervalar[4][0], previsao_intervalar[5][0], previsao_intervalar[6][0]), 
                    self.database_7_path)
        
        if ponto == 'JuizdeFora':
            ponto = 'Juiz de Fora'
        if ponto == 'SaoBras':
            ponto = 'São Brás'
        
        
        self.textbox_msg.insert("0.0", "*Previsões para "+ponto+" salvas com sucesso.")
        self.textbox_msg.configure(text_color='green')
    
    
    def download_7(self, ponto):
        
        self.textbox_msg.delete("0.0","end")
        
        try:
        
            df1 = pd.DataFrame(sd.executep("SELECT * FROM relatorio7_pontual WHERE ponto = ?", (ponto,), self.database_7_path))
            df1.columns = ['ponto', 'data', 'Vazao_24_1', 'Vazao_24_2', 'Vazao_24_3','Vazao_24_4', 'Vazao_24_5', 'Vazao_24_6', 'Vazao_24_7']
            df1.to_csv('Downloads_7dias/'+ponto+'_pontual.csv', index=True, sep=';')
            
            df2 = pd.DataFrame(sd.executep("SELECT * FROM relatorio7_intervalsup WHERE ponto = ?", (ponto,), self.database_7_path))
            df2.columns = ['ponto', 'data', 'Vazao_24_1', 'Vazao_24_2', 'Vazao_24_3','Vazao_24_4', 'Vazao_24_5', 'Vazao_24_6', 'Vazao_24_7']
            df2.to_csv('Downloads_7dias/'+ponto+'_intervalo_superior.csv', index=True, sep=';')
            
            df3 = pd.DataFrame(sd.executep("SELECT * FROM relatorio7_intervalinf WHERE ponto = ?", (ponto,), self.database_7_path))
            df3.columns = ['ponto', 'data', 'Vazao_24_1', 'Vazao_24_2', 'Vazao_24_3','Vazao_24_4', 'Vazao_24_5', 'Vazao_24_6', 'Vazao_24_7']
            df3.to_csv('Downloads_7dias/'+ponto+'_intervalo_inferior.csv', index=True, sep=';')
            
            if ponto == 'JuizdeFora':
                ponto = 'Juiz de Fora'
            if ponto == 'SaoBras':
                ponto = 'São Brás'
            
            self.textbox_msg.insert("0.0", "*A base de dados de "+ponto+" foi salva na pasta Downloads.")
            self.textbox_msg.configure(text_color='green')
        
        except:
            
            self.textbox_msg.insert("0.0", "*Não existem registros no banco de dados.")
            self.textbox_msg.configure(text_color='red')
        


###############################################################################################################################


    def clear_entry(self):
        self.entry_barbacena1.delete(0, "end")
        self.entry_betim1.delete(0, "end")
        self.entry_brumadinho1.delete(0, "end")
        self.entry_jacutinga1.delete(0, "end")
        self.entry_juizdefora1.delete(0, "end")
        self.entry_saobras1.delete(0, "end")
        
        self.entry_barbacena2.delete(0, "end")
        self.entry_betim2.delete(0, "end")
        self.entry_brumadinho2.delete(0, "end")
        self.entry_jacutinga2.delete(0, "end")
        self.entry_juizdefora2.delete(0, "end")
        self.entry_saobras2.delete(0, "end")
    
    
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    # def open_input_dialog_event(self, message):
    #     CTkMessagebox(title="Mensagem", message=message)

    def sidebar_button_event(self):
        print("sidebar_button click")

    def replace(self, str1):
        maketrans = str1.maketrans
        final = str1.translate(maketrans(',.', '.,', ' '))
        return final.replace(',', ", ")


if __name__ == "__main__":
    app = App()
    app.mainloop()
