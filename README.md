# 🐊 Crocodile Pomodoro Agent

Este projeto é um **agente de estudo estilo Pomodoro** com tema de crocodilos.  
Ele utiliza inteligência artificial para ajudar no foco e gerenciamento de tarefas, funcionando como um assistente interativo.

---

## 🚀 Pré-requisitos

Antes de começar, você precisa ter instalado em sua máquina:

- [Python 3.10+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/)

---

## ⚙️ Configuração do Ambiente

1. Clone este repositório:

   ```bash
   git clone https://github.com/NatanHugo2004/crocodile-agent.git
   cd crocodile-agent
2. Crie um ambiente virtual chamado `lc-crocodile-agent`:

   ```bash
   python -m venv lc-crocodile-agent
3. Ative o ambiente virtual:

   - **Windows (PowerShell):**
     ```bash
     .\lc-crocodile-agent\Scripts\Activate.ps1
     ```

   - **Linux/MacOS:**
     ```bash
     source lc-crocodile-agent/bin/activate
     ```

4. Instale as dependências:

   ```bash
   pip install -r requirements.txt
## 🔑 Configuração da API de IA

Crie um arquivo chamado **`.env`** na raiz do projeto e adicione sua chave de API da IA que deseja utilizar:

```env
API_KEY=sua_chave_aqui
```

## ▶️ Executando o Projeto
Após ativar o ambiente virtual e configurar o .env, basta rodar o agente:
```bash
python crocodile_agent.py
```

Agora, você poderá gerar **resumos inteligentes** e **piadas divertidas** sobre crocodilos para auxiliar nos seus estudos!


