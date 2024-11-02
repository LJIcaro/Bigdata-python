# Bigdata-python
Repositorio do trabalho de roney

# Instruções para Criar um Ambiente Virtual e Instalar Dependências

## Passo a Passo

1. **Abra o terminal (Prompt de Comando ou PowerShell)**:
   - Se estiver usando o Windows, abra o Prompt de Comando ou PowerShell. Se preferir usar o WSL, abra o terminal do WSL.

2. **Navegue até a pasta do seu projeto**:
   - Use o comando `cd` para entrar no diretório onde está o seu projeto. Por exemplo:
     ```bash
     cd C:\Users\icaro.jesus\Desktop\python bigdata
     ```
     (Ajuste o caminho conforme necessário.)

3. **Crie o ambiente virtual**:
   - Execute o seguinte comando para criar um ambiente virtual chamado `venv`:
     ```bash
     python -m venv venv
     ```
     Isso criará uma pasta chamada `venv` dentro do seu diretório do projeto.

4. **Ative o ambiente virtual**:
   - **No Windows**:
     - Para ativar o ambiente virtual no Windows, use:
       ```bash
       .\venv\Scripts\activate
       ```
   - **No WSL**:
     - Para ativar o ambiente virtual no WSL, use:
       ```bash
       source venv/bin/activate
       ```
     Após a ativação, você verá o nome do ambiente virtual (por exemplo, `(venv)`) na frente da linha de comando.

5. **Instale as dependências a partir do `requirements.txt`**:
   - Uma vez que o ambiente virtual esteja ativo, você pode instalar as dependências listadas no arquivo `requirements.txt` com o seguinte comando:
     ```bash
     pip install -r requirements.txt
     ```

6. **Verifique se as dependências foram instaladas**:
   - Você pode listar os pacotes instalados com:
     ```bash
     pip list
     ```

## Nota
Se você não tiver um arquivo `requirements.txt`, você pode criá-lo com as dependências do seu projeto. Para gerar esse arquivo, use o comando:
```bash
pip freeze > requirements.txt
