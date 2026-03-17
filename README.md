# VAE PneumoniaMNIST - Triagem e Monitoramento Clínico

Este repositório apresenta a evolução da interface de usuário para o sistema de triagem baseado em Autoencoders Variacionais (VAE). O objetivo central foi transformar o script original em uma aplicação robusta de suporte à decisão clínica, focando em usabilidade, transparência de processos e monitoramento de performance.

## Alterações Realizadas e Justificativa Técnica

O arquivo original serviu como base funcional para o modelo, enquanto as modificações implementadas focaram nos critérios de experiência do usuário e arquitetura de software exigidos para o projeto.

### 1. Reorganização da Arquitetura de Informação
* **Painel de Controle Lateral**: A barra lateral foi configurada como o centro de comando global, isolando os seletores de sensibilidade (thresholds) e o status do modelo da área de trabalho principal.
* **Interface Baseada em Contextos**: A utilização de abas (tabs) permitiu separar o fluxo de trabalho em três etapas distintas: Diagnóstico, Histórico Operacional e Monitoramento de Performance.
* **Tratamento de Estados Iniciais**: Foi implementada uma lógica de orientação que guia o usuário desde o momento do upload até a exibição dos resultados, evitando confusão visual.

### 2. Gerenciamento de Latência e Pipeline Visual
* **Detalhamento do Processamento**: Em vez de uma indicação de carregamento genérica, foi utilizado o componente de status para descrever as etapas do pipeline: pré-processamento de imagem, codificação latente e cálculo da métrica de erro (MSE).
* **Interface de Confiança**: Os resultados agora acompanham métricas de confiança visual e barras de progresso que facilitam a interpretação rápida do nível de anomalia detectado.

### 3. Gestão de Estado e Interatividade Profissional
* **Persistência de Dados**: O histórico de análises e os registros de feedback são mantidos através do gerenciamento de estado da sessão, garantindo que as informações não sejam perdidas durante a navegação entre abas.
* **Sincronização Reativa**: Implementação de funções de retorno (callbacks) que garantem que, ao alterar parâmetros de sensibilidade, a interface limpe resultados anteriores, prevenindo a leitura de diagnósticos obsoletos.

### 4. Ciclo de Feedback (Human-in-the-Loop)
* **Validação por Especialista**: Inclusão de um sistema de feedback direto onde o profissional pode confirmar ou discordar da análise da inteligência artificial.
* **Monitoramento de Degradação**: O sistema calcula automaticamente a acurácia com base nos feedbacks humanos. Caso a taxa de concordância apresente uma queda significativa, um alerta de degradação é exibido para recomendar a revisão do modelo.

## Tecnologias e Dependências
* Streamlit
* TensorFlow / Keras
* Pandas
* Pillow
* NumPy

## Instruções de Instalação e Execução
1. Verifique se os arquivos de pesos e configuração estão presentes no diretório `models/`.
2. Instale as bibliotecas necessárias:
   ```bash
   pip install streamlit tensorflow pandas pillow numpystema no ambiente operacional.


## Inicie a Aplicação pelo Terminal

streamlit run app.py
