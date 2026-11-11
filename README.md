# MLOps — Análise de Sentimentos

Aplicação de **classificação binária de sentimentos em textos em português**, desenvolvida com **Scikit-learn**, disponibilizada por uma **API REST em FastAPI** e acessível por um **front-end web**. A solução foi implantada em uma instância **Amazon EC2**, com **Nginx** como servidor web/proxy reverso, **systemd** para gerenciamento da API e **HTTPS** por certificado Let's Encrypt para o endereço IP público.

## Demonstração

**Aplicação web (endereço da demonstração):** https://3.230.39.133

O acesso depende de a instância EC2 e o certificado HTTPS continuarem ativos. O certificado usado nesta demonstração possui validade curta e expira em **18/11/2026**; a renovação automática **não foi validada** para esta entrega.

### Funcionalidades

- Inserção de comentários em português com validação de tamanho (3 a 1.000 caracteres na API).
- Predição de sentimento **positivo** ou **negativo**.
- Exibição da **confiança estimada pelo classificador**, em percentual.
- Indicação da versão declarada do modelo: `0.1.0`.
- Interface responsiva em HTML, CSS e JavaScript.
- Endpoint de saúde da aplicação e logs de previsão.

## Arquitetura

```text
Navegador (HTTPS :443)
       |
       v
Amazon EC2 (Ubuntu 24.04, t3.small)
       |
       v
Nginx
  |-- GET /              --> frontend/index.html (cópia publicada em /var/www/mlops-sentimento)
  |-- /api/health        --> FastAPI GET /health
  `-- /api/predict       --> FastAPI POST /predict
                               |
                               v
                       Pipeline Scikit-learn
                       TF-IDF + Regressão Logística
                               |
                               v
                       sentimento + confiança
```

O **Nginx** escuta nas portas 80 (redirecionamento e validação de certificado) e 443 (HTTPS). O serviço **FastAPI/Uvicorn** escuta somente em `127.0.0.1:8000`; a porta 8000 não precisa ser liberada no Security Group.

## Tecnologias utilizadas

| Camada | Tecnologias |
|---|---|
| Dados | CSV e Pandas |
| Processamento de texto | `TfidfVectorizer`, com unigramas e bigramas |
| Classificação | `LogisticRegression` do Scikit-learn |
| Serialização do modelo | Joblib |
| API | FastAPI, Pydantic e Uvicorn |
| Logs | Loguru |
| Front-end | HTML5, CSS3 e JavaScript (`fetch`) |
| Dependências | uv (`pyproject.toml` e `uv.lock`) |
| Infraestrutura | AWS EC2, Ubuntu, Nginx, systemd e Let's Encrypt |
| Versionamento | Git e GitHub |

## Organização do repositório

```text
mlops_analise_sentimento/
├── app/
│   └── api.py                 # Endpoints FastAPI
├── data/
│   └── comentarios.csv        # Dataset rotulado
├── frontend/
│   └── index.html             # Interface web com HTML, CSS e JavaScript
├── models/
│   └── sentiment_model.joblib # Pipeline treinado e serializado
├── src/
│   ├── train.py               # Treinamento, avaliação e persistência
│   └── inference.py           # Exemplos de inferência local
├── pyproject.toml             # Dependências e metadados
├── uv.lock                    # Dependências resolvidas
└── README.md                  # Documentação
```

## Dataset

O arquivo `data/comentarios.csv` contém **358 comentários** rotulados, com duas colunas:

- `texto`: comentário em português.
- `sentimento`: classe-alvo (`positivo` ou `negativo`).

**Distribuição:** 180 comentários positivos (50,3%) e 178 negativos (49,7%). Trata-se de uma base pequena, adequada para prova de conceito, mas limitada para medir generalização em cenários reais.

## Treinamento e avaliação

O script `src/train.py` executa as seguintes etapas:

1. Lê o CSV com Pandas e separa as variáveis `texto` (**X**) e `sentimento` (**y**).
2. Divide os dados em **80% treinamento** e **20% teste**, com `random_state=42`.
3. Aplica `TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 2))`.
4. Treina `LogisticRegression(max_iter=1000)` dentro de um `Pipeline` Scikit-learn.
5. Calcula **acurácia** e **F1-score macro** no conjunto de teste.
6. Salva o pipeline completo em `models/sentiment_model.joblib`.

### Resultados observados na EC2

| Métrica | Resultado |
|---|---:|
| Acurácia (Accuracy) | **90,28%** (`0.9028`) |
| F1-score macro | **90,23%** (`0.9023`) |

Esses resultados correspondem à execução registrada durante o deploy na EC2. **Não representam uma garantia de desempenho em dados externos**; a avaliação usa apenas a partição de teste do dataset disponível. O treinamento atual não realiza validação cruzada nem busca sistemática de hiperparâmetros.

## Como executar localmente

**Pré-requisitos:** Python 3.10 ou superior, Git e [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GabsFMA/mlops_analise_sentimento.git
cd mlops_analise_sentimento
uv sync --frozen

# Treinar e salvar o modelo
uv run python src/train.py

# Executar exemplos de previsão local
uv run python src/inference.py

# Inicializar a API
uv run uvicorn app.api:app --host 127.0.0.1 --port 8000
```

A documentação interativa da API estará em `http://127.0.0.1:8000/docs`. O processo Uvicorn precisa continuar em execução durante os testes.

## Como testar a API

### Saúde da aplicação

```bash
curl http://127.0.0.1:8000/health
```

Resposta esperada:

```json
{"status":"ok","versao_modelo":"0.1.0"}
```

### Classificação de um comentário

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texto":"O atendimento foi excelente!"}'
```

Formato de resposta (valor meramente ilustrativo):

```json
{"sentimento":"positivo","confianca":0.79,"versao_modelo":"0.1.0"}
```

A propriedade `confianca` é a maior probabilidade estimada por `predict_proba()`, e **não** deve ser interpretada como uma taxa de acerto comprovada ou probabilidade necessariamente calibrada.

No ambiente publicado, o Nginx expõe os caminhos **`/api/health`** e **`/api/predict`**, repassando-os internamente aos caminhos `/health` e `/predict` da FastAPI:

```bash
curl https://3.230.39.133/api/health
curl -X POST https://3.230.39.133/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"texto":"Gostei muito do atendimento!"}'
```

## Implantação na AWS EC2

A implantação de demonstração seguiu este fluxo:

1. Criação de uma EC2 `t3.small`, com Ubuntu Server 24.04.
2. Acesso à instância por SSH/EC2 Instance Connect.
3. Instalação de Python, Git, uv e dependências com `uv sync --frozen`.
4. Execução do treinamento e validação da inferência na instância.
5. Inicialização da FastAPI com Uvicorn e configuração de serviço `systemd` (`mlops-sentimento.service`).
6. Instalação do Nginx e configuração como proxy reverso para `127.0.0.1:8000`.
7. Publicação do HTML em `/var/www/mlops-sentimento/index.html`.
8. Associação do **Elastic IP `3.230.39.133`** à EC2.
9. Emissão de certificado Let's Encrypt válido para o IP e configuração de HTTPS na porta 443.
10. Validação com `curl`, PowerShell e teste funcional no navegador/celular.

### Comandos úteis de verificação na EC2

```bash
sudo systemctl status mlops-sentimento --no-pager
sudo systemctl status nginx --no-pager
sudo nginx -t
curl -I https://3.230.39.133
curl https://3.230.39.133/api/health
```

O certificado foi emitido com o Certbot e está localizado em `/etc/letsencrypt/live/3.230.39.133/`. **Não compartilhe o arquivo `privkey.pem`.** A emissão do certificado foi concluída, mas o teste da renovação automática foi deixado fora do escopo da apresentação acadêmica.

## Limitações conhecidas e possíveis melhorias

- **Dataset reduzido:** ampliar a quantidade e a diversidade dos comentários; avaliar textos neutros, ambíguos e irônicos.
- **Avaliação:** adicionar matriz de confusão, precisão/recall por classe, validação cruzada e testes com dados externos.
- **MLOps:** incorporar rastreamento de experimentos, versionamento de modelos/dados e CI/CD.
- **Observabilidade:** acompanhar métricas de uso, latência, erros e mudanças na distribuição de dados (drift).
- **Segurança:** incluir autenticação, limitação de requisições e políticas de retenção de logs antes de uso público amplo.
- **Certificados:** validar renovação automática caso a aplicação precise permanecer online após a apresentação.

## Evidências sugeridas para a apresentação

1. Terminal com os logs de `src/train.py`, mostrando Accuracy e F1-score.
2. Execução de `src/inference.py`, com exemplos positivo e negativo.
3. Resposta JSON do endpoint `/api/health`.
4. Resultado de `/api/predict` via `curl` ou pela documentação da API.
5. Tela do front-end exibindo uma classificação e confiança estimada.
6. Serviço Nginx e `mlops-sentimento` ativos no `systemd`.
7. Navegador acessando a aplicação com HTTPS.

## Natureza do projeto

**Trabalho acadêmico / prova de conceito.** A solução demonstra a integração entre Machine Learning e implantação em nuvem, mas não deve ser considerada pronta para produção ou para processamento de dados sensíveis sem controles adicionais.
