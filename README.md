# Dengue4 Forecasting

Repositório de experimentos para previsão de casos de dengue em capitais brasileiras.

## Estrutura principal
- `src/`: código-fonte dos modelos, utilitários e rotinas de treinamento.
- `check_data.py`: script auxiliar para inspecionar e validar datasets de entrada.
- `artifacts/batch_capital_cities/20251013_144325/`: principais resultados do lote mais recente (relatórios, métricas, previsões e gráficos).

## Como reproduzir
1. Crie um ambiente virtual e instale as dependências listadas no projeto (por exemplo, `pip install -r requirements.txt` se disponível).
2. Ajuste os arquivos de configuração em `src/config.py` conforme a execução desejada.
3. Execute o pipeline ou os scripts de treinamento apropriados em `src/` (por exemplo, `python src/train_gru.py`).

## Resultados
Os outputs consolidados do treinamento por capitais estão em `artifacts/batch_capital_cities/20251013_144325/`, incluindo relatórios (`REPORT.md`), comparativos de métricas e previsões por cidade.
