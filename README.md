# Lab P2-10 - Pipeline Definitivo

*Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Luís Gustavo.*

Pipeline de inferência combinando QLoRA 4-bit e KV Cache para otimizar geração com contexto massivo. Modelo TinyLlama 1.1B recebe texto médico simulando saída de RAG e gera 100 tokens.

## Dependências

    pip install torch transformers bitsandbytes accelerate

## Como rodar

Abrir `pipeline.ipynb` no Google Colab com runtime GPU (T4).

## Benchmark

```
GPU: Tesla T4
VRAM após carga: 788 MB

tokens do RAG bruto: 13021
prompt final: 1914 tokens

sem cache: 12.7s, pico 1954 MB
com cache: 10.1s, pico 1954 MB
speedup: 1.3x
```

Contexto truncado de 13k para 1914 tokens (limite do TinyLlama: 2048). Com Llama-3 (128k de janela), os 15k do RAG caberiam inteiros.

TinyLlama respondeu em inglês por ser treinado predominantemente nesse idioma.

Sobre FA2: FlashAttention-2 exige GPU Ampere+ (A100, H100). T4 é Turing e não suporta, então o pipeline usa atenção padrão do PyTorch com KV Cache ativado.

## Análise Arquitetural

### Parte A

QLoRA comprimiu o TinyLlama de ~2.2 GB (FP16) para ~788 MB (NF4 4-bit), viabilizando a carga na T4. O KV Cache evita recomputar K e V dos tokens anteriores a cada passo de geração. Sem cache, cada novo token roda um forward pass na sequência inteira. Com cache, K e V ficam armazenados e o decode processa 1 token por vez.

O speedup foi 1.3x. Modesto porque o TinyLlama é pequeno (1.1B) e o contexto curto (1914 tokens). Nessa escala o gargalo não é atenção O(n²), é a dequantização NF4 e o FFN. Em modelos de 7B+ com contexto de 8k+, o cache é crítico: gerar 100 tokens sem cache exige 100 forward passes completos em ~8k tokens cada, contra 1 prefill + 100 passes de 1 token com cache.

O pico de VRAM ficou idêntico nos dois cenários (1954 MB) porque o modelo é pequeno o suficiente pro custo da atenção não dominar a memória.

### Parte B

FlashAttention otimiza constantes (5-20x menos tráfego de memória HBM) mas a complexidade assintótica continua O(n²). Com n = 2.000.000 tokens, a matriz de atenção por cabeça teria 4 trilhões de elementos. Inviável até em cluster de A100. O KV Cache também sofre: armazenar K e V para 2M tokens consumiria dezenas de GB só de cache. A solução são State Space Models como o Mamba (Gu & Dao, 2023), que substituem atenção quadrática por recorrência linear com memória O(1) por token.

## Uso de IA

Ferramenta usada: Claude Sonnet 4.6

- Geração do texto médico fictício
- Debug do benchmark de VRAM (reset de peak memory stats entre runs)
- Estruturação do README
