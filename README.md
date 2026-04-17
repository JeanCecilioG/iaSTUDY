# 🧠 AI Learning Journey — Classificação de Intenção (NLP)

Este repositório documenta minha evolução prática em Inteligência Artificial, começando pelos fundamentos de Machine Learning aplicados a problemas reais.

O foco inicial é construir um sistema capaz de classificar intenções de usuários em conversas, simulando cenários reais de atendimento (como e-commerce e vendas via WhatsApp).

---

## 🎯 Problema

Negócios que vendem via WhatsApp recebem mensagens com diferentes intenções:

- Perguntas sobre preço
- Disponibilidade de produtos
- Entrega
- Intenção de compra
- Negociação

O desafio é automatizar a identificação dessas intenções para melhorar atendimento, velocidade de resposta e conversão.

---

## ⚙️ Solução (Versão Atual)

Foi desenvolvido um classificador de texto utilizando:

- `TfidfVectorizer` → transformação de texto em vetores numéricos
- `LogisticRegression` → modelo de classificação supervisionada
- Dataset manual com frases simulando clientes reais

Pipeline:

1. Coleta de frases por categoria
2. Vetorização com TF-IDF
3. Separação treino/teste
4. Treinamento do modelo
5. Avaliação com métricas
6. Teste com novas frases

---

## 🧪 Classes do Modelo

- `preco`
- `disponibilidade`
- `entrega`
- `compra`
- `negociacao`

---

## 📊 Métricas (Exemplo)

- Acurácia: ~0.75
- Precision, Recall e F1-score avaliados por classe

> Observação: dataset pequeno (didático), foco em aprendizado de conceitos.

---

## 📚 Aprendizados por Aula

### ✅ Aula 1 — Fundamentos de NLP + Classificação

- O que é aprendizado supervisionado
- Criação de dataset rotulado
- Vetorização com TF-IDF
- Treinamento com regressão logística
- Avaliação com métricas (accuracy, precision, recall, f1-score)

---

### ✅ Aula 2 — Inspeção e Interpretação do Modelo

- Uso do modelo com novas frases
- Interpretação de probabilidades (`predict_proba`)
- Entendimento das decisões do modelo
- Análise de palavras mais relevantes por classe
- Identificação de limitações (ex: frases com múltiplas intenções)
- Introdução ao conceito de confiança do modelo

---

## 🔍 Insights Importantes

- O modelo não "entende" linguagem — ele aprende padrões estatísticos
- Palavras como "preço", "frete", "desconto" influenciam diretamente a decisão
- Frases reais podem ter múltiplas intenções → limitação do modelo atual
- Probabilidade é mais importante que o rótulo final em aplicações reais

---

## ⚠️ Limitações Atuais

- Dataset pequeno (baixa generalização)
- Classificação single-label (não lida bem com múltiplas intenções)
- Sem pré-processamento avançado (acentos, stopwords, etc.)
- Modelo simples (baseline)

---

## 🚀 Próximos Passos

- [ ] Melhorar pré-processamento de texto
- [ ] Usar n-grams (bigram/trigram)
- [ ] Expandir dataset com dados mais realistas
- [ ] Implementar matriz de confusão
- [ ] Testar multi-label classification
- [ ] Salvar e servir o modelo (API)
- [ ] Integrar com sistema real (WhatsApp / catálogo digital)

---

## 💡 Aplicação Real

Este projeto é base para:

- Chatbots de atendimento
- Sistemas de triagem de mensagens
- Automação de vendas via WhatsApp
- Classificação de intenção em marketplaces

---

## 👨‍💻 Autor

Jean Cecilio  
Estudante de Ciência da Computação  
Construindo sistemas na interseção entre IA e problemas do mundo real

---
