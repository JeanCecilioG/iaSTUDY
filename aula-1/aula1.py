from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

preco = [
    "quanto custa esse produto",
    "qual o preço disso",
    "qual o valor desse item",
    "quanto fica esse brinquedo",
    "esse produto sai por quanto",
    "me fala o preço",
    "quanto tá custando",
    "qual o valor à vista",
    "tem algo até 50 reais",
    "o que você tem nessa faixa de preço",
    "esse item é caro",
    "tem opção mais barata",
    "qual o menor preço",
    "quanto fica para levar hoje",
    "qual o preço final",
    "esse brinquedo custa quanto",
    "me passa o valor",
    "qual o preço dele",
    "quanto eu gasto nesse produto",
    "qual o custo desse item"
]

disponibilidade = [
    "tem esse produto",
    "esse item está disponível",
    "ainda tem em estoque",
    "tem disponível aí",
    "vocês ainda têm esse brinquedo",
    "esse produto acabou",
    "tem essa boneca",
    "esse carrinho ainda tem",
    "tem pronta entrega",
    "consigo encontrar esse item com vocês",
    "tem no estoque agora",
    "esse produto está em falta",
    "ainda vende isso",
    "vocês têm esse modelo",
    "esse item ainda está disponível",
    "tem essa opção aí",
    "esse brinquedo ainda tem",
    "tem sobrando no estoque",
    "ainda encontro esse produto",
    "esse item já acabou"
]

entrega = [
    "faz entrega",
    "entrega para minha cidade",
    "qual o valor do frete",
    "chega hoje",
    "entrega hoje ainda",
    "vocês enviam",
    "tem entrega por motoboy",
    "quanto custa o frete",
    "entrega no meu bairro",
    "quanto tempo demora para chegar",
    "faz envio",
    "entrega amanhã",
    "vocês levam em casa",
    "tem entrega rápida",
    "como funciona a entrega",
    "consegue mandar hoje",
    "faz entrega aqui perto",
    "o frete é grátis",
    "vocês despacham esse produto",
    "chega no mesmo dia"
]

compra = [
    "quero comprar agora",
    "como faço para comprar",
    "quero levar esse",
    "posso fechar a compra",
    "quero pedir esse produto",
    "me manda o link para comprar",
    "dá para comprar agora",
    "vou levar esse item",
    "quero garantir esse produto",
    "como finalizo a compra",
    "tem como comprar pelo whatsapp",
    "quero fechar pedido",
    "posso reservar para comprar",
    "me ajuda a comprar esse",
    "quero pedir agora",
    "já decidi comprar",
    "como faço o pedido",
    "vou comprar esse brinquedo",
    "quero fechar com vocês",
    "onde clico para comprar"
]

negociacao = [
    "tem desconto",
    "consegue fazer um preço melhor",
    "faz mais barato",
    "qual o desconto à vista",
    "esse valor pode melhorar",
    "faz por menos",
    "consegue abaixar o preço",
    "tem promoção",
    "se eu levar dois tem desconto",
    "qual o melhor valor que você faz",
    "tem cupom",
    "esse preço é negociável",
    "me faz um desconto",
    "se pagar no pix melhora",
    "tem condição melhor",
    "faz desconto no pagamento à vista",
    "esse é o menor preço mesmo",
    "consegue dar um abatimento",
    "tem oferta nesse produto",
    "qual valor você consegue fazer"
]


textos = []
classes = []

def add_classe(lista, nome):
    for frase in lista:
        textos.append(frase)
        classes.append(nome)

add_classe(preco, "preco")
add_classe(disponibilidade, "disponibilidade")
add_classe(entrega, "entrega")
add_classe(compra, "compra")
add_classe(negociacao, "negociacao")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

X_train, X_test, y_train, y_test = train_test_split(
    X, classes, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Shape:", X.shape)
print("Vocabulário:", vectorizer.get_feature_names_out()[:10])  # primeiras palavras

