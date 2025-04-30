## Modelo BERT Transformer

BERT (Bidirectional Encoder Representations from Transformers) es un modelo de **transformer pre-entrenado** que revolucionó el campo del Procesamiento del Lenguaje Natural (NLP). Su principal innovación radica en su capacidad para entender el **contexto bidireccional** de las palabras en una oración. A diferencia de modelos anteriores que procesaban el texto de izquierda a derecha o de derecha a izquierda, BERT considera ambas direcciones simultáneamente.

Imagina que estás leyendo la frase: "El banco del río estaba lleno de peces" y "Fui al banco a retirar dinero". Para entender el significado de la palabra "banco" en cada caso, necesitas observar las palabras que la rodean tanto a la izquierda como a la derecha. BERT aprende a hacer esto de manera efectiva.

El término "Transformer" en su nombre se refiere a la arquitectura de red neuronal en la que se basa. Esta arquitectura, introducida en el famoso artículo "Attention Is All You Need", demostró ser muy eficaz para capturar dependencias a largo alcance en las secuencias de texto, superando las limitaciones de las redes neuronales recurrentes (RNNs) como las LSTM.

**Entonces sus principales caracteristicas son que:**

* **Se basa en la arquitectura Transformer.**
* **Está pre-entrenado en grandes cantidades de texto sin etiquetar.** Esto le permite aprender representaciones ricas del lenguaje.
* **Entiende el contexto de las palabras de forma bidireccional.**
* **Puede ser ajustado (fine-tuned) para una amplia variedad de tareas de PNL**, como clasificación de texto, respuesta a preguntas, extracción de entidades nombradas, etc.

## Componentes Clave de la Arquitectura BERT Transformer

La arquitectura de BERT se basa principalmente en el **mecanismo de atención (Attention Mechanism)** y se compone de varias capas de **codificadores Transformer**. Vamos a ver los componentes más importantes:

### 1. El Mecanismo de Atención (Attention Mechanism)

Este es el corazón de la arquitectura Transformer y, por ende, de BERT. El mecanismo de atención permite al modelo **ponderar la importancia de diferentes palabras en una secuencia** al procesar una palabra en particular.

Imagina que estás traduciendo la frase "La capital de Francia es París". Cuando el modelo está procesando la palabra "París", el mecanismo de atención le permite darse cuenta de que las palabras "capital" y "Francia" son muy relevantes para entender su significado.

**¿Cómo funciona a nivel técnico?**

Para cada palabra en la secuencia de entrada, el modelo calcula tres vectores:

* **Query (Q):** Representa la consulta de la palabra actual.
* **Key (K):** Representa la clave de cada una de las palabras en la secuencia.
* **Value (V):** Representa el valor asociado a cada una de las palabras en la secuencia.

Luego, se calcula una **puntuación de atención** entre la Query de la palabra actual y las Keys de todas las demás palabras. Esta puntuación indica cuánto "presta atención" la palabra actual a cada una de las otras palabras. Una forma común de calcular esta puntuación es mediante el producto punto escalado:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Donde:

* $Q$ es la matriz de Queries.
* $K$ es la matriz de Keys.
* $V$ es la matriz de Values.
* $d_k$ es la dimensión de los vectores Key (se utiliza para estabilizar los gradientes).
* $\text{softmax}$ normaliza las puntuaciones de atención para que sumen 1.

El resultado es una **suma ponderada de los vectores Value ($K$)**, donde los pesos están dados por las puntuaciones de atención. Esto significa que las palabras a las que la palabra actual presta más atención tendrán una mayor influencia en su representación final.

### 2. Auto-Atención Multi-Cabeza (Multi-Head Self-Attention)

BERT no utiliza un único mecanismo de atención, sino múltiples mecanismos de atención paralelos, cada uno llamado "cabeza" o "head". Esto permite al modelo capturar diferentes tipos de relaciones entre las palabras.

Imagina que una cabeza de atención se enfoca en las relaciones sintácticas (sujeto-verbo), mientras que otra se enfoca en las relaciones semánticas (sinónimos, antonónimos). Al combinar la información de múltiples cabezas de atención, el modelo puede construir representaciones más ricas y completas de las palabras.

La salida de cada cabeza de atención se concatena y luego se proyecta a través de una capa lineal para producir la salida final de la capa de Auto-atención multi-cabeza.

### 3. Capas de Codificador Transformer (Transformer Encoder Layers)

La arquitectura de BERT se compone de múltiples capas de codificadores Transformer apiladas secuencialmente. Cada capa de codificador contiene dos sub-capas principales:

* **Capa de Auto-Atención Multi-Cabeza:** Como se explicó anteriormente, esta capa procesa la secuencia de entrada y calcula las representaciones contextualizadas de cada palabra.
* **Capa de Feed-Forward:** Esta es una red neuronal feed-forward de dos capas que se aplica de forma independiente a la salida de cada posición en la secuencia. Ayuda a procesar la información capturada por la capa de auto-atención.

**Conexiones Residuales y Normalización de Capa (Residual Connections and Layer Normalization):**

Después de cada una de estas sub-capas (auto-atención y feed-forward), se aplica una **conexión residual** (la salida de la sub-capa se suma a su entrada original) seguida de una **normalización de capa**. Esto ayuda a mejorar el flujo de gradientes durante el entrenamiento y a estabilizar el proceso de aprendizaje.

### 4. Capa de Embedding

Antes de que la secuencia de entrada pase por las capas de codificador Transformer, las palabras se convierten en representaciones vectoriales llamadas **embeddings**. BERT utiliza tres tipos de embeddings que se suman para obtener la representación inicial de cada token:

* **Token Embeddings:** Representaciones vectoriales aprendidas para cada palabra en el vocabulario.
* **Segment Embeddings:** Indicadores que identifican a qué segmento pertenece cada token (útil para tareas que involucran dos secuencias de texto, como la respuesta a preguntas). Para tareas de una sola secuencia, todos los embeddings de segmento son iguales.
* **Position Embeddings:** Vectores que codifican la posición de cada token en la secuencia. Esto es crucial porque el mecanismo de atención por sí solo no tiene información sobre el orden de las palabras.

### 5. Capa de Salida para Tareas Específicas

La salida de la última capa del codificador Transformer es una secuencia de vectores contextualizados, donde cada vector corresponde a una palabra en la secuencia de entrada y captura su significado en función del contexto de toda la oración.

Para realizar tareas específicas de PNL, se agrega una capa de salida adicional sobre esta representación. Por ejemplo:

* **Clasificación de Secuencia:** Se toma la representación del token especial `[CLS]` (añadido al inicio de cada secuencia durante el pre-entrenamiento) y se pasa a través de una capa lineal seguida de una función softmax para obtener las probabilidades de las diferentes clases.
* **Clasificación de Token (e.g., Named Entity Recognition):** La representación de cada token en la secuencia se pasa a través de una capa lineal seguida de una función softmax para predecir la etiqueta de cada token.
* **Respuesta a Preguntas:** Se utilizan dos vectores de selección aprendidos para identificar el inicio y el final del fragmento de texto que contiene la respuesta dentro de un contexto dado.

## El Proceso de Pre-entrenamiento de BERT

Una de las claves del éxito de BERT es su **proceso de pre-entrenamiento no supervisado** en grandes cantidades de texto sin etiquetar (como Wikipedia y un gran corpus de libros). BERT se pre-entrena utilizando dos tareas principales:

* **Masked Language Modeling (MLM):** Se enmascara aleatoriamente un cierto porcentaje de las palabras en la secuencia de entrada (por ejemplo, el 15%) y el modelo debe predecir las palabras enmascaradas basándose en el contexto de las palabras no enmascaradas. Esto obliga al modelo a entender el contexto bidireccional.
* **Next Sentence Prediction (NSP):** Se presentan al modelo pares de oraciones. En el 50% de los casos, la segunda oración es la siguiente oración real en el corpus, y en el otro 50%, es una oración aleatoria. El modelo debe predecir si la segunda oración sigue o no a la primera. (Aunque la efectividad de NSP ha sido cuestionada en investigaciones posteriores).

Este pre-entrenamiento permite a BERT aprender representaciones del lenguaje muy ricas y generales que luego pueden ser transferidas y ajustadas para tareas específicas con muchos menos datos etiquetados.