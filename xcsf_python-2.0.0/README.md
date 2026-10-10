# XCSF in Python — versione 2.1

Reimplementazione del nucleo di regressione di **XCSF**, interamente in Python
con NumPy, a partire dai sorgenti in `xcslib-1.5-rc1-niches` e dai paper in
`references`. Non compila, carica o esegue codice della libreria C++ e non usa
binding, `ctypes`, Cython o subprocess per l'apprendimento.

`XCSFRegressor` espone `fit`, `predict`, `partial_fit`, `score` (R²), `get_params`
e `set_params`. È un estimatore scikit-learn: si può clonare, serializzare con
pickle/joblib e utilizzare in `Pipeline`, `GridSearchCV`, `cross_val_score` e
`MultiOutputRegressor`.

La struttura segue l'articolo **Lanzi e Loiacono (2025), “A Proposal for a Leaner
Narrative of Learning Classifier Systems”**, §6.2 e Algoritmo 7: i dati `X` sono
gli input, `y` è il valore da approssimare e la regressione usa una sola azione
implicita. Il sistema riceve input e target da un ambiente a passo singolo (un
dataset o una funzione campionata), senza un ambiente RL.

La **2.1** riorganizza il pacchetto in componenti separati, come xcslib —
condizioni, azioni, ambienti, funzioni di predizione, classificatore, sistema,
esperimento — e rivede i predittori locali:

- un **solo RLS** (`rls`), in forma QR, con oblio, rumore di processo e varianza
  di misura: sostituisce `rls` e `rlsk` della 2.0;
- un **Lasso online ricorsivo** (`lasso_online`) che calcola la soluzione Lasso
  esatta sulle statistiche di RLS e converge alla velocità di RLS; la versione a
  gradiente prossimale della 2.0 resta disponibile come `lasso_sgd`;
- `constant`, `lms`, `nlms` e `lasso_batch` invariati.

Gli aggiornamenti sono implementati in Python/NumPy; SciPy fornisce la
risoluzione triangolare e la fattorizzazione QR.

| documento | contenuto |
|---|---|
| [docs/architecture.md](docs/architecture.md) | moduli, corrispondenza con xcslib, come estendere, migrazione dalla 2.0 |
| [docs/prediction-updates.md](docs/prediction-updates.md) | teoria delle funzioni di update di tutti i predittori |
| [docs/algorithm.md](docs/algorithm.md) | ciclo di apprendimento, corrispondenza con i sorgenti C++, differenze intenzionali |
| [docs/validation.md](docs/validation.md) | verifiche eseguite |

## Installazione

Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10, scikit-learn >= 1.6 e < 2.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

## Uso come regressore

```python
import numpy as np
from sklearn.model_selection import train_test_split
from xcsf import XCSFRegressor

rng = np.random.RandomState(42)
X = rng.uniform(0, 1, (800, 1))
y = np.sin(2 * np.pi * X[:, 0])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = XCSFRegressor(
    prediction="rls",      # anche "constant", "lms", "nlms" (default), "lasso_online", ...
    population_size=400,   # numero massimo di microclassificatori
    n_epochs=20,
    epsilon_0=0.05,         # errore tollerato nelle unità di y
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
print(model.n_macroclassifiers_, model.n_microclassifiers_)
```

`fit` riparte da una popolazione vuota. Ogni epoca presenta tutti i campioni una
volta; con `shuffle=True` l'ordine cambia a ogni epoca. `random_state` intero
rende riproducibili ordine e operatori evolutivi. La capacità della popolazione
è espressa in **microclassificatori**: una regola può rappresentarne più di uno
tramite `numerosity`.

## Predittori RLS e Lasso

```python
rls = XCSFRegressor(
    prediction="rls", rls_delta=1000.0,   # covarianza iniziale V0 = rls_delta * I
    forgetting_factor=1.0,                # < 1: privilegia le osservazioni recenti
    process_noise=0.0, kalman_noise=False, random_state=42,
)
online = XCSFRegressor(
    prediction="lasso_online", lasso_alpha=0.001,
    rls_delta=1000.0, forgetting_factor=1.0,   # come per RLS; nessun learning rate
    lasso_max_iter=1000, lasso_tol=1e-6, random_state=42,
)
local_batch = XCSFRegressor(
    prediction="lasso_batch", lasso_alpha=0.001,
    lasso_window=256,  # None conserva tutta la storia della singola regola
    lasso_max_iter=1000, lasso_tol=1e-6, random_state=42,
)
gradient = XCSFRegressor(
    prediction="lasso_sgd", prediction_learning_rate=0.05,
    lasso_alpha=0.001, lasso_learning_rate_decay=0.0, random_state=42,
)
online.fit(X_train, y_train)
print(online.get_rules()[0]["prediction_diagnostics"])
```

`rls` è l'unica implementazione di RLS: con i default è la RLS di Lanzi et al.
(2005); `rls_delta=0, process_noise=1` riproduce la `rls` di xcslib.
`lasso_online` restituisce a ogni aggiornamento la soluzione Lasso esatta su tutte
le osservazioni della regola (scontate da `forgetting_factor`), senza conservarle:
usa le statistiche di RLS e pochi passi di discesa per coordinate. `lasso_batch`
risolve lo stesso problema sui campioni conservati in una finestra. `lasso_sgd`
esegue un passo di gradiente prossimale per campione: costa poco ma converge
come LMS e richiede feature scalate e un passo scelto con attenzione.
Tutti escludono l'intercetta dalla penalizzazione. `lasso_alpha` non è il
parametro `alpha` della fitness e va scelto rispetto alla scala del target.
I solver iterativi segnalano la mancata convergenza con `ConvergenceWarning` e
con i diagnostici della regola. Teoria e confronto:
[docs/prediction-updates.md](docs/prediction-updates.md).

## Pipeline e ricerca degli iperparametri

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from xcsf import XCSFRegressor

pipeline = make_pipeline(
    MinMaxScaler(),
    XCSFRegressor(normalize=False, prediction="rls", random_state=42),
)
search = GridSearchCV(
    pipeline,
    {"xcsfregressor__epsilon_0": [0.01, 0.05],
     "xcsfregressor__population_size": [200, 400]},
    cv=3,
    scoring="neg_mean_squared_error",
)
search.fit(X_train, y_train)
print(search.best_params_)
```

Per default il regressore applica una trasformazione min-max stimata **solo sui
dati di training**. `normalize=False` lascia la scala degli input all'utente o
alla pipeline. `cover_radius` e `mutation_scale` sono nelle coordinate interne
delle condizioni. `y` non è normalizzato: scegliere `epsilon_0` rispetto alla
scala e al rumore del target. I polinomi di `degree > 1` aggiungono le potenze di
ciascuna feature, senza prodotti incrociati, come nel codice fornito.

## Apprendimento incrementale

```python
online = XCSFRegressor(prediction="rls", normalize=False, random_state=42)
for start in range(0, len(X_train), 50):
    online.partial_fit(X_train[start:start + 50], y_train[start:start + 50])
```

`partial_fit` esegue **un solo passaggio nell'ordine ricevuto**, indipendentemente
da `n_epochs` e `shuffle`, preservando popolazione e stato casuale. Con
`normalize=True`, la trasformazione è stimata sul primo batch e poi congelata;
non viene ristimata perché cambierebbe il significato delle condizioni e dei
pesi già appresi. Per stream con dominio noto, usare una trasformazione fissa
esterna e `normalize=False`. Cambiare i parametri dell'algoritmo durante una
sequenza di `partial_fit` richiede un nuovo `fit`.

`bounded=True` limita le condizioni al dominio del primo batch; ulteriori batch
di training esterni al dominio vengono rifiutati. Il default `bounded=False`
permette nuovi input oltre il dominio iniziale senza cambiare la trasformazione.

## Ispezione e salvataggio

```python
import joblib

membership = model.match(X_test)  # campioni x macroclassificatori, booleani
rules = model.get_rules()        # copie indipendenti dei dati delle regole
print(rules[0])
print(model.stats_)
print(model.history_[-1])

joblib.dump(model, "model.joblib")
restored = joblib.load("model.joblib")
```

`get_rules()` restituisce condizioni nelle unità originali di `X`; i pesi si
riferiscono invece agli input trasformati `z = (X - feature_offset_) /
feature_scale_`. Il vettore delle feature locali è
`[x0, z1, ..., zd, z1**2, ..., zd**degree]`. Per i predittori costanti si usa
`value`. `population_` espone gli oggetti interni e va trattato come sola lettura.
`niche_history > 0` conserva gli ultimi timestamp di matching di ogni regola.
`history_` riporta l'MSE online prima dell'aggiornamento: **non è una misura su
validation set**.

### Stampare la popolazione

```python
from xcsf.utils import print_population

print_population(model)
print_population(model, precision=6)
with open("population.txt", "w") as output:
    print_population(model, file=output)
```

La tabella mostra tutte le macroregole: ID, condizioni nelle unità originali,
tipo di predittore, pesi (o valore costante), fitness, errore della regola,
esperienza, numerosity e dimensione stimata della nicchia. Per i pesi vengono
stampati anche la base polinomiale e la trasformazione delle feature.

### Andamento del training ogni TOT passi

Una **epoca** è una passata completa del dataset dato a `fit`. Un **passo** è
l'aggiornamento del sistema su un singolo campione, quindi 20 epoche su 500
campioni producono 10.000 passi. `partial_fit` esegue una passata del batch dato
alla chiamata. Anche gli aggiornamenti nelle epoche di condensazione contano
come passi.

Installare la dipendenza opzionale per il plotting:

```bash
python -m pip install -e '.[plot]'
```

```python
import matplotlib.pyplot as plt
from xcsf import XCSFRegressor
from xcsf.utils import plot_training_history

model = XCSFRegressor(
    n_epochs=20, history_interval=100, random_state=42,
).fit(X_train, y_train)

fig, axes = plot_training_history(model)  # MAE e macroclassificatori, ogni 100 passi
fig.savefig("training.png", dpi=150)
plt.show()

# Alternative:
# fig, axes = plot_training_history(model, metric="rmse")
# fig, axes = plot_training_history(model, metric="mse", history="epochs")
```

`history_interval` (default 100) controlla la registrazione **durante il training**.
In `performance_history_`, ogni record contiene il passo cumulativo, il numero
di campioni della finestra, MAE/MSE/RMSE e i numeri di macro/microclassificatori.
Le finestre sono consecutive e non sovrapposte; attraversano epoche, batch e
l'eventuale passaggio alla condensazione. L'errore è quello della predizione
aggregata del sistema **prima dell'aggiornamento sul campione corrente** (dopo
l'eventuale covering). Il numero di classificatori è rilevato **dopo** update,
evoluzione e cancellazione al termine della finestra. Non è la media degli
errori memorizzati nelle regole e non è un errore su test set.

Viene conservata anche l'eventuale finestra finale incompleta, marcata con
`complete=False` e mediata sul suo effettivo numero di campioni. Se si prosegue
con `partial_fit`, quel riepilogo provvisorio viene esteso fino a completare
la stessa finestra, senza duplicare i campioni. `fit` azzera entrambe le
cronologie; `history_interval=None` disattiva quella per passi, mantenendo
`history_` per epoca. Quest'ultima registra ora anche MAE, RMSE e passo cumulativo.

`plot_training_history` restituisce figura e due assi, senza chiamare `show` né
modificare il modello. Si possono passare due assi esistenti con `axes=...`.
Con `history="auto"` (default) usa i dati per passi quando disponibili, altrimenti
quelli per epoca. Per modelli addestrati con la versione precedente, è possibile
plottare lo storico già esistente con `metric="mse"` o `metric="rmse"`: l'asse
indicherà le epoche, poiché i passi intermedi e il MAE non erano registrati.
Per avere una nuova frequenza di registrazione occorre riaddestrare il modello;
il grafico non ricostruisce informazioni mancanti a posteriori.

`predict` non applica covering né aggiorna regole o generatore casuale. Quando
nessuna condizione copre il punto, `unmatched="nearest"` usa le predizioni locali
delle regioni geometricamente più vicine, pesate per fitness in caso di parità.
Si tratta di estrapolazione, che può essere poco affidabile fuori dal dominio.
`unmatched="mean"` usa la media dei target osservati;
`unmatched="raise"` segnala l'assenza di copertura. `match` consente di rilevare
questi casi esplicitamente.

## Componenti implementate

- Condizioni a intervalli reali, covering e matching; azione implicita unica.
- Ambienti a passo singolo: dataset e funzione reale campionata.
- Predizione aggregata pesata per fitness, con costante, LMS, NLMS, RLS (QR, con
  oblio, rumore di processo e varianza di misura), Lasso online ricorsivo, Lasso
  a gradiente prossimale e Lasso batch; basi polinomiali per-feature.
- Errore assoluto, errore quadratico, esperienza, stima della dimensione della
  nicchia, aggiornamento MAM e fitness basata sull'accuratezza relativa.
- GA nelle nicchie, selezione roulette/tournament, crossover a uno/due punti e
  uniforme, mutazione fixed/proportional/Gaussian, eredità dei predittori.
- Numerosity, fusione delle condizioni identiche, GA e match-set subsumption,
  cancellazione fitness/niche-size, fase di condensazione e statistiche.
- API scikit-learn, training batch/incrementale, ispezione e serializzazione.

La reimplementazione riguarda **XCSF per regressione a input reali e target
scalare**. Gli ambienti RL multi-passo, gli altri eseguibili XCS, le
rappresentazioni binarie, le azioni multiple e l'experiment manager C++ non sono
parte del pacchetto.
Le configurazioni e i dump testuali del C++ non vengono importati direttamente.
Gli input sparsi, NaN/infinito e `sample_weight` non sono supportati. Per più
target, `MultiOutputRegressor` addestra una popolazione indipendente per target.

Questa è una reimplementazione algoritmica, non una replica bit-per-bit delle
traiettorie casuali C++. Le correzioni e le differenze sono descritte in
[docs/algorithm.md](docs/algorithm.md), con una mappa verso i sorgenti e i paper.
La firma e la documentazione completa dei parametri sono disponibili con
`help(XCSFRegressor)`.

## Esempi e verifica

```bash
python examples/sine_regression.py
python examples/benchmark.py --output tmp/benchmark.json
python examples/prediction_comparison.py --output tmp/prediction-benchmark.json
python examples/training_diagnostics.py --output tmp/training_history.png
python -m pytest -q
```

I test verificano gli aggiornamenti numerici contro calcoli indipendenti:
RLS contro minimi quadrati regolarizzati, la ricorsione di Kalman in tutte le
modalità e un riferimento in aritmetica esatta; Lasso online e batch contro
scikit-learn e le condizioni di ottimalità. Verificano inoltre ambienti, ciclo
dell'esperimento, invarianti della popolazione, ciclo evolutivo e i controlli
ufficiali scikit-learn su NLMS, RLS e Lasso online. Il benchmark separa training e test e riporta più seed; non sostituisce
un confronto statistico con la libreria C++ originale.

Risultati dell'esecuzione e ambiente utilizzato: [docs/validation.md](docs/validation.md).
