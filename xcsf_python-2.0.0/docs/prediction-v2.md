# Predictor della versione 2.0

La versione 2.0 implementa gli algoritmi di apprendimento in Python/NumPy.
SciPy fornisce solo la risoluzione di sistemi triangolari; nessun estimatore
esterno viene addestrato dentro i classifier. Scikit-learn resta la base dell'API
pubblica e fornisce riferimenti indipendenti nei test.

## Scelta del metodo

```python
from xcsf import XCSFRegressor

model = XCSFRegressor(
    prediction="lasso_batch",  # lms, nlms, rls, lasso_online, lasso_batch
    lasso_alpha=0.001,
    lasso_window=256,          # None: tutti i campioni dalla nascita della regola
    lasso_max_iter=1000,
    lasso_tol=1e-6,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Restano disponibili `constant` e `rlsk`. Il default resta `nlms`.
Tutti i metodi supportano `fit`, `partial_fit`, clone scikit-learn, pickle/joblib,
Pipeline e ricerca degli iperparametri. `partial_fit` conserva anche finestre,
fattorizzazioni e contatori locali. `fit` li reinizializza.

Indicando con `p` il numero di coefficienti (bias incluso), `m` la dimensione
della finestra e `k` le passate del solver:

| Metodo | Aggiornamento | Stato numerico principale | Costo per campione locale |
|---|---|---|---|
| `lms` | Gradiente sull'errore quadratico | O(p) | O(p) |
| `nlms` | Gradiente normalizzato per norma dell'input | O(p) | O(p) |
| `rls` | QR ricorsiva con rotazioni di Givens | O(p²) | O(p²) |
| `lasso_online` | Gradiente più soglia L1 | O(p) | O(p) |
| `lasso_batch` | Coordinate descent sui dati conservati | O(mp) | O(kmp) |

Il costo totale di XCSF dipende anche da quante regole fanno matching. Il solver
batch rifà il fitting a ogni osservazione locale. `lasso_window=None` comporta
memoria e costo crescenti con la storia della singola regola.

## LMS e NLMS

Con `e = y - phi @ w` e `eta = prediction_learning_rate`:

- LMS: `w <- w + eta * e * phi`.
- NLMS: `w <- w + eta * e * phi / (phi @ phi)`.

NLMS calcola la normalizzazione tramite un vettore riscalato per evitare che
il quadrato della norma vada in overflow o underflow; un vettore nullo non cambia
i pesi. LMS è intenzionalmente non normalizzato: un passo adatto a NLMS potrebbe
essere troppo grande per LMS. I controlli numerici segnalano un errore se un
aggiornamento non è rappresentabile; non applicano clipping silenzioso ai pesi.

## RLS con QR

Per un classifier con pesi iniziali `w0`, `delta = rls_delta`, fattore di oblio
`lambda = forgetting_factor` e `t` osservazioni, l'obiettivo è:

```
sum(lambda**(t-i) * (y_i - phi_i @ w)**2, i=1..t)
+ lambda**t / delta * ||w - w0||²
```

L'inizializzazione usa `R = I / sqrt(delta)` e `z = R @ w0`. Ogni aggiornamento
triangularizza la matrice aumentata con rotazioni di Givens:

```
[sqrt(lambda) R | sqrt(lambda) z]
[      phi     |       y       ]
```

Il sistema `R w = z` viene risolto per sostituzione triangolare. Non si formano
le equazioni normali né si aggiorna per sottrazione una matrice inversa.
`predictor.covariance` ricostruisce `R^-1 R^-T` solo per ispezione, con costo
O(p³); la matrice restituita non è uno stato modificabile del predictor.

Con `lambda=1` il problema è quello della precedente RLS, a meno delle differenze
di arrotondamento. Nella 2.0 `forgetting_factor` agisce anche su `rls`; prima era
usato solo da `rlsk`. Il fattore sconta anche il prior iniziale, quindi non va
interpretato come una penalità ridge costante nel tempo quando `lambda < 1`.
Non è una protezione contro gli outlier. Scale ragionevoli restano importanti;
oblio molto forte per molti passi senza informazione in alcune direzioni può
far perdere rango numerico, segnalato come errore.

`rlsk` mantiene oblio, rumore di processo e varianza adattiva del port precedente,
con aggiornamento della covarianza nella forma di Joseph:
`(I-K phi) P_prior (I-K phi).T + noise*K*K.T + Q*I`.
Questa variante resta separata dalla QR RLS e ha costo O(p³) nell'implementazione
densa. `process_noise` e `kalman_noise` si applicano solo a `rlsk`.

## Lasso online

Sul campione locale numero `t`, a partire da 1:

```
eta_t = prediction_learning_rate / t**lasso_learning_rate_decay
v = w + eta_t * (y - phi @ w) * phi
w[0] = v[0]
w[1:] = sign(v[1:]) * maximum(abs(v[1:]) - eta_t * lasso_alpha, 0)
```

È un passo di gradiente prossimale stocastico per errore quadratico / 2 più
penalità L1. Può produrre coefficienti esattamente nulli. Non garantisce la
soluzione Lasso batch dopo ogni campione e non ha un certificato di convergenza
batch. Il default `lasso_learning_rate_decay=0` usa un passo costante, utile
per adattarsi continuamente. Un valore positivo (per esempio 0.6) riduce il
passo con l'esperienza locale; convergenza e precisione dipendono anche dal flusso
e dal passo iniziale. Il conteggio avanza solo quando la regola viene aggiornata.

Come LMS, richiede feature ben scalate e un passo adeguato. `lasso_window`,
`lasso_max_iter` e `lasso_tol` non si applicano al metodo online.

## Lasso batch locale

L'obiettivo sui `m` campioni attualmente conservati è:

```
||y - Phi @ w||² / (2*m) + lasso_alpha * sum(abs(w[1:]))
```

Il bias non è penalizzato. Il solver centra input e target della finestra,
aggiorna ciclicamente le coordinate con soft threshold e ricostruisce
l'intercetta. Usa i dati effettivi e residui ricalcolati, evitando la matrice di
Gram. Parte dai pesi precedenti (warm start). Le colonne costanti hanno pendenza
zero; con un solo campione la predizione è il suo target e le pendenze sono nulle.
`lasso_alpha=0` elimina la penalità, mantenendo lo stesso solver iterativo.

La convergenza richiede che la massima violazione delle condizioni KKT sulle
pendenze sia <= `lasso_tol`, una tolleranza **assoluta nelle unità del gradiente**.
L'intercetta è ottimizzata analiticamente. Non è lo stesso criterio di arresto
usato da `sklearn.Lasso`. Se il budget `lasso_max_iter` si esaurisce, si conserva
la soluzione corrente e si emette `sklearn.exceptions.ConvergenceWarning` una
volta per classifier; lo stato di convergenza viene aggiornato a ogni fitting.
Con feature fortemente correlate potrebbe servire aumentare il budget.

```python
for rule in model.get_rules():
    print(rule["id"], rule["prediction_diagnostics"])
# n_updates: aggiornamenti del predictor dalla nascita
# n_samples: osservazioni conservate (0 per metodi senza buffer)
# n_iter: passate dell'ultimo fitting batch
# converged: True/False per un batch già addestrato, altrimenti None
# kkt_violation: massima violazione dell'ultimo batch, altrimenti None
```

La finestra è FIFO per singola regola. Solo i campioni che la aggiornano entrano
nella sua storia; la predizione non aggiunge dati. Più epoche ripresentano i dati
e quindi contano come ulteriori osservazioni. Una finestra non equivale a tutti
i campioni del dataset contenuti nella condizione: conta la storia effettiva
degli aggiornamenti dalla nascita del classifier.

## Scala, intercetta ed eredità

Il vettore locale resta `[x0, z1, ..., zd, z1**2, ...]`, senza interazioni, e
`w[0] * x0` rappresenta l'intercetta. L1 penalizza soltanto gli altri coefficienti.
`lasso_alpha` è distinto da `alpha`, che controlla la fitness XCSF. La scala delle
feature cambia il significato della penalità: la trasformazione min-max esistente
resta congelata dopo il primo fit/batch. Non vengono aggiunti scaler dinamici
che cambierebbero il significato dei pesi ereditati.

Tutti i figli ereditano una copia dei pesi e del valore costante, ma ripartono con
contatori a zero, fattorizzazione/covarianza iniziale e buffer Lasso vuoto.
Il primo update RLS usa i pesi ereditati come prior; il primo fitting Lasso batch
ottimizza sui soli nuovi dati e può quindi cambiarli drasticamente. La copia
non condivide array o buffer con il genitore. L'ereditarietà non conserva campioni
raccolti nella regione del genitore prima della mutazione.

## Compatibilità e verifiche

La distribuzione e `xcsf.__version__` passano a `2.0.0`. Le vecchie scelte
`nlms`, `rls`, `rlsk`, `constant` restano valide. `get_rules()` aggiunge
`prediction_diagnostics`. Le traiettorie numeriche ed evolutive possono cambiare.
I pickle/joblib della versione precedente non hanno uno schema di migrazione:
conservare l'ambiente 1.0 per leggerli oppure riaddestrare con la 2.0. I nuovi
salvataggi conservano tutto lo stato necessario per riprendere l'apprendimento.

I test confrontano QR RLS con least squares aumentati risolti mediante SVD,
e Lasso batch con `sklearn.Lasso`, includendo finestre FIFO, feature costanti e
collineari, bias non unitario e penalità nulla. Verificano inoltre l'aggiornamento
prossimale, le copie dei figli, il fitting incrementale, la serializzazione,
l'integrazione evolutiva e la regressione dei comportamenti precedenti.

Eseguire:

```bash
python -m pytest tests xcsf_duplication/tests -q --tb=short -rs
python examples/prediction_comparison.py --output docs/prediction-benchmark-v2.json
```
