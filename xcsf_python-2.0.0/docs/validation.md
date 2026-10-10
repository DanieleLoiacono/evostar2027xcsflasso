# Verifica della reimplementazione

## Versione 2.1 — 10 ottobre 2026

Ambiente: Python 3.14.4, NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1. Versione della libreria: 2.1.0.

La 2.1 riorganizza il pacchetto in componenti separati
([architecture.md](architecture.md)), unifica RLS e introduce il Lasso online
ricorsivo ([prediction-updates.md](prediction-updates.md)).

### Suite di test

Comando: `python -m pytest tests -q`.

**344 test superati, 3 saltati.** I tre skip sono i controlli opzionali Array API
di scikit-learn (`SCIPY_ARRAY_API` non impostato), uno per ciascuno dei tre
estimatori sottoposti ai controlli ufficiali (NLMS, RLS, Lasso online).

Che cosa viene verificato per i predittori:

- **RLS**: soluzione dei minimi quadrati regolarizzati (equazioni normali e SVD
  del sistema aumentato, con prior non nullo e oblio); ricorsione di Kalman in
  forma di covarianza in tutte le modalità (oblio, rumore di processo, varianza
  di misura), su pesi e covarianza; caso `rls_delta=0, process_noise=1` contro la
  trascrizione di `rls.cpp` di xcslib; feature grandi quasi collineari;
  ingressi traslati (x ≈ 1000, regola larga 1) contro un riferimento in
  aritmetica razionale esatta, con errore relativo sui pesi ≤ 1e-10.
- **Lasso online**: coincide con `sklearn.linear_model.Lasso` su tutta la storia
  (1 e 4 feature); soddisfa le condizioni KKT dell'obiettivo documentato con
  prior e oblio; con penalità nulla coincide bit per bit con RLS; converge come
  RLS e non come il gradiente prossimale; segnala la mancata convergenza.
- **Lasso batch**: confronto con scikit-learn su finestra e storia completa, bias
  non penalizzato, colonne costanti/duplicate, penalità nulla, KKT.
- **Lasso SGD**: passo prossimale calcolato a mano; con L1 = 0 coincide con LMS.
- Per tutti: eredità dei pesi senza condivisione di stato, impostazioni ereditate,
  campioni non validi che non alterano lo stato, parametri non validi rifiutati.

Per il resto del pacchetto: ambienti (ordine, rimescolamento, dominio continuo),
ciclo dell'esperimento, uso del sistema senza scikit-learn, equivalenza tra
`fit` e il ciclo esplicito su `DatasetEnvironment`, accesso alle condizioni solo
tramite la rappresentazione, oltre ai test già presenti su update, GA,
subsumption, cancellazione, API scikit-learn, pickle e storia di training.

### Confronto con la versione 2.0

Dodici configurazioni (tutti i predittori; operatori di mutazione, crossover e
selezione alternativi; `degree=2`; condizioni limitate; subsumption del match
set; condensazione), ciascuna con ingressi a 1 e 2 dimensioni, eseguite con la
2.0 e con la 2.1 a parità di seme:

| predittore 2.0 → 2.1 | popolazione e statistiche | predizioni e curva di errore |
|---|---|---|
| `constant`, `lms`, `nlms`, `rls`, `lasso_batch` | identiche | identiche bit per bit |
| `lasso_online` → `lasso_sgd` | identiche | identiche bit per bit |
| `rlsk` → `rls` (tre impostazioni: Q = 0; V0 ≈ 0 con Q = 1; oblio + Q + varianza adattiva) | identiche | differenza massima 1,3e-11 |

Il refactoring non ha quindi modificato l'algoritmo evolutivo né i predittori
invariati; il nuovo `rls` riproduce il vecchio `rlsk`.

### Confronto dei predittori isolati

Comando: `python examples/prediction_comparison.py --output docs/prediction-benchmark.json`.

*Problema lineare sparso.* Un solo seed (42), 600 campioni di training e 400 di
test indipendenti, 6 feature uniformi in [-1, 1], tre pendenze realmente non
nulle, rumore gaussiano sigma = 0,05 solo sul training. Cinque passate, 3.000
update per predittore, stesso ordine. Passo 0,03 per `lms` e `lasso_sgd`, 0,2 per
`nlms` e `constant`; L1 = 0,01; `rls_delta` = 1000; finestra batch 256.

| Metodo | RMSE test | Pendenze non nulle | Campioni conservati | Byte degli array persistenti |
|---|---:|---:|---:|---:|
| constant | 1.402797 | 0 | 0 | 56 |
| lms | 0.010192 | 6 | 0 | 56 |
| nlms | 0.015727 | 6 | 0 | 56 |
| rls | 0.003883 | 6 | 0 | 504 |
| lasso_online | 0.028641 | 3 | 0 | 504 |
| lasso_sgd | 0.026740 | 5 | 0 | 56 |
| lasso_batch | 0.025898 | 3 | 256 | 14392 |

`lasso_online` e `lasso_batch` selezionano le tre feature reali; `lasso_sgd` a
passo costante conserva anche piccoli coefficienti residui. `lasso_online` non
conserva campioni: il suo stato è quello di RLS. Il bias dei tre Lasso rispetto
a RLS è quello atteso dalla penalizzazione. Tempi e risultati sono una misura
su questo ambiente con iperparametri non ottimizzati, non una classifica.

*Convergenza dentro una regola.* Tabella e commento in
[prediction-updates.md](prediction-updates.md), §7.1. Dati completi:
[prediction-benchmark.json](prediction-benchmark.json).

## Archivio: versione 2.0 — 1 ottobre 2026

Ambiente: Python 3.14.4, NumPy 2.5.3, SciPy 1.18.1,
scikit-learn 1.9.1. Versione della libreria: 2.0.0.

Comando: `python -m pytest tests xcsf_duplication/tests -q --tb=short -rs`.

**261 test superati, 2 saltati** in 50,40 s. I due skip sono i controlli
opzionali Array API di scikit-learn (`SCIPY_ARRAY_API` non impostato).
Sono inclusi i controlli ufficiali scikit-learn per NLMS e RLS, i test del
protocollo di riproduzione dei paper e i nuovi test dei cinque predictor.
Per LMS e i due Lasso sono verificati anche clone, pickle con ripresa,
equivalenza fit/partial_fit a parità di ordine, lettura senza mutazioni,
evoluzione, apprendimento su dati separati e validazione dei parametri.

QR RLS viene confrontata con una soluzione SVD del sistema aumentato, anche
con prior non nullo, oblio e feature grandi quasi collineari. Lasso batch è
confrontato con scikit-learn su finestra e storia completa; vengono verificati
bias non penalizzato, sparsità, colonne costanti/duplicate, penalità nulla,
KKT e segnalazione della mancata convergenza. Lasso online è verificato contro
un aggiornamento prossimale calcolato indipendentemente e contro LMS con L1=0.

### Confronto dei predictor isolati

Comando (2.0): `python examples/prediction_comparison.py --output docs/prediction-benchmark-v2.json`.

Un solo seed (42), 600 campioni di training e 400 di test indipendenti,
6 feature uniformi in [-1, 1], tre pendenze realmente non nulle, rumore gaussiano
sigma=0,05 solo sul training. Cinque passate, 3.000 update per predictor,
stesso ordine. Il test confronta con la funzione senza rumore.
Passo 0,03; L1=0,01; Lasso batch su 256 osservazioni; online a passo costante.

| Metodo | RMSE test | Secondi training | Pendenze non nulle | Byte degli array persistenti |
|---|---:|---:|---:|---:|
| lms | 0.010192 | 0.012 | 6 | 56 |
| nlms | 0.005920 | 0.019 | 6 | 56 |
| rls | 0.003883 | 0.104 | 6 | 504 |
| lasso_online | 0.026740 | 0.018 | 5 | 56 |
| lasso_batch | 0.025898 | 0.441 | 3 | 14392 |

Il numero di pendenze non nulle usa soglia 1e-8. La memoria riporta soltanto il
contenuto degli array, senza oggetti Python, target conservati o memoria
temporanea del solver. Tempi e risultati sono una misura su questo ambiente,
non una classifica generale: gli iperparametri non sono ottimizzati e il test
isola il predictor dall'evoluzione XCSF. Lasso introduce il bias atteso dalla
penalizzazione; in questa esecuzione il batch seleziona le tre feature reali,
mentre l'online a passo costante conserva anche piccoli coefficienti residui.
Il solver batch finale soddisfa la tolleranza KKT richiesta.

Il file dei dati di questa esecuzione è stato rigenerato con la 2.1 (sezione
precedente); in questa tabella `lasso_online` è il predittore che la 2.1 chiama
`lasso_sgd`.

### Pacchetto distribuibile

Build sdist/wheel riuscita con `python -m build --no-isolation`:
`dist/xcsf_python-2.0.0.tar.gz` e
`dist/xcsf_python-2.0.0-py3-none-any.whl`.
La wheel è stata estratta in una directory temporanea e importata da lì,
verificando versione e training con tutti e cinque i predictor richiesti.

## Archivio: verifica del port precedente

Verifica eseguita il 30 settembre 2026. Ambiente: Python 3.14.4, NumPy 2.5.3, scikit-learn 1.9.1.

### Suite di test

Comando: `python -m pytest -q --tb=short -rs`.

**183 test superati, 2 saltati**. I due skip riguardano i controlli opzionali
Array API di scikit-learn: `SCIPY_ARRAY_API` non era abilitato. Non ci sono
fallimenti attesi o controlli di accuratezza disabilitati.

La suite include i controlli ufficiali `parametrize_with_checks` per il regressore
NLMS con i parametri predefiniti e per RLS, oltre a calcoli numerici indipendenti,
operatori evolutivi, invarianti della popolazione, DataFrame, Pipeline, GridSearchCV,
cross-validation, MultiOutputRegressor, pickle/joblib e apprendimento incrementale.

Le utility esterne `print_population` e `plot_training_history` sono verificate
anche per assenza di modifiche allo stato del modello. Gli errori registrati per
passi sono confrontati con una sequenza di aggiornamenti nota, comprese finestre
che attraversano epoche/batch e la ripresa dopo pickle. La registrazione abilitata
o disabilitata produce le stesse predizioni e statistiche evolutive.

Il plotting è stato verificato con Matplotlib 3.11.2 e backend Agg. L'esempio
`examples/training_diagnostics.py` genera due pannelli su 10.000 aggiornamenti,
con 100 finestre da 100 passi; il PNG è stato controllato visivamente.

### Benchmark su dati separati

Comando: `python examples/benchmark.py --output docs/benchmark-results.json`.

Per ciascuna configurazione: 600 campioni di training, 300 di test indipendenti,
20 epoche, capacità 400, seed 0/1/2, input uniformi in [0, 1], nessun rumore.
Il caso quadratico usa degree=2; gli altri degree=1. La funzione a tratti ha
una discontinuità in x=0.5. RMSE e R² sono medie sui tre seed.

| Funzione | Predittore | R² medio | R² minimo | RMSE medio | Copertura media |
|---|---|---:|---:|---:|---:|
| linear_2d | nlms | 0.993576 | 0.980733 | 0.031245 | 100.0% |
| linear_2d | rls | 1.000000 | 1.000000 | 0.000000 | 100.0% |
| quadratic_2d | nlms | 0.998700 | 0.998060 | 0.024396 | 100.0% |
| quadratic_2d | rls | 1.000000 | 1.000000 | 0.000011 | 100.0% |
| sine_1d | nlms | 0.995457 | 0.994852 | 0.047176 | 100.0% |
| sine_1d | rls | 0.997797 | 0.997681 | 0.032942 | 100.0% |
| piecewise_1d | nlms | 0.973093 | 0.944233 | 0.077949 | 100.0% |
| piecewise_1d | rls | 0.983708 | 0.978211 | 0.062524 | 100.0% |

I risultati per seed e i tempi sono disponibili in [benchmark-results.json](benchmark-results.json).

L’esempio `examples/sine_regression.py`, con 600 esempi di training e 200 di test,
ha ottenuto R²=0.998008 con RLS e copertura del 100% sui campioni di test.

Queste verifiche dimostrano funzionamento e compatibilità nei casi testati,
non equivalenza bit-per-bit con il C++ o prestazioni garantite su dati reali.
Non è stato eseguito un confronto sperimentale diretto con un eseguibile C++.
La compatibilità con versioni diverse da quelle riportate è dichiarata nelle
dipendenze, ma non è stata verificata qui con una matrice di ambienti.

### Packaging

Installazione editable e build di sdist/wheel verificati. La wheel ha tag
`py3-none-any` e contiene soltanto moduli Python della libreria, oltre ai metadati.
