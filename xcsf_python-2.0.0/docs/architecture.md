# Architettura del pacchetto

Dalla versione 2.1 il pacchetto è organizzato come xcslib: condizioni, azioni,
ambienti, funzioni di predizione, classificatore, sistema ed esperimento sono
componenti separati, ciascuno con un'interfaccia astratta e una o più
implementazioni. Il sistema a classificatori conosce solo le interfacce.

## Mappa dei moduli

| `src/xcsf/` | corrispondente in xcslib | contenuto |
|---|---|---|
| `conditions/base.py` | `conditions/condition_base.h` | `Condition` (una regione), `ConditionRepresentation` (covering e operatori genetici con i loro parametri), `PopulationMatcher` |
| `conditions/real_interval.py` | `conditions/real_interval_condition` | `IntervalCondition`, `RealIntervalRepresentation`, `IntervalMatcher` (matching vettorizzato) |
| `actions/` | `actions/action_base.h`, `dummy_action` | `Action`, `DummyAction` (l'unica azione implicita della regressione) |
| `environments/base.py` | `environments/environment_base.h` | `Environment`: problemi a passo singolo, stato + ricompensa |
| `environments/dataset.py` | – | `DatasetEnvironment`: le righe di un dataset, in ordine o rimescolate a ogni passata |
| `environments/real_functions.py` | `environments/real_functions_env` | `RealFunctionEnvironment`: campionamento uniforme di una funzione reale |
| `prediction/base.py` | `xcsf/pf/base` | `LocalPredictor` (interfaccia), `design_matrix` |
| `prediction/constant.py` | `pf/value` | `ConstantPredictor` |
| `prediction/lms.py` | `pf/nlms` | `LMSPredictor`, `NLMSPredictor` |
| `prediction/information.py` | – | nucleo numerico: filtro a radice quadrata dell'informazione |
| `prediction/rls.py` | `pf/rls`, `pf/rlsk`, `pf/rls_delta` | `RLSPredictor` (unica implementazione) |
| `prediction/lasso.py` | – | `LassoOnlinePredictor`, `LassoSGDPredictor`, `LassoBatchPredictor` |
| `prediction/__init__.py` | `pf/utility` | registro per nome, `make_predictor`, `PredictorFactory` |
| `classifier.py` | `xcsf/xcsf_classifier` | `Classifier`: condizione, azione, predittore, statistiche |
| `classifier_system.py` | `xcsf/xcsf_classifier_system` | `XCSFClassifierSystem`: matching, covering, update, GA, subsumption, cancellazione |
| `experiments.py` | `experiments/experiment_mgr` | `run_problems` (ciclo ambiente → sistema), `TrainingMonitor` (statistiche a finestre) |
| `parameters.py` | `configuration_manager` (controlli) | validazione degli iperparametri |
| `regressor.py` | – | `XCSFRegressor`: interfaccia scikit-learn |
| `utils.py` | – | stampa della popolazione, grafici della storia |

## Chi parla con chi

```
XCSFRegressor ── valida i parametri, normalizza gli ingressi
      │ crea
      ├── DatasetEnvironment(X, y) ─────────────┐ stato, target
      │                                         ▼
      └── run_problems(system, environment) ── XCSFClassifierSystem.step(x, target)
                                                │
                    ┌───────────────────────────┼──────────────────────────┐
                    ▼                           ▼                          ▼
        ConditionRepresentation          PredictorFactory             Classifier
        cover / mutate / crossover       crea LocalPredictor          statistiche,
        matcher(...)                     .predict / .update           numerosity
```

- `XCSFClassifierSystem` non contiene codice specifico degli intervalli né
  formule di aggiornamento dei pesi: usa `representation.cover/mutate/crossover`,
  `representation.matcher(...)`, `predictor.predict/update/offspring`.
- L'ambiente non conosce il sistema: espone `begin_problem()`, `state()`,
  `perform(action)`, `reward()`. Il ciclo `run_problems` li collega.
- Gli iperparametri restano *piatti* in `XCSFRegressor` (convenzione
  scikit-learn). Ogni classe di predittore dichiara in `parameters` quali
  parametri dell'estimatore la riguardano; `PredictorFactory` seleziona quelli
  e ignora gli altri, come le sezioni `<prediction::...>` di un file confsys.

### Azioni

L'approssimazione di funzioni non richiede decisioni: tutti i classificatori
propongono la stessa azione implicita e l'action set coincide con il match set
(xcslib ottiene lo stesso effetto compilando XCSF con `ACTIONS=dummy_action`).
`Classifier.action` rende esplicita questa struttura: due classificatori sono la
stessa regola solo se coincidono condizione **e** azione.

## Uso senza scikit-learn

Il sistema si può usare direttamente con un ambiente, come in xcslib:

```python
import numpy as np
from xcsf import RealFunctionEnvironment, XCSFClassifierSystem, XCSFRegressor
from xcsf.experiments import run_problems

rng = np.random.RandomState(0)
parameters = XCSFRegressor(prediction="rls", population_size=200).get_params()
system = XCSFClassifierSystem(parameters, rng)
environment = RealFunctionEnvironment(lambda x: np.sin(2 * np.pi * x[0]), [0.0], [1.0], rng)

errors = []
run_problems(system, environment, 5000,
             on_problem=lambda prediction, target: errors.append(abs(target - prediction)))
```

`XCSFRegressor.fit` fa esattamente questo con un `DatasetEnvironment`, più la
normalizzazione degli ingressi e la raccolta della storia di training.

## Estendere

- **Nuovo predittore.** Sottoclasse di `LocalPredictor` con `name`, `parameters`
  (parola chiave del costruttore → parametro di `XCSFRegressor`) e `_update`;
  va aggiunta a `PREDICTORS` in `prediction/__init__.py`. `_update` deve essere
  "tutto o niente": lo stato cambia solo se il calcolo va a buon fine.
- **Nuova condizione.** Sottoclassi di `Condition` e `ConditionRepresentation`;
  il sistema la riceve con `XCSFClassifierSystem(..., representation=...)`.
  Un `matcher` vettorizzato è facoltativo: quello predefinito interroga le
  condizioni una alla volta.
- **Nuovo ambiente.** Sottoclasse di `Environment` con `begin_problem`, `state`
  e `reward`.

## Cosa è cambiato rispetto alla 2.0

| 2.0 | 2.1 |
|---|---|
| `conditions.py`, `rule.py`, `core.py`, `prediction.py` | pacchetti `conditions/`, `actions/`, `environments/`, `prediction/`; `classifier.py`, `classifier_system.py`, `experiments.py`, `parameters.py` |
| `XCSFCore.update(x, y, phi)` | `XCSFClassifierSystem.step(x, target, phi=None)` |
| `LocalPredictor(size, method=...)` | `make_predictor(method, size, ...)`; `LocalPredictor` è la classe base |
| `prediction="rls"` (QR) e `"rlsk"` (covarianza, Joseph) | solo `"rls"`: QR con oblio, rumore di processo e varianza di misura |
| `prediction="lasso_online"` = gradiente prossimale | `"lasso_online"` = Lasso ricorsivo sulle statistiche RLS; il gradiente prossimale è `"lasso_sgd"` |
| `rls_delta > 0` | `rls_delta >= 0` (zero solo per `rls` con `process_noise > 0`) |

Per i predittori invariati (`constant`, `lms`, `nlms`, `lasso_batch`, il vecchio
`rls`, e `lasso_sgd` rispetto al vecchio `lasso_online`) popolazioni, curve di
errore e predizioni sono identiche bit per bit a quelle della 2.0 a parità di
seme; il nuovo `rls` riproduce il vecchio `rlsk` con le stesse popolazioni e
differenze sulle predizioni non superiori a $10^{-10}$. I modelli salvati con pickle dalla
2.0 non sono leggibili dalla 2.1.
