# Algoritmo, corrispondenza con xcslib e scelte del port

La versione 2.0 sostituisce gli aggiornamenti dei predictor e aggiunge LMS e due
varianti Lasso. Vedere [prediction-v2.md](prediction-v2.md) per le formule
implementate, i parametri e le differenze di compatibilità.

## Riferimenti utilizzati

1. **Wilson (2002), Classifiers that approximate functions**:
   [`A_1016535925043.pdf`](../references/A_1016535925043.pdf).
   Introduce XCSF e la coevoluzione di regioni e approssimatori locali.
2. **Butz e Wilson (2002), An algorithmic description of XCS**:
   [`s005000100111.pdf`](../references/s005000100111.pdf).
   Aggiornamento della fitness, GA, subsumption, cancellazione e numerosity.
3. **Lanzi, Loiacono, Wilson e Goldberg (2005), Generalization in the XCSF
   Classifier System: Analysis, Improvement, and Extension**, IlliGAL 2005012:
   [`XCSF_Generalization.pdf`](../references/XCSF_Generalization.pdf).
   In particolare §7.3 e Algoritmo 5 per RLS e l'inizializzazione V = delta I.
4. **Lanzi e Loiacono (2025), A Proposal for a Leaner Narrative of Learning
   Classifier Systems**, GECCO Companion:
   [`3712255.3735661.pdf`](../references/3712255.3735661.pdf), §5.1–5.2 e §6.2,
   Algoritmi 4 e 7. Separa inizializzazione, query e aggiornamento, e riconduce
   la regressione a una singola azione implicita con target fornito dall'esterno.

L'API segue le [convenzioni ufficiali per estimatori scikit-learn](https://scikit-learn.org/stable/developers/develop.html).

## Ciclo di apprendimento

Per ogni coppia `(x, y)`:

1. Incrementa il tempo e costruisce la nicchia delle regole che contengono `x`.
2. Se la nicchia è vuota, crea una condizione con estremi
   `x_i - U(0, cover_radius)` e `x_i + U(0, cover_radius)` e un predittore iniziale.
3. Calcola la predizione precedente all'aggiornamento per la cronologia di training.
4. Incrementa esperienza; aggiorna errore, predittore e dimensione della nicchia.
5. Aggiorna la fitness dalla precisione relativa, includendo numerosity.
6. Applica l'eventuale subsumption del match set.
7. Se l'età media pesata per numerosity è almeno `theta_ga`, esegue il GA nella
   nicchia: selezione, copia, crossover, mutazione, subsumption/inserimento.
8. Ripristina la capacità massima cancellando microclassificatori secondo il
   voto di cancellazione. In condensazione, riproduce solo regole esistenti.

La predizione è `sum(F_i * p_i(x)) / sum(F_i)` sulle macroregole della nicchia.
`F_i` incorpora già la numerosity: moltiplicarla nuovamente per `n_i` darebbe
un'aggregazione diversa dal sorgente.

Con errore assoluto stimato `e_i`, l'accuratezza è `k_i = 1` per `e_i < epsilon_0`,
altrimenti `alpha * (e_i / epsilon_0)**(-nu)`. L'obiettivo di fitness è
`k_i * n_i / sum(k_j * n_j)` e il suo aggiornamento usa `learning_rate` (beta).
Il calcolo usa logaritmi per evitare underflow comune a tutti gli `k_i`.
Errore e dimensione della nicchia usano `max(beta, 1/experience)` quando MAM è
attivo, altrimenti beta. L'errore viene per default calcolato prima di aggiornare
il predittore. L'errore quadratico segue lo stesso schema per la variante Kalman.

Il voto di cancellazione è `set_size * numerosity`. Se la regola ha esperienza
oltre `theta_delete` e `F_i/n_i < delta * mean_micro_fitness`, il voto è
moltiplicato per `mean_micro_fitness / (F_i/n_i)`.

## Mappa dei sorgenti

I percorsi C++ nella tabella sono relativi a `xcslib-1.5-rc1-niches/`.

| Sorgente C++ | Implementazione Python | Contenuto |
|---|---|---|
| `src/conditions/real_interval_condition.cpp` | `src/xcsf/conditions.py` | Covering, matching, contenimento, crossover, mutazione |
| `src/xcsf/xcsf_classifier.cpp` | `src/xcsf/rule.py` | Stato, numerosity, copia, esperienza e cronologia delle nicchie |
| `src/xcsf/xcsf_classifier_system.cpp` | `src/xcsf/core.py` | Update, fitness, GA, selezione, subsumption, cancellazione, condensazione |
| `src/pf/base.cpp` | `src/xcsf/prediction.py:design_matrix` | Basi polinomiali senza interazioni |
| `src/pf/nlms.cpp` | `src/xcsf/prediction.py:LocalPredictor` | NLMS: `w += eta * (y - phi@w) * phi / (phi@phi)` |
| `src/pf/value.cpp` | stesso modulo, `constant` | Predizione costante aggiornata con passo eta |
| `src/pf/rls.cpp`, `src/pf/rlsk.cpp` | stesso modulo, `rls` e `rlsk` | Minimi quadrati ricorsivi e variante con covarianza/forgetting |
| Gestione degli esperimenti / ambiente | `src/xcsf/regressor.py` | Sostituita da fit su dataset e aggiornamento incrementale |

### Parametri delle configurazioni fornite

| Nome nel C++ | Parametro Python |
|---|---|
| population size | population_size |
| learning rate, classifier_system | learning_rate |
| learning rate, prediction::nlms | prediction_learning_rate |
| epsilon zero | epsilon_0 |
| vi | nu |
| theta GA | theta_ga |
| crossover probability / mutation probability | crossover_probability / mutation_probability |
| r0 / m0 | cover_radius / mutation_scale |
| theta delete | theta_delete |
| theta GA sub / theta AS sub | theta_subsume / theta_match_subsume |
| GA subsumption e GA subsumption on [A] | ga_subsumption (genitori, poi match set) |
| AS subsumption | match_subsumption |
| offspring selection for GA | selection e tournament_fraction |
| prediction function = value | prediction="constant" |
| prediction function / degree / x0 | prediction / degree / x0 |
| delta, prediction::rlsk | rls_delta |
| lambda / Q / kalman | forgetting_factor / process_noise / kalman_noise |
| number of condensation problems | condensation_epochs × numero di campioni |
| niche queue max size | niche_history |

Per una configurazione vicina a `examples/confsys.sin`, usare i default
`population_size=400`, NLMS, beta=eta=0.2, `theta_ga=25`, crossover=0.8,
mutazione=0.04, `epsilon_0=0.05`, r0=m0=0.2, GA subsumption attiva;
`degree=2` corrisponde a `confsys.sin_q`. Impostare `normalize=False` per dati
già generati nel dominio `[0, 1]` e scegliere `n_epochs` in funzione del numero
di aggiornamenti desiderato. La libreria non interpreta i file confsys.

## Predittori RLS e RLSK

`rls` nella 2.0 usa una fattorizzazione QR. Con `forgetting_factor=1`
risolve lo stesso problema della ricorrenza del paper del 2005:

```
g = V @ phi / (1 + phi.T @ V @ phi)
w = w + g * (y - phi.T @ w)
V = V - outer(g, phi.T @ V)
```

`rlsk` usa la forma di Joseph, algebricamente equivalente alle formule
seguenti in aritmetica esatta. Imposta `V_prior = V / forgetting_factor` e la varianza di misura
`R = max(squared_error, 1e-4)` se `kalman_noise=True`, oppure R=1:

```
g = V_prior @ phi / (R + phi.T @ V_prior @ phi)
w = w + g * (y - phi.T @ w)
V = V_prior - outer(g, phi.T @ V_prior) + process_noise * I
```

Con lambda=1, Q=0 e `kalman_noise=False`, RLSK coincide con RLS. `rls` usa anch'esso `forgetting_factor` dalla 2.0, ma ignora
`process_noise` e `kalman_noise`. I nuovi classificatori hanno V=`rls_delta * I` e
pesi nulli salvo l'intercetta impostata da `initial_prediction`. I figli
ereditano i pesi dei genitori, senza ricombinarli, e inizializzano nuovamente V,
come nei metodi `clone` forniti. Questa scelta consente al predittore di adattarsi
alla nuova regione dopo la mutazione.

## Correzioni e differenze intenzionali

Il codice fornito contiene componenti storiche incompiute o incongruenti. Non
viene promessa identità bit-per-bit: NumPy usa un generatore e un ordine di
campionamento diversi e le popolazioni possono divergere rapidamente.

- **Mutazione degli estremi**: `fixed_mutation` e `gaussian_mutation` assegnano
  `lower` anche all'estremo superiore dopo aver calcolato `upper`. Qui l'estremo
  superiore viene aggiornato indipendentemente. Gaussian è selezionabile anche
  nel dispatcher; nel C++ non lo è. Dopo tutti gli operatori si ripristinano
  estremi ordinati e si applicano i limiti solo se `bounded=True`.
- **Intervalli chiusi**: il C++ usa `(lower, upper]` con un caso speciale per
  il minimo di dominio. Qui si usa `[lower, upper]`, così covering e matching
  funzionano anche per feature costanti e condizioni degeneri.
- **RLS**: nel file C++ `rls.cpp`, delta non viene letto dalla configurazione e
  l'update aggiunge sempre I. La variante Python `rls` segue il paper, senza
  rumore di processo; `rlsk(process_noise=1)` consente l'aggiunta di I.
- **RLSK**: nel ramo Q il C++ scala V per Q e poi aggiunge I, non Q*I come
  indicano i commenti. Qui si aggiunge Q*I. `x0` è inizializzato esplicitamente;
  la squared error necessaria al Kalman è effettivamente aggiornata, mentre
  `qerror` non risulta aggiornata nel ciclo del sistema fornito.
- **Predittore costante**: il costruttore C++ non inizializza esplicitamente
  il valore della predizione; qui `initial_prediction` lo definisce. Si mantiene
  il passo costante eta del suo update, senza introdurre MAM sul predittore.
- **Covering con popolazione piena**: si libera un posto prima di inserire la
  nuova regola. Evita cicli di covering/cancellazione senza limite e garantisce
  un classificatore per il campione corrente anche con capacità uno.
- **Subsumption del match set**: è implementata e invocata se richiesta. Nel
  C++ la procedura è presente, ma non viene chiamata dal ciclo di aggiornamento.
- **Stato per istanza**: nessun parametro statico condiviso tra estimatori.
  Timestamp di creazione e cronologia appartengono alla regola: la fusione di
  un duplicato ne aumenta la numerosity senza cancellarne la cronologia.
- **Nessun covering in predict**: a differenza di `ActionValues` nell'Algoritmo
  4 del paper 2025, le query restano pure per l'uso scikit-learn. Il covering si
  trova nell'update; le politiche `unmatched` definiscono esplicitamente i buchi.
- **Normalizzazione**: trasformazione globale opzionale e congelata dal training;
  non implementa la normalizzazione locale proposta tra le varianti del paper
  2005 (il metodo `normalize` delle condizioni C++ contiene `assert(false)`).

## Ambito e limiti

Il pacchetto riproduce la componente XCSF di approssimazione per input reali:
non contiene una policy RL, un simulatore d'ambiente o un classificatore a
etichette discrete. Non porta le condizioni ternarie/code-fragment degli altri
eseguibili, l'experiment manager, o il formato di salvataggio C++.
Non espone opzioni del sorgente prive di effetto nell'update fornito, come
`gradient descent`, né inizializzazioni da file/soluzioni. Le subsumption GA
sui genitori e sul match set sono controllate insieme da `ga_subsumption`;
la condensazione usa la stessa strategia di selezione del GA.

Costo indicativo per osservazione: O(P*d) per matching; per ciascuna regola
attiva O(d*degree) per NLMS e O((d*degree+1)^2) per RLS. In molte dimensioni,
regioni iniziali piccole possono coprire pochissimi esempi: capacità, raggio,
numero di epoche e scala dei dati incidono molto sui risultati. Nessuna garanzia
di accuratezza deriva dai soli benchmark sintetici.

