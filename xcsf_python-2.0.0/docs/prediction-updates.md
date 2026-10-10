# Le funzioni di update dei predittori locali: teoria

Questo documento descrive **che cosa calcola** ciascun predittore quando riceve
un'osservazione: quale funzione costo minimizza, con quale metodo, quanto
velocemente converge e perché. Non descrive il codice; i moduli corrispondenti
sono in `src/xcsf/prediction/` (vedere [architecture.md](architecture.md)).

Le formule usano la sintassi matematica di Markdown (`$...$`), visualizzata da
GitHub e dall'anteprima di VS Code.

## 1. Il problema locale

Ogni classificatore possiede un modello lineare del target nella propria regione:

$$\hat y(x) = w^\top \varphi(x), \qquad \varphi(x) = [\,x_0,\ x_1,\dots,x_d,\ x_1^2,\dots,x_d^{g}\,]^\top$$

dove $x_0$ è l'ingresso costante (`x0`), $g$ è `degree` e $p = 1 + d\,g$ è il
numero di pesi. $w_1$ è l'intercetta (moltiplica $x_0$), gli altri pesi sono le
*pendenze*.

Tre aspetti valgono per tutti i predittori:

- **L'update è locale.** Un classificatore viene aggiornato solo dalle
  osservazioni che cadono nella sua condizione. L'indice $t$ conta questi update
  (l'*esperienza* della regola), non i passi globali dell'algoritmo.
- **L'errore usato è quello a priori**: $e_t = y_t - w_{t-1}^\top\varphi_t$, cioè
  l'errore commesso *prima* di imparare dall'osservazione.
- **I figli ereditano i pesi e nient'altro.** Dopo crossover e mutazione la
  regione del figlio è diversa da quella del genitore: i pesi sono un buon punto
  di partenza, le statistiche accumulate (covarianze, campioni, contatori) no.

Tutti i predittori cercano il minimo della stessa superficie, l'errore quadratico
medio nella regione:

$$J(w) = \tfrac12\,\mathbb E\big[(y - w^\top\varphi)^2\big], \qquad
\nabla J(w) = R\,w - b, \qquad R = \mathbb E[\varphi\varphi^\top],\quad b = \mathbb E[\varphi\,y]$$

Il minimo è la soluzione di Wiener $w^\star = R^{-1} b$. I predittori si
distinguono per **quanta informazione su $R$ usano**:

| predittore | problema risolto | metodo | usa $R$? | costo per update | memoria |
|---|---|---|---|---|---|
| `constant` | $\min_v \tfrac12\mathbb E[(y-v)^2]$ | gradiente stocastico | – | $O(1)$ | $O(1)$ |
| `lms` | $\min_w J(w)$ | gradiente stocastico | no | $O(p)$ | $O(p)$ |
| `nlms` | minima perturbazione | gradiente normalizzato | solo $\lVert\varphi\rVert^2$ | $O(p)$ | $O(p)$ |
| `rls` | minimi quadrati pesati esatti | Newton ricorsivo (QR) | sì, tutta | $O(p^2)$ | $O(p^2)$ |
| `lasso_sgd` | $J(w) + \alpha\lVert w_{2:}\rVert_1$ | gradiente prossimale | no | $O(p)$ | $O(p)$ |
| `lasso_online` | Lasso pesato esatto | statistiche RLS + discesa per coordinate | sì, tutta | $O(k\,p^2)$ | $O(p^2)$ |
| `lasso_batch` | Lasso sugli ultimi $m$ campioni | discesa per coordinate | sì (dai dati) | $O(k\,m\,p)$ | $O(m\,p)$ |

($k$ = passate del solver, tipicamente 1–10 grazie alla partenza dalla soluzione
precedente.)

### Perché il condizionamento di $R$ conta

In una regola XCSF gli ingressi sono confinati in un intervallo stretto, quindi
$x_0$ e $x$ sono quasi collineari. Per un ingresso uniforme in $[c-h,\ c+h]$:

$$R = \begin{bmatrix} x_0^2 & x_0 c \\ x_0 c & c^2 + h^2/3 \end{bmatrix},
\qquad \det R = \frac{x_0^2 h^2}{3},
\qquad \kappa(R) \approx \frac{3\,(x_0^2 + c^2 + h^2/3)^2}{x_0^2\,h^2}$$

| situazione | $c$ | $h$ | $\kappa(R)$ |
|---|---|---|---|
| ingresso in $[0,1]$, regola larga 0,12 | 0,36 | 0,06 | $\approx 10^3$ |
| ingresso grezzo in $[1000,1100]$, regola larga 20 | 1050 | 10 | $\approx 4\cdot10^{10}$ |

I metodi del primo ordine (`lms`, `nlms`, `lasso_sgd`) convergono con una
velocità inversamente proporzionale a $\kappa(R)$: sono lenti proprio nelle
regole strette e con ingressi non centrati. I metodi che usano $R$ (`rls`,
`lasso_online`, `lasso_batch`) non ne risentono.

## 2. `constant`: media mobile esponenziale

**Modello.** Un solo valore $v$ per tutta la regione (gli ingressi sono ignorati).

**Update.** Un passo di gradiente stocastico su $\tfrac12 (y-v)^2$:

$$v_t = v_{t-1} + \eta\,(y_t - v_{t-1})$$

**Che cosa calcola.** Svolgendo la ricorsione,
$v_t = (1-\eta)^t v_0 + \eta \sum_{i=1}^{t} (1-\eta)^{t-i} y_i$: una media dei
target con pesi che decadono geometricamente e memoria effettiva di circa
$1/\eta$ campioni. A regime oscilla attorno alla media dei target nella regione
con varianza $\frac{\eta}{2-\eta}\sigma_y^2$: $\eta$ grande segue i cambiamenti ma
è rumoroso, $\eta$ piccolo è stabile ma lento. Come in xcslib (`value.cpp`) il
passo è costante: non c'è la media campionaria iniziale (MAM) usata per errore e
dimensione della nicchia.

**Parametri.** `prediction_learning_rate` ($\eta$), `initial_prediction` ($v_0$).

## 3. `lms`: gradiente stocastico (Widrow–Hoff)

**Update.** Si sostituisce al gradiente $\nabla J = -\mathbb E[e\,\varphi]$ la sua
stima su un solo campione:

$$w_t = w_{t-1} + \eta\, e_t\, \varphi_t$$

**Convergenza.** In media l'errore sui pesi evolve come
$\mathbb E[w_t] - w^\star = (I - \eta R)^t\,(w_0 - w^\star)$. Lungo l'autovettore
$i$ di $R$ l'errore si riduce di un fattore $(1-\eta\lambda_i)$ per update:

- stabilità in media solo se $0 < \eta < 2/\lambda_{\max}(R)$; in pratica serve un
  passo dell'ordine di $1/\operatorname{tr}(R)$, perché deve restare limitata
  anche la varianza dei pesi;
- costante di tempo del modo $i$: $\tau_i \approx 1/(\eta\lambda_i)$ update. Il
  modo più lento richiede circa $\kappa(R)$ volte più update del più veloce;
- a regime i pesi fluttuano attorno a $w^\star$ con un eccesso di errore
  (*misadjustment*) circa $\eta\operatorname{tr}(R)/2$.

**Conseguenza pratica.** Il passo ammissibile dipende dalla scala degli ingressi
($\lambda_{\max}$ cresce con $x^2$): con ingressi non scalati LMS diverge o va
reso lentissimo. Per questo `lms` richiede ingressi normalizzati.

**Parametri.** `prediction_learning_rate` ($\eta$).

## 4. `nlms`: gradiente normalizzato

**Principio di minima perturbazione.** Tra tutti i pesi che spiegano
esattamente la nuova osservazione si sceglie quello più vicino ai pesi correnti:

$$\min_w \lVert w - w_{t-1}\rVert^2 \quad \text{con} \quad w^\top\varphi_t = y_t
\qquad\Longrightarrow\qquad w = w_{t-1} + \frac{e_t\,\varphi_t}{\lVert\varphi_t\rVert^2}$$

Con un passo $\eta$ si percorre solo una frazione di questa correzione:

$$w_t = w_{t-1} + \eta\,\frac{e_t\,\varphi_t}{\lVert\varphi_t\rVert^2},
\qquad \lVert\varphi_t\rVert^2 = x_0^2 + \sum_j \varphi_{t,j}^2$$

**Che cosa garantisce.** L'errore sullo stesso campione dopo l'update è
$(1-\eta)\,e_t$: l'update è stabile per $0<\eta<2$ **qualunque sia la scala
degli ingressi**, e $\eta$ ha un significato diretto (frazione di errore
corretta). È la regola di xcslib (`nlms.cpp`) e il default della libreria.

**Che cosa non risolve.** La direzione resta quella del gradiente: si corregge
solo lungo $\varphi_t$. In una regola stretta tutti i $\varphi_t$ puntano quasi
nella stessa direzione, quindi la componente dei pesi ortogonale (in sostanza la
pendenza) si corregge pochissimo a ogni passo. La velocità resta limitata da
$\kappa(R)$. Con ingressi grezzi grandi e $x_0=1$ l'intercetta riceve una
frazione $x_0^2/\lVert\varphi\rVert^2 \approx 10^{-6}$ della correzione: è il
motivo per cui NLMS degrada sui domini traslati.

**Ruolo di $x_0$.** Stabilisce quanto della correzione va all'intercetta
rispetto alle pendenze. Un ingresso nullo ($\varphi_t = 0$) non produce update.

**Parametri.** `prediction_learning_rate` ($\eta$), `x0`.

## 5. `rls`: minimi quadrati ricorsivi

La libreria ha **una sola** implementazione di RLS, che copre come casi
particolari tutte le varianti di xcslib (§5.6).

### 5.1 Il problema

Dopo $t$ osservazioni i pesi sono il minimo esatto di

$$J_t(w) = \sum_{i=1}^{t} \frac{\lambda^{\,t-i}}{r_i}\,\big(y_i - \varphi_i^\top w\big)^2
\;+\; \frac{\lambda^{\,t}}{\delta}\,\lVert w - w_0\rVert^2$$

- il primo termine è l'errore quadratico su **tutte** le osservazioni della
  regola, scontate dal fattore di oblio $\lambda\in(0,1]$ e pesate dall'inverso
  della varianza di misura $r_i$;
- il secondo è un *prior*: tiene i pesi vicini a quelli iniziali o ereditati
  $w_0$ con una forza $1/\delta$ che svanisce al crescere dei dati.

Non c'è un learning rate: ogni update fornisce la soluzione ottima del problema
con un'osservazione in più.

### 5.2 La ricorsione in forma di covarianza

Chiamando $P_t$ l'inversa della matrice dell'informazione
$\Lambda_t = \sum_i \lambda^{t-i}\varphi_i\varphi_i^\top/r_i + \lambda^t I/\delta$, il
lemma di inversione dà la ricorsione classica:

$$P^- = P_{t-1}/\lambda, \qquad
k_t = \frac{P^-\varphi_t}{r_t + \varphi_t^\top P^-\varphi_t}, \qquad
w_t = w_{t-1} + k_t\,e_t, \qquad
P_t = P^- - k_t\,\varphi_t^\top P^-, \qquad P_0 = \delta I$$

**Interpretazione.** $k_t = P_t\varphi_t/r_t$: rispetto a LMS il passo scalare
$\eta$ è sostituito dalla matrice $P_t \approx (t\,R)^{-1}$. È un passo di
**Newton**: la correzione viene riscalata dall'inversa di $R$, quindi tutte le
direzioni convergono alla stessa velocità, indipendentemente da $\kappa(R)$.
Bastano circa $p$ osservazioni informative per avvicinarsi a $w^\star$; poi il
guadagno decresce come $1/t$ e i pesi diventano la media ottima di tutta la
storia.

### 5.3 Interpretazione come filtro di Kalman

La stessa ricorsione è il filtro di Kalman per il modello

$$w_t = w_{t-1} + v_t,\quad v_t \sim \mathcal N(0,\ qI) \qquad\qquad
y_t = \varphi_t^\top w_t + n_t,\quad n_t \sim \mathcal N(0,\ r_t)$$

in cui i pesi "veri" possono muoversi di un passo casuale di varianza $q$ dopo
ogni osservazione: $P_t \leftarrow P_t + qI$. Questo dà un significato a tutti
i parametri:

| parametro | significato | effetto |
|---|---|---|
| `rls_delta` ($\delta$) | incertezza iniziale sui pesi, $P_0=\delta I$ | grande: i pesi iniziali/ereditati vengono dimenticati alle prime osservazioni; piccolo: vengono difesi a lungo |
| `forgetting_factor` ($\lambda$) | sconto delle osservazioni passate | memoria effettiva $\approx 1/(1-\lambda)$ update; $\lambda=1$ pesa tutta la storia allo stesso modo |
| `process_noise` ($q$) | deriva dei pesi tra due osservazioni | impedisce al guadagno di annullarsi: la regola resta adattiva |
| `kalman_noise` | $r_t = \max(\text{errore quadratico stimato della regola},\ 10^{-4})$ invece di 1 | una regola ancora imprecisa si fida meno di ogni singola osservazione |

Oblio e rumore di processo sono due modi diversi di mantenere la regola
adattiva: $\lambda<1$ gonfia $P$ in proporzione a se stessa, $q>0$ aggiunge
incertezza uguale in tutte le direzioni.

### 5.4 La forma numerica: radice quadrata dell'informazione

La ricorsione del §5.2 sottrae a ogni passo una matrice da $P$. In aritmetica
finita la differenza può perdere simmetria e positività, e l'errore cresce con
$\kappa(R)$: proprio il caso delle regole strette. L'implementazione propaga
invece un fattore triangolare superiore $R_t$ e un vettore $z_t$ tali che

$$\Lambda_t = P_t^{-1} = R_t^\top R_t, \qquad J_t(w) = \lVert R_t\,w - z_t\rVert^2 + \text{costante}$$

Lo stato iniziale è $R_0 = I/\sqrt\delta$, $z_0 = R_0 w_0$. Un'osservazione è una
riga in più nel sistema, riassorbita con una trasformazione **ortogonale** $Q$
(rotazioni di Givens):

$$Q \begin{bmatrix} \sqrt\lambda\,R_{t-1} & \sqrt\lambda\,z_{t-1} \\[2pt]
\varphi_t^\top/\sqrt{r_t} & y_t/\sqrt{r_t} \end{bmatrix}
= \begin{bmatrix} R_t & z_t \\ 0 & \rho_t \end{bmatrix},
\qquad R_t\,w_t = z_t \ \text{(sostituzione all'indietro)}$$

Proprietà:

- **non si forma mai $\Lambda$ né si sottrae nulla**: $R^\top R$ è definita
  positiva per costruzione;
- il condizionamento con cui si lavora è $\kappa(R_t) = \sqrt{\kappa(\Lambda_t)}$:
  si perde la metà delle cifre significative che si perderebbero lavorando
  con $\Lambda$ o con $P$;
- le trasformazioni ortogonali non amplificano gli errori di arrotondamento:
  l'errore non si accumula lungo la storia della regola;
- costo $O(p^2)$ per update, come la forma di covarianza.

In aritmetica esatta il risultato è identico a quello del §5.2. La matrice $P$
è ricostruita solo su richiesta (`predictor.covariance`), per ispezione.

### 5.5 Rumore di processo nella forma dell'informazione

Aggiungere $qI$ a $P$ senza invertire $R$ si fa con l'aggiornamento temporale di
Dyer–McReynolds: si scrivono, nelle incognite $(v, w')$ con $w' = w + v$, le
equazioni $v/\sqrt q = 0$ e $R\,(w' - v) = z$, e si triangolarizza:

$$Q \begin{bmatrix} I/\sqrt q & 0 \\ -R & R \end{bmatrix}
= \begin{bmatrix} \ast & \ast \\ 0 & R' \end{bmatrix}, \qquad
R'^\top R' = (P + qI)^{-1}$$

La stima dei pesi non cambia (il rumore ha media nulla), quindi $z' = R' w$.

### 5.6 Le varianti come casi particolari

| variante | impostazione |
|---|---|
| RLS del paper (Lanzi et al. 2005, Alg. 5; xcslib `rls_delta`) | $\delta>0$, $\lambda=1$, $q=0$, $r=1$ |
| xcslib `rls` ($V_0 = 0$, $V \leftarrow V + I$ a ogni update) | $\delta=0$, $\lambda=1$, $q=1$, $r=1$ |
| RLS con oblio | $\lambda<1$ |
| variante Kalman (`rlsk` di xcslib e della versione 2.0) | $q>0$ e/o `kalman_noise=True` |

**Il caso $\delta = 0$.** Covarianza iniziale nulla significa "pesi noti con
certezza": il guadagno è zero e la prima osservazione viene ignorata; subito
dopo $P = qI$. Per questo $\delta=0$ è ammesso solo con $q>0$ (altrimenti la
regola non imparerebbe mai). È il comportamento di `rls.cpp`, dove $\delta$ non
viene letto dalla configurazione.

**Eredità.** Il figlio riparte da $R_0 = I/\sqrt\delta$ con $w_0$ = pesi del
genitore: i pesi ereditati sono il centro del prior, non dati.

## 6. Lasso: predittori sparsi

### 6.1 Il problema e le sue condizioni di ottimalità

I tre predittori Lasso aggiungono all'errore quadratico una penalità $\ell_1$
sulle pendenze (mai sull'intercetta). Sulle osservazioni che ciascun metodo
considera, con peso totale $n$:

$$\min_w\ \frac{1}{2n}\sum_i \omega_i\,(y_i - \varphi_i^\top w)^2 + \alpha \sum_{j\ge 2} |w_j|,
\qquad n = \sum_i \omega_i$$

La penalità non è derivabile in zero: è questo che produce pesi **esattamente**
nulli. Chiamando $g$ il gradiente della parte quadratica, $w$ è ottimo se e solo
se (condizioni KKT):

$$g_1 = 0, \qquad
g_j = -\alpha\,\operatorname{sign}(w_j)\ \ \text{se } w_j \ne 0, \qquad
|g_j| \le \alpha\ \ \text{se } w_j = 0$$

Lo strumento comune è l'operatore di **soglia morbida**, soluzione del problema
unidimensionale $\min_u \tfrac12 (u-c)^2 + \tau|u|$:

$$S(c,\tau) = \operatorname{sign}(c)\,\max(|c| - \tau,\ 0)$$

**Effetto con una sola pendenza.** Dopo aver eliminato l'intercetta la soluzione è

$$s = \frac{S\big(\operatorname{cov}(x,y),\ \alpha\big)}{\operatorname{var}(x)}
= s_{\text{LS}} - \frac{\alpha\,\operatorname{sign}(s_{\text{LS}})}{\operatorname{var}(x)}
\quad\text{se } |s_{\text{LS}}| > \frac{\alpha}{\operatorname{var}(x)},\ \text{altrimenti } 0$$

Per una regola di semi-ampiezza $h$ con ingressi uniformi $\operatorname{var}(x) = h^2/3$:
la pendenza viene azzerata quando $|s_{\text{LS}}| \le 3\alpha/h^2$. Due
conseguenze da tenere presenti nella scelta di `lasso_alpha`:

- la soglia va letta rispetto alla **scala del target** (la covarianza ha le
  unità di $x\cdot y$): lo stesso $\alpha$ è irrilevante per un target di
  ampiezza 100 e drastico per uno di ampiezza 1;
- la soglia **cresce al restringersi della regola**: le regole strette tendono a
  diventare predittori costanti, quelle larghe conservano le pendenze.

### 6.2 `lasso_sgd`: gradiente prossimale stocastico

**Update.** Un passo LMS seguito dalla soglia morbida sulle pendenze
(*forward–backward*, noto anche come *truncated gradient* o FOBOS):

$$\eta_t = \frac{\eta}{t^{\,\gamma}}, \qquad
v = w_{t-1} + \eta_t\,e_t\,\varphi_t, \qquad
w_{t,1} = v_1, \qquad w_{t,j} = S(v_j,\ \eta_t\alpha)\ \ (j\ge 2)$$

**Convergenza.** È LMS con una contrazione in più: ne eredita tutti i limiti
(§3). La velocità dipende da $\kappa(R)$, il passo ammissibile dalla scala degli
ingressi. Con passo costante ($\gamma=0$) i pesi non convergono alla soluzione
Lasso ma oscillano in un suo intorno, e l'insieme dei pesi nulli cambia da un
update all'altro. Con passo decrescente ($\gamma>0$) le oscillazioni si
smorzano, ma la convergenza rallenta ulteriormente. Non esiste un certificato
di ottimalità.

**Quando usarlo.** Come riferimento del primo ordine a costo $O(p)$. Per
ottenere davvero la soluzione Lasso si usa `lasso_online`.

**Parametri.** `prediction_learning_rate` ($\eta$), `lasso_alpha` ($\alpha$),
`lasso_learning_rate_decay` ($\gamma$).

### 6.3 `lasso_online`: Lasso ricorsivo sulle statistiche di RLS

È il predittore Lasso online consigliato: converge alla velocità di RLS, non di
LMS, e non conserva campioni.

**Il problema.** Dopo $t$ osservazioni, con $n_t = \sum_{i\le t}\lambda^{t-i}$:

$$w_t = \arg\min_w\ \frac12\sum_{i=1}^{t}\lambda^{t-i}\big(y_i-\varphi_i^\top w\big)^2
+ \frac{\lambda^{t}}{2\delta}\lVert w - w_0\rVert^2
+ \alpha\,n_t \sum_{j\ge2}|w_j|$$

Diviso per $n_t$ è l'obiettivo del §6.1 con pesi $\omega_i = \lambda^{t-i}$, più un
prior che svanisce. Con $\lambda=1$ e $\delta$ grande è **il Lasso esatto su tutte
le osservazioni viste dalla regola**.

**Idea 1: le statistiche di RLS sono una compressione senza perdita.** I primi
due termini sono esattamente la funzione costo di RLS (§5.1 con $r=1$), che il
filtro del §5.4 mantiene nella forma $\tfrac12\lVert R_t w - z_t\rVert^2$ +
costante. Quindi il problema su $t$ osservazioni equivale, senza alcuna
approssimazione, a un Lasso su sole $p$ equazioni:

$$w_t = \arg\min_w\ \tfrac12\lVert R_t\,w - z_t\rVert^2 + \alpha\,n_t\sum_{j\ge2}|w_j|$$

Il costo non dipende più da quante osservazioni la regola ha visto, e l'update
delle statistiche è quello $O(p^2)$ di RLS.

**Idea 2: l'intercetta si elimina gratis.** $R_t$ è triangolare e l'intercetta è
la prima incognita, non penalizzata:

$$R_t = \begin{bmatrix} \rho & r^\top \\ 0 & \tilde R \end{bmatrix},\quad
z_t = \begin{bmatrix} \zeta \\ \tilde z \end{bmatrix}
\quad\Longrightarrow\quad
\tfrac12\lVert R_t w - z_t\rVert^2 = \tfrac12\big(\rho\,w_1 + r^\top s - \zeta\big)^2 + \tfrac12\lVert \tilde R\,s - \tilde z\rVert^2$$

Per qualunque scelta delle pendenze $s$ il primo addendo si annulla scegliendo
$w_1 = (\zeta - r^\top s)/\rho$. Le pendenze risolvono quindi un Lasso sul solo
blocco triangolare finale, e l'intercetta segue per sostituzione:

$$s_t = \arg\min_s\ \tfrac12\lVert \tilde R\,s - \tilde z\rVert^2 + \alpha\,n_t\lVert s\rVert_1$$

$\tilde R^\top\tilde R$ è la matrice di dispersione **centrata** degli ingressi e
$\tilde R^\top \tilde z$ la loro covarianza con il target: l'eliminazione equivale
al centraggio che il Lasso batch esegue esplicitamente sui dati.

**Idea 3: discesa per coordinate sul sistema compresso.** Si aggiorna una
pendenza alla volta, tenendo ferme le altre; ogni aggiornamento è una soglia
morbida. Con $\tilde R_j$ la colonna $j$ e $\varrho = \tilde z - \tilde R s$ il
residuo:

$$s_j \leftarrow \frac{S\big(\tilde R_j^\top \varrho + \lVert\tilde R_j\rVert^2 s_j,\ \ \alpha n_t\big)}{\lVert\tilde R_j\rVert^2}$$

Una passata costa $O(p^2)$. Si parte dalle pendenze dell'update precedente:
un'osservazione in più sposta la soluzione di poco, quindi bastano pochissime
passate. Ci si ferma quando la massima violazione delle condizioni KKT, divisa
per $n_t$, scende sotto `lasso_tol` (lo stesso criterio di `lasso_batch`).
Lavorare su $\tilde R$ anziché sulla matrice di Gram $\tilde R^\top\tilde R$ evita
di elevarne al quadrato il condizionamento.

**Casi limite che ne certificano il significato.**

- **Una sola pendenza** (ingresso scalare, `degree=1`): la discesa per coordinate
  termina in una passata con la formula chiusa del §6.1.
- $\alpha = 0$: il problema è quello di RLS e si risolve per sostituzione; i
  pesi coincidono bit per bit con quelli di `rls`.
- $\lambda=1$, $\delta\to\infty$: coincide con `lasso_batch` con
  `lasso_window=None`, senza conservare alcun campione.

**Perché converge più velocemente di `lasso_sgd`.** Non segue il gradiente di
un campione alla volta: a ogni update restituisce il minimo del problema su
tutti i dati visti. La curvatura $R$ è nelle statistiche, quindi la velocità non
dipende da $\kappa(R)$ né dalla scala degli ingressi, e non c'è un learning rate
da calibrare. L'insieme dei pesi nulli è quello della soluzione ottima, non
l'effetto transitorio di una soglia applicata dopo un passo rumoroso.

**Eredità e parametri.** Il figlio riparte da $R_0 = I/\sqrt\delta$, $n_0=0$ e dai
pesi del genitore come centro del prior. Finché le osservazioni non identificano
le pendenze (alla prima osservazione, nel caso scalare) la penalità le porta a
zero, perché il prior è debole: una regola appena nata è un predittore quasi
costante che acquista pendenze man mano che i dati le giustificano. Un
$\delta$ più piccolo conserva più a lungo le pendenze ereditate. Parametri: `lasso_alpha` ($\alpha$),
`rls_delta` ($\delta>0$), `forgetting_factor` ($\lambda$), `lasso_max_iter`,
`lasso_tol`. `process_noise` e `kalman_noise` non si applicano.

### 6.4 `lasso_batch`: Lasso su finestra

**Il problema.** Il Lasso del §6.1 sugli ultimi $m$ = `lasso_window` campioni
della regola, con pesi uniformi (tutta la storia con `lasso_window=None`).

**Metodo.** A ogni update si centra la finestra, si esegue la discesa per
coordinate sulle pendenze partendo dalla soluzione precedente e si ricostruisce
l'intercetta $w_1 = (\bar y - \bar x^\top s)/x_0$. Con $\tilde x_j$ la colonna
centrata $j$ e $\varrho$ il residuo:

$$s_j \leftarrow \frac{S\big(\tfrac1m\tilde x_j^\top\varrho + \tfrac1m\lVert\tilde x_j\rVert^2 s_j,\ \alpha\big)}{\tfrac1m\lVert\tilde x_j\rVert^2}$$

Una passata costa $O(m\,p)$ e richiede di conservare i campioni. Le colonne
costanti hanno pendenza zero. Il criterio di arresto è la violazione KKT.

**Differenza rispetto a `lasso_online`.** Cambia *quali* osservazioni contano:
una finestra scorrevole dimentica bruscamente i campioni più vecchi di $m$;
l'oblio esponenziale li sconta gradualmente. Con finestra illimitata e
$\lambda=1$ i due metodi risolvono lo stesso problema (a meno del prior).
`lasso_batch` non usa i pesi ereditati come prior: il primo fit ottimizza sui
soli dati nuovi.

**Parametri.** `lasso_alpha`, `lasso_window`, `lasso_max_iter`, `lasso_tol`.

## 7. Confronto

### 7.1 Convergenza dentro una regola

Misura riproducibile con `python examples/prediction_comparison.py`: un
predittore appena creato riceve 400 campioni di $\sin(2\pi x)$ con
$x\in[0{,}30,\ 0{,}42]$ (una regola stretta, ingressi scalati in $[0,1]$,
$\kappa(R)\approx10^3$). Errore assoluto medio *prima* di ogni update; il
miglior modello lineare sulla regione ha errore 0,0144. $\eta = 0{,}2$,
$\alpha = 0{,}001$, $\delta = 1000$.

| predittore | update 1–20 | 21–50 | 51–100 | 301–400 |
|---|---:|---:|---:|---:|
| `constant` | 0,2316 | 0,1175 | 0,1308 | 0,1210 |
| `lms` | 0,2292 | 0,1248 | 0,1369 | 0,1191 |
| `nlms` | 0,2355 | 0,1232 | 0,1364 | 0,1197 |
| `lasso_sgd` | 0,2292 | 0,1247 | 0,1364 | 0,1188 |
| `rls` | 0,0698 | 0,0173 | 0,0157 | 0,0135 |
| `lasso_online` | 0,0895 | 0,0303 | 0,0352 | 0,0264 |
| `lasso_batch` (finestra 256) | 0,0782 | 0,0283 | 0,0338 | 0,0263 |

I metodi del primo ordine imparano subito l'intercetta e poi restano al livello
del predittore costante: la pendenza richiederebbe migliaia di update
($\tau \approx 1/(\eta\lambda_{\min}) \approx 5000$). `rls` raggiunge il limite
del modello lineare in poche decine di update. `lasso_online` e `lasso_batch`
sono praticamente equivalenti e restano poco sopra `rls`: la differenza è il bias
introdotto dalla penalità, non un ritardo di convergenza.

### 7.2 Guida alla scelta

| esigenza | predittore |
|---|---|
| costo minimo, regole molto piccole | `constant` |
| modello lineare a costo $O(p)$, ingressi scalati | `nlms` |
| massima accuratezza e convergenza rapida | `rls` |
| modelli locali sparsi, apprendimento online | `lasso_online` |
| modelli sparsi con dimenticanza netta dei dati vecchi | `lasso_batch` |
| riferimento del primo ordine per il Lasso | `lasso_sgd` |

## 8. Riferimenti

- B. Widrow, M. E. Hoff, *Adaptive switching circuits*, IRE WESCON Convention
  Record, 1960 (LMS).
- J. Nagumo, A. Noda, *A learning method for system identification*, IEEE
  Trans. Automatic Control 12(3), 1967 (NLMS).
- S. Haykin, *Adaptive Filter Theory*, Prentice Hall; A. H. Sayed, *Adaptive
  Filters*, Wiley, 2008 (analisi di LMS/NLMS/RLS, algoritmi QR).
- G. J. Bierman, *Factorization Methods for Discrete Sequential Estimation*,
  Academic Press, 1977 (filtro a radice quadrata dell'informazione).
- P. Dyer, S. McReynolds, *Extension of square-root filtering to include process
  noise*, J. Optimization Theory and Applications 3(6), 1969.
- P. L. Lanzi, D. Loiacono, S. W. Wilson, D. E. Goldberg, *Generalization in the
  XCSF classifier system: analysis, improvement, and extension*, IlliGAL Report
  2005012; Evolutionary Computation 15(2), 2007 (RLS in XCSF, Algoritmo 5).
- R. Tibshirani, *Regression shrinkage and selection via the lasso*, JRSS B
  58(1), 1996.
- J. Friedman, T. Hastie, H. Höfling, R. Tibshirani, *Pathwise coordinate
  optimization*, Annals of Applied Statistics 1(2), 2007 (discesa per coordinate).
- D. Angelosante, J. A. Bazerque, G. B. Giannakis, *Online adaptive estimation
  of sparse signals: where RLS meets the $\ell_1$-norm*, IEEE Trans. Signal
  Processing 58(7), 2010 (Lasso ricorsivo con discesa per coordinate online).
- J. Langford, L. Li, T. Zhang, *Sparse online learning via truncated gradient*,
  JMLR 10, 2009; J. Duchi, Y. Singer, *Efficient online and batch learning using
  forward backward splitting*, JMLR 10, 2009 (gradiente prossimale stocastico).
