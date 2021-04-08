# [Warsztaty Badawcze](https://github.com/mini-pw/2021L-WarsztatyBadawcze) - Machine Learning (ML)

Semestr Letni 2020/21 [@hbaniecki](https://github.com/hbaniecki)

## [WB-Book](https://mini-pw.github.io/2021L-WB-Book) / [GitHub](https://github.com/mini-pw/2021L-WB-Book)

## Opis

Przedmiot ten poświęcony jest tematyce badań naukowych. Na zajęciach skupimy się na odtwarzaniu wyników artykułów naukowych, tworzeniu nowych wyników, oraz napisaniu mini-artykułu. Tematem przewodnim projektu jest wykorzystanie modeli uczenia maszynowego do predykcji śmiertelności COVID-19. Poboczne tematy: wyjaśnianie modeli uczenia maszynowego, odporność modeli predykcyjnych i związana z nią odpowiedzialność, reprodukowalność wyników naukowych, pierwsze kroki w świecie badań naukowych.

## Projekt

Całość przedmiotu realizowana jest w zespołach 3/4-osobowych. 

### Faza 1

Pierwsza faza projektu polega na przeanalizowaniu artykułu naukowego, odtworzeniu jego wyników, a następnie zaproponowaniu nowych wyników np. przeprowadzenie analizy danych, stworzenie nowych modeli, wyjaśnienie modeli, przeanalizowanie odpowiedzi do analizowanego artykułu (odtworzenie ich wyników, wykorzystanie dodatkowych danych itp.). Całość powinna zostać udokumentowana w postaci kodu na GitHub oraz raportu - oceniane w dwóch krokach (<b>10 + 20 punktów</b>).

Artykuł do analizy - state-of-the-art (SOTA) w tematyce ML dla COVID-19:

> Yan, L. et al. "An interpretable mortality prediction model for COVID-19 patients" Nature Machine Intelligence (2020) https://www.nature.com/articles/s42256-020-0180-7

Odpowiedzi do powyższego artykułu (Matters Arising):

- [External validation demonstrates limited clinical utility of the interpretable mortality prediction model for patients with COVID-19](https://www.nature.com/articles/s42256-020-00254-2)
- [Replication of a mortality prediction model in Dutch patients with COVID-19](https://www.nature.com/articles/s42256-020-00253-3)
- [Limited applicability of a COVID-19 specific mortality prediction rule to the intensive care setting](https://www.nature.com/articles/s42256-020-00252-4)

### Faza 2

*(08/04): W drugiej fazie można kontynuować dotychczasowe analizy po zaplanowaniu dalszych prac z prowadzącym (15/04).*

Druga faza projektu polega na przeprowadzeniu podobnej analizy, ale:

- Każdy zespół pracuje z innym artykułem - jeżeli artykuł okaże się mało skomplikowany i/lub zespół szybko skończy analizę to można sięgać po pozostałe artykuły z puli.
- Część projektu polega na porównaniu wyników/odniesieniu się do artykułu z **fazy 1**.
- Całość powinna zostać udokumentowana w postaci mini-artykułu (razem z kodami na GitHub), w którym oprócz opisania wyników **fazy 2**, można opisać wyniki **fazy 1** - oceniane w dwóch krokach (<b>10 + 40 punktów</b>).
- Zespół prezentuje najciekawsze wyniki projektu na koniec semestru (<b>20 punktów</b>).

Artykuły do analizy:

- [Deep learning prediction of likelihood of ICU admission and mortality in COVID-19 patients using clinical variables](https://peerj.com/articles/10337)
- [A Learning-Based Model to Evaluate Hospitalization Priority in COVID-19 Pandemics](https://www.cell.com/patterns/fulltext/S2666-3899(20)30120-3)
- [Predicting Mortality Due to SARS-CoV-2: A Mechanistic Score Relating Obesity and Diabetes to COVID-19 Outcomes in Mexico](https://academic.oup.com/jcem/article/105/8/2752/5849337)
- inny w tej samej tematyce (z danymi)

## Spotkania

<table>
<thead>
  <tr>
    <th>nr</th>
    <th>data</th>
    <th>plan</th>
    <th>plan+</th>
    <th>deadline</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>1</td>
    <td>26/02</td>
    <td>wprowadzenie do przedmiotu</td>
    <td>uczenie maszynowe (gradient descent, decision tree, gradient boosting)</td>
    <td></td>
  </tr>
  <tr>
    <td>2</td>
    <td>04/03</td>
    <td><b>faza 1</b> zadanie artykułu do analizy</td>
    <td>uczenie maszynowe dla COVID-19 / typy artykułów na przykładach związanych z ML dla COVID-19</td>
    <td></td>
  </tr>
  <tr>
    <td>3</td>
    <td>11/03</td>
    <td colspan="2">wyjaśnialne uczenie maszynowe (Break Down, Feature Importance, Ceteris Paribus, Partial Dependence Plot)</td>
    <td></td>
  </tr>
  <tr>
    <td>4</td>
    <td>18/03</td>
    <td colspan="2"><b>faza 1</b> omówienie wstępnych wyników</td>
    <td>raport v1 (<b>10 pkt</b>)</td>
  </tr>
  <tr>
    <td>5</td>
    <td>25/03</td>
    <td colspan="2">SHAP, Responsible AI (fairness, h-c xai, security)</td>
    <td></td>
  </tr>
  <tr>
    <td>6</td>
    <td>01/04</td>
    <td>istotne artykuły z dziedziny ML dla COVID-19</td>
    <td>zaawansowane metody eksploracji modeli (modelStudio, Arena)</td>
    <td></td>
  </tr>
  <tr>
    <td>7</td>
    <td>08/04</td>
    <td colspan="2"><b>faza 1</b> przedstawienie wyników</td>
    <td>raport v2 (<b>20 pkt</b>)</td>
  </tr>
  <tr>
    <td>8</td>
    <td>15/04</td>
    <td><b>faza 2</b> konsultacje - zaplanowanie dalszych prac</td>
    <td>TabNet, AUPRC / gość</td>
    <td></td>
  </tr>
  <tr>
    <td>9</td>
    <td>22/04</td>
    <td><b>faza 2</b> przedstawienie koncepcji / postępów</td>
    <td>TabNet, AUPRC / gość</td>
    <td></td>
  </tr>
  <tr>
    <td>10</td>
    <td>29/04</td>
    <td colspan="2">Automated ML / konsultacje projektów</td>
    <td></td>
  </tr>
  <tr>
    <td>11</td>
    <td>06/05</td>
    <td colspan="2"><b>faza 2</b> przedstawienie wstępnych wyników</td>
    <td>projekt v1 (<b>10 pkt</b>)</td>
  </tr>
  <tr>
    <td>12</td>
    <td>12/05</td>
    <td colspan="2">*</td>
    <td></td>
  </tr>
  <tr>
    <td>13</td>
    <td>13/05</td>
    <td colspan="2">wskazówki do wyników projektu: mini-artykuł i prezentacja</td>
    <td></td>
  </tr>
  <tr>
    <td>14</td>
    <td>20/05</td>
    <td colspan="2"><b>faza 2</b> omówienie końcowych wyników</td>
    <td>projekt v2</td>
  </tr>
  <tr>
    <td></td>
    <td>27/05</td>
    <td colspan="2">prezentacje projektów na wykładzie w czasie laboratoriów</td>
    <td>prezentacja (<b>20 pkt</b>)</td>
  </tr>
  <tr>
    <td>12'</td>
    <td>28/05</td>
    <td colspan="2">*</td>
    <td></td>
  </tr>
  <tr>
    <td></td>
    <td>05/06</td>
    <td colspan="2">deadline oddania projektu</td>
    <td>projekt v3 (<b>40 pkt</b>)</td>
  </tr>
</tbody>
</table>


## Bibliografia

> Baniecki, H., and Biecek, P. "Responsible Prediction Making of COVID-19 Mortality (Student Abstract)" Proceedings of the AAAI Conference on Artificial Intelligence (2021) https://www.aaai.org/AAAI21Papers/SA-299.BanieckiH.pdf

> Wynants, L. et al. "Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal" BMJ (2020) https://www.bmj.com/content/369/bmj.m1328

> Barredo Arrieta, A. et al. "Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI" Information Fusion (2020) https://www.sciencedirect.com/science/article/pii/S1566253519308103

## Materiały

- wyjaśnianie (predykcyjnych) modeli uczenia maszynowego: [Explanatory Model Analysis](https://ema.drwhy.ai), [dalex](https://dalex.drwhy.ai/), [Arena](https://arena.drwhy.ai/docs/), [modelStudio](https://modelstudio.drwhy.ai/) 
- bazy preprintów: [arXiv](https://arxiv.org/), ['arXiv for med'](https://medrxiv.org/), ['arXiv for bio'](https://biorxiv.org)
- bazy konferencji/czasopism: [CORE ranking](https://www.core.edu.au/conference-portal), [Scopus](https://www.scopus.com/)
- baza [papers with code](https://paperswithcode.com/)
- [konferencje AI/ML](https://jackietseng.github.io/conference_call_for_paper/conferences.html) 
- [wykaz czasopism i konferencji punktowanych](https://www.gov.pl/web/edukacja-i-nauka/nowy-rozszerzony-wykaz-czasopism-naukowych-i-recenzowanych-materialow-z-konferencji-miedzynarodowych) przez MEN
- personal website/cv: [hugo](https://themes.gohugo.io/tags/personal), [al-folio](https://github.com/alshedivat/al-folio)
- AutoML: [PyCaret](https://pycaret.org/automl/), [Microsoft Azure](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)

<details>
<summary><strong><em>warto wspomnieć</em></strong></summary>

* [backpropagation](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* AUPRC
* [TabNet](https://github.com/dreamquark-ai/tabnet) [/ in R](https://github.com/mlverse/tabnet)
* [YK](https://www.youtube.com/c/YannicKilcher)

</details>
