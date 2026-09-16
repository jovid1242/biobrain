# Milestone 1 — план реализации

Статус: план написан **до** загрузки данных (2026-09-16). Всё, что здесь названо «оценкой»,
после выполнения сверяется с измерениями в `docs/MEMORY.md` и `results/`.

## 1. Цель Milestone 1

Найти официальный dataset, скачать минимально необходимые graph-level данные, проверить
целостность, загрузить граф в компактном виде, посчитать реальные neurons / synapses / edges,
определить annotations и regions, измерить RAM, сделать ConnectomeAnalyzer, написать
`RESEARCH.md` и `DATASET.md`. **SNN-симулятор в Milestone 1 не входит.**

## 2. Выбранный dataset (обоснование — `docs/RESEARCH.md`)

**FlyWire FAFB, materialization 783** (одна взрослая самка *D. melanogaster*, весь мозг
включая optic lobes, без VNC), синапсы Buhmann et al. 2021 с фильтром cleft score > 50.

- 783 — последний публичный release FAFB (flywire.ai: «FlyWire's latest public release is version 783»).
- Единственный вариант с публичной загрузкой **без логина**, с DOI и официальными md5 (Zenodo 10676866).
- Именно его описывают peer-reviewed публикации (Dorkenwald 2024, Schlegel 2024) — есть с чем сверять.
- Новее по синапсам: «Princeton synapses» (Yu et al. 2025, bioRxiv, **не** peer-reviewed; Codex по
  умолчанию с июля 2025; требует Google-логин + api_token; лицензия данных не указана). В Milestone 1
  **не используем**; позже — как sensitivity analysis («переживает ли результат смену детектора синапсов»).

## 3. Что скачиваем

Все ссылки закреплены (Zenodo record + md5; GitHub — по SHA коммита + git blob SHA-1).

| # | файл | источник | байт | формат | зачем | обязателен |
|---|---|---|---:|---|---|---|
| 1 | `proofread_root_ids_783.npy` | Zenodo 10676866 | 1,114,168 | NumPy `<u8`, (139255,) | **узлы**: канонический список proofread-нейронов | да |
| 2 | `proofread_connections_783.feather` | Zenodo 10676866 | 852,022,274 | Arrow IPC (ZSTD), 16,847,997 строк: `pre_pt_root_id, post_pt_root_id, neuropil, syn_count, gaba/ach/glut/oct/ser/da_avg` | **рёбра**: строка = (pre, post, neuropil), без порога | да |
| 3 | `per_neuron_neuropil_count_pre_783.feather` | Zenodo 10676866 | 16,853,770 | Arrow IPC, 2,781,037 строк: `pre_pt_root_id, neuropil, count` | выходные синапсы по neuropil **со всеми партнёрами, включая непроверенные фрагменты** → аудит полноты и сверка с официальными итогами (~130 M синапсов) | да (аудит) |
| 4 | `per_neuron_neuropil_count_post_783.feather` | Zenodo 10676866 | 233,843,050 | Arrow IPC, 43,439,994 строк: `post_pt_root_id, neuropil, count` | то же для входов; какая доля входов нейрона видна в proofread-графе | да (аудит) |
| 5 | `Supplemental_file1_neuron_annotations.tsv` @ **v3.1.0** (commit `8587524c…`) | GitHub flyconnectome/flywire_annotations | 31,718,505 | TSV, 31 колонка | **аннотации для анализа** (самая свежая систематическая версия): flow, super_class, cell_class, cell_type, side, top_nt, soma/pos | да |
| 6 | тот же файл @ **v2.1.0** (commit `ebd66db2…`) | GitHub, тот же репозиторий | 27,015,208 | TSV | версия, соответствующая статьям 2024 → **сверка** с опубликованными числами (9 superclasses, 8,453 cell types, 96.4 %) | да (валидация) |

**Итого к загрузке: 1,162,566,975 байт ≈ 1.16 GB.**

### Что сознательно НЕ скачиваем

| файл / продукт | размер | почему нет |
|---|---:|---|
| `flywire_synapses_783.feather` (все 130,054,535 синапсов с координатами и cleft score) | 9,492,998,242 (8.8 GiB) | Для графа достаточно агрегированных `syn_count`. Нужен только для ре-порогования cleft score, пространственных задержек или morphology-анализа — не в Milestone 1. В каталоге помечен `optional` + `large`, качается только с `--allow-large`. |
| Codex CSV (`connections*`, `classification`, `neurons`, …) | 1 MB – 276 MB | Документированный путь требует Google-логин + `api_token`; то же содержимое в аннотациях GitHub/Zenodo. Codex сам предупреждает, что его аннотации — «mix from different sources». |
| Princeton synapses | ~276 MB (no-threshold CSV) | preprint, логин, лицензия данных не указана (см. §2). |
| Skeletons, meshes, EM imagery | ГБ – ~106 TB | не graph-level; для Milestone 1 не нужны. |

## 4. Pipeline

```
biobrain download   → data/raw/…        (resume, retry, лимит размера, md5 / git-blob проверка)
biobrain preprocess → data/processed/…  (агрегация, CSR/CSC, коды аннотаций, manifest.json со sha256)
biobrain validate   → results/data_validation.{json,md}
biobrain analyze    → results/analysis/…
biobrain bench-memory → results/memory/… + docs/MEMORY.md
```

- Каталог файлов: `catalog/flywire_fafb_v783.toml` (URL, байты, официальные checksums, назначение, флаги).
- После загрузки sha256 каждого файла пишется в `catalog/flywire_fafb_v783.lock.json` (в git) —
  повторная загрузка через полгода проверяется и по официальному md5, и по нашему sha256.
- Секретов для Milestone 1 не требуется. `.env.example` описывает только опциональные переменные.

## 5. Processed-представление (первая версия; сравнение вариантов — `docs/MEMORY.md`)

Индекс нейрона = позиция в отсортированном списке root_id (0…139,254).

| массив | dtype | размер (оценка) |
|---|---|---:|
| `out_indptr` / `in_indptr` | int64, N+1 | 1.1 MB × 2 |
| `out_indices` (post), `in_indices` (pre), `in_edge` (перестановка CSC→CSR) | int32, E | 60 MB × 3 |
| `syn_count` на ребро (сумма по neuropil) | минимальный беззнаковый, влезающий в max | ~30 MB (uint16) |
| NT-вероятности на ребро (6 классов, synapse-weighted mean) | uint8, квантованные | ~91 MB |
| строки (ребро × neuropil): edge id, neuropil code, syn_count | int32 / uint8 / uint16 | ~118 MB |
| аннотации нейронов (коды + словари) | uint8/uint16/int32/float32 | ~10–20 MB |

**Оценка processed на диске: ~0.45 GB.**

## 6. Оценка RAM (машина: Apple M5, 16 GB)

| этап | оценка пика | как держим в рамках |
|---|---:|---|
| preprocess | ~2–3 GB | читаем feather по колонкам, сразу переводим root_id → int32 индексы, освобождаем Arrow-буферы |
| загрузка processed (memmap) | ~0.1 GB RSS до обращения; ≤ ~0.5 GB при полном проходе | `np.load(mmap_mode="r")` |
| анализ (igraph, Leiden, BFS) | ~1–3 GB | BFS батчами; дорогие шаги — сначала оценка стоимости, потом exact или sampling |

Жёсткий потолок проекта — `--memory-budget` (по умолчанию 8GB): перед тяжёлыми шагами
оценка памяти сравнивается с бюджетом → предупреждение/отказ с понятным сообщением, а не crash.
Телеметрия RSS/peak-RSS пишется в JSON каждого запуска.

## 7. Валидация (что сверяем и с чем)

| проверка | ожидание | источник ожидания |
|---|---|---|
| нейроны | 139,255, уникальны | Zenodo npy; Dorkenwald 2024 |
| строки connection table | 16,847,997 | footer файла (измерено research-агентом) |
| пары (pre, post) без порога | 15,091,983 | Yu et al. 2025 (Buhmann-набор); Schlegel 2024 «~15.1 million weighted edges» |
| синапсы между proofread-нейронами | 54,492,922 | Yu et al. 2025; Dorkenwald 2024 «54.5 million» |
| пары ≥ 5 синапсов / нейронов в них | 2,700,513 / 134,181 (Lin 2024: 2,701,601) | Dorkenwald 2024; Lin 2024 |
| все pre/post ∈ root_ids; syn_count > 0; нет дублей ключа; NT-вероятности в [0, 1], сумма ≈ 1 | — | инварианты |
| self-connections (pre == post) | неизвестно — **считаем** | не указано в источниках |
| Σ pre/post count по всем сегментам | ≈ 130,038,118 (с neuropil) | Codex neuropil_stats (измерено агентом) |
| аннотации v2.1.0: 9 superclasses, счёт по классам, 8,453 типа, 96.4 % покрытие | Schlegel 2024 | |
| аннотации v3.1.0: покрытие root_id, пропуски, расхождение с v2.1.0 | — | фиксируем как drift |

Любое расхождение не «подгоняется», а объясняется в `results/data_validation.md`.

## 8. ConnectomeAnalyzer: объём и стоимость

Дешёвые (O(N+E), точно): counts, in/out/weighted degree, распределения, hubs, strongest
connections, WCC/SCC, isolated, reciprocity, super-class и neuropil матрицы связности.

Потенциально дорогие — **сначала оценка стоимости** (Σ d², время на подграфах), затем exact или
sampling с доверительными интервалами:
triad census (feed-forward / feedback motifs), clustering coefficient, path lengths (BFS из K источников),
small-worldness (против ER того же размера), sampled betweenness (bottlenecks), Leiden communities,
degree-preserving null (rewiring) для сравнения.

Сверка с Lin et al. 2024 (v630, порог ≥ 5): reciprocity 0.138, clustering 0.0463–0.0477,
SCC 93.3 % / WCC 98.8 %, path length 4.42 — **точного совпадения не ожидаем** (другая версия),
но порядок величин должен совпасть; расхождения объясняем.

## 9. Документы Milestone 1

`README.md`, `docs/RESEARCH.md`, `docs/DATASET.md`, `docs/MEMORY.md`, `docs/ARCHITECTURE.md`,
`docs/ASSUMPTIONS.md`, `docs/DECISIONS.md`, `docs/LAB_NOTES.md`, `docs/RESULTS.md`,
`docs/EXPERIMENTS.md` (конвенции будущих экспериментов), `docs/MILESTONE1_REPORT.md`.

## 10. Порядок

1. каталог + downloader + тесты → загрузка (§3)
2. preprocess + validate → разбор расхождений
3. analyzer (оценка стоимости → расчёт) → отчёт
4. memory benchmark → `MEMORY.md`
5. документы, собственная проверка результатов, тесты
6. git + публичный GitHub-репозиторий (по запросу владельца)
7. отчёт Milestone 1 → **стоп**, Milestone 2 только после подтверждения.
