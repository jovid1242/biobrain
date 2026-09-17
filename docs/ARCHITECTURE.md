# Архитектура BioBrain (состояние на Milestone 2.5)

Milestone 1 — **данные и анализ графа**; Milestone 2 — **минимальный spiking-движок** (раздел ниже). Пластичности,
pruning и эволюции пока нет: модули для них будут созданы, когда до них дойдёт очередь, а не заранее пустыми папками.

## Поток данных

```
catalog/flywire_fafb_v783.toml      закреплённый список файлов: URL, байты, официальные md5 / git-blob SHA-1
        │  biobrain download        размер-политика, докачка (HTTP Range), повторы, проверка checksum
        ▼
data/raw/flywire_fafb_v783/         файлы как есть  →  catalog/flywire_fafb_v783.lock.json (sha256 полученного)
        │  biobrain preprocess      агрегация (pre, post, neuropil) → пары; CSR + CSC; аннотации → коды
        ▼
data/processed/flywire_fafb_v783/   arrays/*.npy + manifest.json (схема, sha256, observed, provenance)
        │
        ├─ biobrain validate        → results/data_validation.{json,md}
        ├─ biobrain analyze         → results/analysis/flywire_fafb_v783/{REPORT.md, summary.json, tables/, figures/}
        ├─ biobrain bench-memory    → results/memory/memory_benchmark.json   (→ docs/MEMORY.md)
        └─ biobrain neuron <id>     поиск одного нейрона: аннотации, home neuropil, партнёры
```

`data/` целиком не в git (объём + лицензия FlyWire). Всё, что в `data/`, воспроизводится командами выше.

## Модули

| модуль | ответственность |
|---|---|
| `biobrain.paths` | корень проекта, `data/`, `results/`, чтение `.env` (без перезаписи уже заданных переменных) |
| `biobrain.telemetry` | RSS / peak RSS, `MemoryBudget` (`--memory-budget`, по умолчанию 8 GiB): проверка **до** тяжёлой аллокации → понятная ошибка вместо swap/OOM |
| `biobrain.runinfo` | provenance в каждом результате: git commit + dirty, версии пакетов, CPU, RAM, время, команда |
| `connectome.catalog` | загрузка TOML-каталога, пути файлов, lock-файл |
| `connectome.download` | воспроизводимое скачивание; ошибка одного файла не останавливает остальные |
| `connectome.preprocess` | raw → processed store; всё замеченное по пути → `manifest["observed"]` |
| `connectome.store` | `StoreWriter` (атомарная запись через `.tmp`), `Connectome` (memmap-загрузка, проверка dtype/shape/sha256, степени, adjacency) |
| `connectome.validate` | инварианты, опубликованные числа с источниками, независимый пересчёт через Arrow `group_by` |
| `analysis.graph` | `Graph` (CSR), простое неориентированное представление, ER и degree-preserving null |
| `analysis.stats` | степени, reciprocity, компоненты, clustering, пути (сэмплинг), triad census, rich club, Leiden, NMI/ARI, sampled betweenness |
| `analysis.regions` | home neuropil (правило Lin et al. 2024), матрицы групп, синапсы по neuropil |
| `analysis.analyzer` | `ConnectomeAnalyzer`: оркестрация + **cost gate** на каждый дорогой шаг |
| `analysis.report` | `summary.json`, CSV, графики, `REPORT.md` (генерируется, руками не правится) |
| `analysis.lookup` | `biobrain neuron` |
| `benchmarks.memory`, `benchmarks.varint` | сравнение представлений графа, каждое в отдельном процессе |

## Processed store (контракт — `manifest.json`)

Индекс нейрона `i` = позиция root_id в отсортированном по возрастанию списке. Ребро `e` = позиция в CSR
(отсортировано по pre, затем post). Строка = исходная запись (pre, post, neuropil).

| массив | dtype | форма | смысл |
|---|---|---|---|
| `root_id` | int64 | N | FlyWire 783 root id |
| `out_indptr`, `out_indices` | int64, int32 | N+1, E | CSR исходящих рёбер |
| `syn_count` | минимальный uint | E | синапсы ребра (сумма по neuropil) |
| `nt_prob_q` | uint8 | E × 6 | средневзвешенные по синапсам вероятности gaba/ach/glut/oct/ser/da, `q = rint(255·p)` |
| `in_indptr`, `in_indices`, `in_edge` | int64, int32, int32 | N+1, E, E | CSC; `in_edge` — id ребра в CSR, атрибуты не дублируются |
| `row_edge`, `row_neuropil`, `row_syn_count` | int32, uint8, uint | R | разбивка рёбер по neuropil |
| `np_{pre,post}_indptr/_neuropil/_count` | int64, uint8, uint | N+1, … | синапсы нейрона по neuropil **со всеми партнёрами, включая фрагменты** |
| `ann_*` | коды uint / float32 / int64 | N | аннотации; 0 / NaN / −1 = нет значения; словари в `manifest["vocab"]` |

## Правила, которые код соблюдает

1. **Никаких Python-объектов на синапс или ребро.** Только numpy-массивы; igraph получает рёбра numpy-массивом (C-ядро).
2. **Memmap по умолчанию.** Загрузка store почти не увеличивает RSS; eager-загрузка идёт через проверку бюджета.
3. **Сначала стоимость, потом расчёт.** Шаги, ограниченные числом wedges, оцениваются по калибровке на этой машине; BFS-шаги — по пилоту. Решение, оценка и фактическое время пишутся в отчёт.
4. **Результат = данные + provenance.** Каждый JSON несёт `runinfo`; каждый store — sha256 массивов и исходных файлов.
5. **Детерминизм.** Seed задаётся явно (`--seed`), RNG igraph привязан к `random.Random(seed)`; повторная сборка store даёт те же sha256 (тест).

## Тесты

`tests/` — 192 теста (49 M1 + 59 M2 + 84 M2.5), без сети:
- загрузчик: file://-URL, подменённый `urlopen` (Range, 5xx, 404), размер-политика, lock;
- preprocess/validate: крошечный релиз того же формата с заложенными аномалиями (фрагмент, self-connection, unassigned neuropil, нейрон без аннотации);
- статистика: сравнение с перебором (reciprocity, SCC/WCC, clustering, пути, все 16 классов triad census по определениям igraph, rich club, NMI/ARI, Leiden на заложенных сообществах);
- сквозной тест: синтетический коннектом (3,000 нейронов) → preprocess → validate → analyzer → отчёт;
- memory benchmark: все варианты на синтетических данных.

## Spiking-движок (Milestone 2): `biobrain.snn`

Подробно — `docs/SIMULATOR.md`.

| модуль | ответственность |
|---|---|
| `snn.config` | `SimConfig` (neuron / weights / signs / inputs / run), валидация, хэши (`model_hash` общий для обоих режимов), `PROVENANCE` |
| `snn.subgraph` | связные подграфы (`expand`, `neuropil`), индуцированный CSR из store, manifest + sha256, кэш списков нейронов |
| `snn.network` | число синапсов → вес (transform, нормировка по всему коннектому, gain), таблица знаков (нейрон / ребро), удаление рёбер с нулевым знаком |
| `snn.inputs` | заранее сгенерированный входной поток (CSR по шагам): poisson, burst, pulse, sparse_pattern |
| `snn.engine` | один LIF, два режима: `run_time_step`, `run_event_driven` (sparse / auto); счётчики, память, таймеры фаз; `run(..., backend=)` выбирает NumPy или compiled |
| `snn.compiled` | M2.5: Numba-ядра той же модели (time-step; event-driven touched / dense / copy), побитово равные NumPy; без выделений в цикле; таймеры фаз внутри ядер |
| `snn.metrics` | метрики, режимы DEAD / STABLE / SATURATED, сравнение эквивалентности |
| `snn.nulls` | degree-preserving и reciprocity-preserving топологии через тот же интерфейс |
| `snn.energy` | модельная оценка энергии CPU процесса от ядра macOS (помечена как не измерение), состояние питания |
| `snn.experiments` | кэш сети и входа, изолированные worker-процессы, записи с provenance, калибровка, эквивалентность, чувствительность, null-контроли, профиль, матрица бенчмарков |
| `snn.pipeline` | `biobrain m2 <шаг>` |
| `snn.report` | графики, точка перелома, ESTIMATE для полного коннектома, `summary.json` — только из сырых файлов |
| `snn.m25` | `biobrain m25 <шаг>`: заморозка baseline M2, профили до/после, проверки, изолированная перемешанная матрица, полный коннектом |
| `snn.report25` | сводка и графики M2.5: два вида ускорения раздельно, исход A/B/C, перелом, пропускная способность, память, предел event-driven |

Результаты M2.5: `results/milestone25/{profile, experiments, benchmarks, full_connectome, subgraphs, figures, logs}/`,
`baseline_m2.json`, `environment.json`, `summary.json`. Результаты M2: `results/milestone2/{subgraphs, experiments, benchmarks, figures, logs}/`, `summary.json`,
`estimate_full_connectome.json`. Кэши (`data/cache/m2/`, `data/processed/subgraphs/`) в git не входят.

## Что дальше (не реализовано)

`plasticity/` (M3), `baselines/` + контрольные топологии (M4, протокол — `docs/EXPERIMENTS.md`),
`pruning/` (M5), структурная пластичность и эволюция (M6–M7). Решение о компилируемом backend — после замера,
описанного в `docs/PERFORMANCE_PLAN.md`.
