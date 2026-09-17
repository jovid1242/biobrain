# Архитектура BioBrain (состояние на Milestone 1)

Milestone 1 — это **данные и анализ графа**. Симулятора, пластичности, pruning и эволюции пока нет:
модули для них будут созданы, когда до них дойдёт очередь (см. «Что дальше»), а не заранее пустыми папками.

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

`tests/` — 49 тестов, без сети:
- загрузчик: file://-URL, подменённый `urlopen` (Range, 5xx, 404), размер-политика, lock;
- preprocess/validate: крошечный релиз того же формата с заложенными аномалиями (фрагмент, self-connection, unassigned neuropil, нейрон без аннотации);
- статистика: сравнение с перебором (reciprocity, SCC/WCC, clustering, пути, все 16 классов triad census по определениям igraph, rich club, NMI/ARI, Leiden на заложенных сообществах);
- сквозной тест: синтетический коннектом (3,000 нейронов) → preprocess → validate → analyzer → отчёт;
- memory benchmark: все варианты на синтетических данных.

## Что дальше (не реализовано)

`simulator/` (Milestone 2: 100 → 1k → 10k нейронов, event-driven против time-step — бенчмарком),
`plasticity/` (M3), `baselines/` + контрольные топологии (M4, протокол — `docs/EXPERIMENTS.md`),
`pruning/` (M5), структурная пластичность и эволюция (M6–M7). Узкие места Python переносятся в
другой backend только после профилирования.
