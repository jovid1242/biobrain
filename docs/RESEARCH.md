# RESEARCH — исходные данные и контекст BioBrain

Состояние на **2026-09-16**. Все числа ниже взяты из первоисточников (цитата или поле API указаны рядом);
выводы, которые источники прямо не формулируют, помечены **[вывод]**. Числа, **измеренные нами** на
скачанных данных, находятся в `docs/DATASET.md` и `results/data_validation.md` — здесь только то, что
утверждают авторы и официальные сервисы.

Первичная сверка проводилась отдельными research-проходами (заметки с дословными цитатами хранятся вне
репозитория); ключевые утверждения перепроверены вручную: файлы, размеры и md5 записи Zenodo 10676866
(API), теги и blob SHA репозитория аннотаций (GitHub API), цитаты Dorkenwald 2024 и Schlegel 2024 про
54.5 M синапсов, 2,700,513 связей и 15.1 M рёбер, аннотации пяти работ 2025–2026 годов (arXiv API).

---

## 1. Кратко

- **Выбранный dataset:** FlyWire FAFB, materialization **783** — весь мозг одной взрослой самки
  *D. melanogaster* (центральный мозг + оба optic lobe, без брюшной нервной цепочки). Последний публичный
  релиз FAFB: flywire.ai — «FlyWire's latest public release is version 783»; Codex — «v783 - Oct 2023 [latest release]».
- **Размер (по статьям):** 139,255 proofread-нейронов; ~130 M синапсов после фильтрации, из них **54.5 M —
  между proofread-нейронами**; **~15.1 M взвешенных рёбер** (пар нейронов) без порога; **2,700,513 связей** при
  пороге ≥ 5 синапсов; 78 neuropil; 8,453 cell types.
- **Что новее 783:** не новый граф, а (а) обновлённые аннотации (flywire_annotations v3.1.0, 2026-07-21) и
  (б) альтернативный набор синапсов «Princeton synapses» (bioRxiv 2025, не peer-reviewed, в Codex по умолчанию
  с июля 2025, требует логин). Для Milestone 1 берём опубликованный набор Buhmann (Zenodo, без логина, с md5).
- **Главный научный риск гипотезы H1:** в литературе «преимущество топологии коннектома» почти всегда
  исчезало или сжималось до нескольких процентов при строгих нулевых моделях (сохранение степеней, общая
  инициализация, одинаковые вход/выход). Проверки H1 на графе FlyWire с синаптическим разрешением, SNN и
  мультизадачным бенчмарком в литературе не найдено — ответ неизвестен в обе стороны.

---

## 2. Термины (не путать)

| термин | что это в FlyWire | как считается |
|---|---|---|
| **synaptic contact / synapse** | одна автоматически предсказанная пара точек (пресинаптическая + постсинаптическая) между двумя сегментами. Полиадный синапс (один пресинаптический сайт — несколько партнёров) записан **как несколько синапсов**: «a polyadic synapse is represented as multiple synapses, each of which is a pair of presynaptic and postsynaptic locations» (Dorkenwald 2024) | строка `flywire_synapses_783.feather`; 130,054,535 строк после фильтрации cleft score > 50 |
| **presynapse count** | «pre-synapse counts … do not represent the number of presynaptic sites but rather the number of outgoing connections» (Schlegel 2024) | — |
| **neuron** | proofread-сегмент из списка релиза 783 (у остальных сегментов — фрагменты, глия и т. п.) | `proofread_root_ids_783.npy`: 139,255 id |
| **connection (строка таблицы)** | (pre-нейрон, post-нейрон, neuropil) с ≥ 1 синапсом: «one entry per neuron-neuron pair and neuropil if there is 1 or more synapses» (Zenodo) | 16,847,997 строк (footer файла) |
| **edge / пара** | упорядоченная пара нейронов (pre → post) с ≥ 1 синапсом в **любом** neuropil; вес = сумма синапсов | Schlegel 2024: «139,255 nodes and around 15.1 million weighted edges» |
| **connection (в статьях)** | пара нейронов с **≥ 5 синапсами**: «we use a threshold of five synapses to determine a connection» (Dorkenwald 2024) | 2,700,513 |
| **cell type** | группа нейронов с согласованной морфологией и связностью, сопоставленная между полушариями и с hemibrain | 8,453 типа, 96.4 % нейронов типизированы (Schlegel 2024) |
| **super class / flow** | иерархия аннотаций: flow (afferent / intrinsic / efferent) > super class > class > cell type | 9 super classes в статье; 10 в текущем Codex (добавлен `sensory_ascending`) |
| **region / neuropil** | анатомическая область по номенклатуре Ito et al. 2014 + lamina и ocellar ganglion; парные области с суффиксом `_L`/`_R` | 78 neuropil; синапс относится к neuropil по положению **пресинаптической** точки (Dorkenwald 2024) |

Поэтому «число синапсов» ≠ «число рёбер» ≠ «число строк»: 54.5 M синапсов распределены по ~15.1 M парам,
которые разложены на 16.8 M строк по neuropil. Порог ≥ 5 оставляет 2.7 M пар.

---

## 3. Какие датасеты существуют

> Раздел будет дополнен сводной таблицей обзорного прохода по всем датасетам (hemibrain, MANC, FANC, BANC,
> male CNS, optic lobe, личинка). Уже подтверждено: после FAFB 783 не появилось другого публичного
> **взрослого whole-brain** коннектома самки; Janelia объявила реконструкцию female CNS (2025-10-06) без данных;
> **male CNS v1.0** (Janelia, CC-BY) опубликован и сопоставлен с типами FlyWire (`flywireType`) — главный
> кандидат для проверки результатов на второй особи; BANC (brain + nerve cord, CC BY 4.0) сопоставлен с FAFB,
> MANC, hemibrain, FANC и male CNS.

---

## 4. Почему FlyWire FAFB 783 для Milestone 1

| критерий | FAFB 783 |
|---|---|
| охват | весь мозг, включая optic lobes (139,255 нейронов) |
| аннотации | flow / super class / class / cell type / hemilineage / side / soma / предсказанный нейромедиатор (flywire_annotations) |
| нейромедиаторы | 6 вероятностей на синапс и среднее на связь (Eckstein 2024) |
| загрузка без логина | да: Zenodo 10676866 (connectivity), GitHub (аннотации) |
| версии и целостность | Zenodo DOI + md5; GitHub — фиксированные теги/коммиты + blob SHA-1 |
| с чем сверять | Dorkenwald 2024, Schlegel 2024, Lin 2024 (сетевая статистика), Shiu 2024 (LIF-модель всего мозга) |
| компактность | graph-level файлы ~1.1 GB против ~9.5 GB таблицы синапсов и ~106 TB EM-изображений |

Ограничение выбора: одна особь, самка, без VNC. Для H1 это важно — «топологический эффект» может быть свойством
конкретной особи; поэтому репликация на male CNS / BANC запланирована после появления результата на FAFB.

---

## 5. FlyWire 783: опубликованные числа и их определения

| величина | значение | определение / порог | источник |
|---|---|---|---|
| proofread-нейроны | 139,255 | релиз 783 | Dorkenwald 2024; Zenodo npy |
| предсказанные синаптические партнёры (до фильтрации) | 244 M | Buhmann 2021 | Buhmann 2021 |
| синапсы после фильтрации | ~130 M (130,054,535 строк) | удалены: не прикреплённые к сегменту, cleft score ≤ 50, дубликаты в радиусе 100 нм | Dorkenwald 2024 Methods; Zenodo footer |
| прикреплённость | 93.7 % пресинаптических, 44.7 % постсинаптических сайтов | к proofread-нейронам | Dorkenwald 2024 |
| синапсы между proofread-нейронами | 54.5 M (54,492,922) | без порога | Dorkenwald 2024; точное число — Yu et al. 2025 (preprint) |
| пары нейронов (рёбра) | ~15.1 M (15,091,983) | ≥ 1 синапс | Schlegel 2024; точное число — Yu et al. 2025 |
| связи ≥ 5 синапсов | 2,700,513 между 134,181 нейронами | пара, сумма по neuropil ≥ 5 | Dorkenwald 2024 (Lin 2024 для 783: 2,701,601 — расхождение 1,088 не объяснено) |
| строки connection table | 16,847,997 | (pre, post, neuropil) | footer Zenodo-файла |
| neuropil | 78 | Ito 2014 + lamina + ocellar ganglion | Dorkenwald 2024 |
| super classes | 9 | sensory, motor, endocrine, ascending, descending, visual projection, visual centrifugal, intrinsic optic, intrinsic central | Schlegel 2024 |
| cell types | 8,453; 96.4 % нейронов | 98 % центральный мозг, 92 % optic lobes | Schlegel 2024 |
| hemilineages | 183 (из 120 нейробластных линий) | | Schlegel 2024 |
| точность нейромедиатора | 87 % на синапс, 94 % на нейрон (большинство голосов) | серотонин — 33 % на известных типах | Eckstein 2024 |
| детекция синапсов | F-score 0.74 (precision 0.72, recall 0.77) на CREMI | по регионам 0.59–0.73 | Buhmann 2021 |
| EM-объём | ~21 M снимков, ~106 TB, 7,062 среза, 12 срезов потеряно | одна 7-дневная самка | Zheng 2018 |

Сетевая статистика (Lin et al. 2024, **релиз 630**, 127,978 нейронов, 2,613,129 связей ≥ 5): reciprocity 0.138
(×858 ER, ×43.8 degree-preserving), clustering 0.0477 в тексте / 0.0463 в Table 2 (×144 ER, ×7.57), гигантская
SCC 93.3 %, WCC 98.8 %, средний путь 4.42 (ориентированный, SCC) / 3.91 (неориентированный), small-worldness
S^Δ = 141, rich club с общей степени > 37 (40,218 нейронов). Для 783 эти значения **не обязаны** совпасть.

---

## 6. Файлы, форматы, размеры, доступ

| файл | источник | байт | формат | доступ | берём? |
|---|---|---:|---|---|---|
| `proofread_root_ids_783.npy` | Zenodo 10676866 | 1,114,168 | NumPy `<u8` (139,255) | открыто, md5 | да |
| `proofread_connections_783.feather` | Zenodo 10676866 | 852,022,274 | Arrow IPC/ZSTD: `pre_pt_root_id, post_pt_root_id` (int64), `neuropil` (utf8), `syn_count` (int64), `gaba/ach/glut/oct/ser/da_avg` (float64) | открыто, md5 | да |
| `per_neuron_neuropil_count_pre_783.feather` | Zenodo 10676866 | 16,853,770 | Arrow: `pre_pt_root_id, neuropil, count`; **все сегменты** | открыто, md5 | да (аудит) |
| `per_neuron_neuropil_count_post_783.feather` | Zenodo 10676866 | 233,843,050 | то же для постсинапсов (43,439,994 строк) | открыто, md5 | да (аудит) |
| `flywire_synapses_783.feather` | Zenodo 10676866 | 9,492,998,242 | 130,054,535 синапсов: координаты, cleft/connection score, 6 вероятностей | открыто, md5 | нет (не нужен графу) |
| `Supplemental_file1_neuron_annotations.tsv` | GitHub flyconnectome/flywire_annotations | 31,718,505 (v3.1.0) / 27,015,208 (v2.1.0) | TSV: 31 / 27 колонок | открыто, blob SHA-1 | да (обе версии) |
| Codex CSV (`connections*`, `classification`, `neurons`, …) | codex.flywire.ai | 1 MB – 276 MB | CSV.gz; у connections одна метка `nt_type` на строку | Google-логин + `api_token` | нет |
| Princeton synapses | Codex | 276 MB (no threshold) | CSV.gz | логин; лицензия данных не указана | нет |
| CAVE (`flywire_fafb_public`) | CAVE API | — | запросы | CAVE-токен (Google-аккаунт) | нет |

Колонки аннотаций v3.1.0: `supervoxel_id, root_id, pos_x/y/z, soma_x/y/z, nucleus_id, flow, super_class,
cell_class, cell_sub_class, supertype, cell_type, hemibrain_type, ito_lee_hemilineage, hartenstein_hemilineage,
top_nt, top_nt_conf, known_nt, known_nt_source, side, nerve, vfb_id, fbbt_id, status, dimorphism, matching_notes,
fru_dsx, synonyms`. Координаты — в вокселях 4×4×40 нм. `top_nt` — среднее уверенности по всем пресинапсам нейрона.

---

## 7. Лицензии

- Zenodo 10676866 (и другие записи FlyWire на Zenodo): **CC BY 4.0** (поле API).
- flywire.ai guidelines: «FlyWire's public release data is made available under license **CC BY-NC 4.0**».
- flywire_annotations: файла лицензии нет; README требует цитировать Berg 2025, Schlegel 2024, Matsliah 2024, Dorkenwald 2024.
- Решение BioBrain: применяем более строгую **CC BY-NC 4.0**, данные не распространяем (`docs/DECISIONS.md` D-005).

---

## 8. Что коннектом содержит и чего НЕ содержит

**Содержит:** proofread-нейроны мозга одной особи; предсказанные химические синапсы с положением и neuropil;
агрегированные связи; предсказанный нейромедиатор (6 классов); иерархические аннотации типов; координаты сомы.

**Не содержит** (по словам авторов или по устройству данных):
- **электрические синапсы** — «Our connectome includes only chemical synapses»;
- **веса в физическом смысле** — есть число синапсов, не их сила; нет рецепторов, поэтому нет достоверного
  знака для глутамата (в мухе бывает тормозным) и модуляторов;
- **нейромодуляцию и нейропептиды**, ко-трансмиссию, гистамин, тирамин, NO (Eckstein 2024 перечисляет их как невозможные для модели);
- **динамику нейронов:** пороги, постоянные времени, задержки, спайковый или градуальный режим, внутреннее состояние;
- **полноту:** ложноотрицательные синапсы преобладают; прикреплено лишь 44.7 % постсинаптических сайтов; у части
  сенсорных нейронов детекция хуже; левая lamina частично повреждена;
- **тело и периферию:** VNC отсутствует, афферентные аксоны обрезаны на границе мозга;
- **индивидуальную вариабельность:** одна особь; «over 50% of the connectome graph is a snowflake», в основном
  слабые рёбра (Schlegel 2024); связи по одному синапсу между типами повторяются в обоих полушариях лишь в 16 %
  случаев, связи > 10 синапсов — более чем в 90 %.

Beiran & Litwin-Kumar 2025 (Nature Neuroscience) показывают, что коннектом часто не ограничивает
рекуррентную динамику существенно — в модели остаются свободные параметры, которые он не фиксирует.

---

## 9. Расхождения между источниками

| тема | значения | объяснение |
|---|---|---|
| синапсы всего | 130,054,535 (таблица Zenodo) / 130,095,025 (сводка Codex) / 130,038,118 (сумма по 78 neuropil) | −16,417 — вероятно, синапсы без neuropil **[вывод]**; +40,490 не объяснено |
| связи ≥ 5 для 783 | 2,700,513 (Dorkenwald) / 2,701,601 (Lin) | не объяснено |
| пары vs строки | 15.1 M пар / 16.8 M строк | строки разбиты по neuropil |
| Codex сейчас | «3,732,460 connections» | вероятно, пары ≥ 5 по Princeton-синапсам **[вывод]**; Codex с июля 2025 показывает другой набор синапсов |
| super classes | 9 (статья) / 10 (Codex: + `sensory_ascending`) | обновления аннотаций |
| счёт по super class в статье | суммируется в 127,864, а не в 139,255 | вероятно, снимок раньше 783 **[вывод]**; сверяем с v2.1.0 в валидации |
| сенсорные нейроны | 5,375 (Dorkenwald) / 5,512 (Schlegel) | не объяснено |
| лицензия | CC BY 4.0 (Zenodo) / CC BY-NC 4.0 (flywire.ai) | конфликт; берём строгую |
| статистика Lin 2024 | clustering 0.0477 / 0.0463; узлов 127,978 / 127,319 / 124,891 | внутренние несоответствия статьи и кода |

---

## 10. Предшествующие работы по нашим гипотезам

### 10.1 Модели всего мозга мухи

- **Shiu et al. 2024** (Nature 634:210): LIF-модель центрального мозга FlyWire (релиз 630) на Brian2.
  Вес = число синапсов × знак (нейромедиатор нейрона: ACh/DA/5-HT/OA → +, GABA/Glu → −) × W_syn.
  Мембранные параметры взяты из литературы (Kakaria & de Bivort 2017 и др.), задержка 1.8 мс, τ синапса 5 мс,
  **подобран один параметр** W_syn = 0.275 мВ. 91 % из 164 предсказаний совпали с экспериментом. Перемешивание
  весов с сохранением распределения разрушило ключевой отклик (1 из 100 прогонов). Глутамат как возбуждающий
  менял качественные выводы. Модель не обучается и решает не ML-задачи.
- **Lappalainen et al. 2024** (Nature 634:1132): связность и знаки оптической доли зафиксированы, 734 параметра
  обучены на задаче оптического потока; ансамбль моделей предсказывает поляризацию 32 известных типов.
  Контроля с перемешанной топологией нет; параметры неоднозначны.
- **Pospisil et al. 2024** (Nature 634:201): линейная модель по коннектому 783 со знаками, порог ≥ 5 синапсов;
  серотонин и октопамин считаются тормозными, то есть наоборот по сравнению с Shiu.
- **Wang et al. 2025** (arXiv:2508.16792): модель Shiu на 12 чипах Intel Loihi 2 — измерено только время, не энергия.

### 10.2 Топология коннектома как архитектура (H1)

| работа | система | контроль | результат |
|---|---|---|---|
| conn2res (Suárez et al. 2024, Nat Commun) | регионы: человек, мышь, крыса, макака, муха (FlyCircuit) | 500 degree-preserving | млекопитающие лучше null; **муха — нет (p = 0.11)** |
| Damicelli, Hilgetag & Goulas 2022 (PLoS CB) | межобластные графы 3 видов | случайные графы | коннектом = случайный; эмпирические ранги весов **ухудшили** результат |
| Suárez et al. 2021 (Nat Mach Intell) | коннектом человека, reservoir | 1,000 degree-preserving | лучше только около критичности; в других режимах — хуже |
| Dhiman 2026 (arXiv:2604.04033) | flyvis, оптическая доля мухи | naive random; degree-preserving | «преимущество» исчезло при общей инициализации и degree-preserving null |
| Therianos 2026 (arXiv:2606.17745) | коннектом личинки, rate-оператор | degree- и weight-matched rewiring | усиление и размерность в пределах нескольких % от null; точная проводка важна для маршрутизации входа |
| Jin et al. 2026, FlyGM (arXiv:2602.17997) | коннектом взрослой мухи как контроллер тела (RL) | graph и non-graph baselines | лучшая sample efficiency — самый сильный позитивный результат; число seed-ов и контроль общей инициализации не описаны |
| Churchland et al. 2026 (arXiv:2609.07355) | C. elegans, reservoir | перемешивания степеней, весов, знаков; ER; WS | различия связаны с устойчивостью к гиперпараметрам и неотделимы от нормировки спектрального радиуса |
| FlyHash (Dasgupta et al. 2017, Science) | модель mushroom body | LSH | выигрыш даёт расширение + winner-take-all; сама проекция **случайная**, не измеренная проводка |
| Zheng et al. 2022 (Curr Biol) | измеренная проводка MB взрослой мухи | равномерно случайная | измеренная проводка **хуже** на общей задаче различения запахов |

**[вывод]** Позитивные результаты концентрируются против слабых null-моделей (ER, случайные графы без
сохранения степеней). Против degree-preserving null с одинаковыми входами/выходами и общей инициализацией
преимущество обычно исчезает или сводится к устойчивости (к шуму, повреждениям, гиперпараметрам), а не к точности.
Поэтому протокол сравнения в `docs/EXPERIMENTS.md` строгий заранее.

### 10.3 Pruning (H2)

Прямой прецедент на мухе: Pugliese et al. 2025 (bioRxiv) — итеративное «выключение» нейронов сети передней
ноги MANC (4,604 нейрона) до 3-нейронного генератора ритма; одна функция, без контроля перемешанной проводки.
В ML случайные маски при той же послойной разреженности часто не хуже «умного» pruning (Su 2020; Frankle 2021;
Liu 2022) — любой результат H2 обязан сравниваться со случайным pruning. Слабые рёбра FlyWire плохо
воспроизводятся, поэтому их удаление — отчасти шумоподавление, а не функциональное открытие.

### 10.4 Эффективность (H3) и энергия

NeuroBench (Nature Communications 2025) задаёт аппаратно-независимые счётчики: synaptic operations (сложения /
умножения), разреженность, объём модели. Популярные константы энергии (0.9 pJ на сложение, 4.6 pJ на умножение
при 45 нм; Horowitz 2014) — **табличные оценки, не измерения**; память в них обычно игнорируется, хотя доступ к
памяти может доминировать (Lemaire 2022). Поэтому до инструментального измерения BioBrain сообщает только
счётчики и явно помеченные оценки.

---

## 11. Источники

- Dorkenwald S. et al. Neuronal wiring diagram of an adult brain. Nature 634, 124–138 (2024). doi:10.1038/s41586-024-07558-y
- Schlegel P. et al. Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature 634, 139–152 (2024). doi:10.1038/s41586-024-07686-5
- Lin A. et al. Network statistics of the whole-brain connectome of Drosophila. Nature 634, 153–165 (2024). doi:10.1038/s41586-024-07968-y
- Shiu P.K. et al. A Drosophila computational brain model reveals sensorimotor processing. Nature 634, 210–219 (2024). doi:10.1038/s41586-024-07763-9
- Pospisil D.A. et al. The fly connectome reveals a path to the effectome. Nature 634, 201–209 (2024). doi:10.1038/s41586-024-07982-0
- Lappalainen J.K. et al. Connectome-constrained networks predict neural activity across the fly visual system. Nature 634, 1132–1140 (2024). doi:10.1038/s41586-024-07939-3
- Eckstein N. et al. Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster. Cell 187, 2574–2594 (2024). doi:10.1016/j.cell.2024.03.016
- Buhmann J. et al. Automatic detection of synaptic partners in a whole-brain Drosophila EM data set. Nat Methods 18, 771–774 (2021). doi:10.1038/s41592-021-01183-7
- Heinrich L. et al. Synaptic cleft segmentation in non-isotropic volume EM of the complete Drosophila brain. MICCAI 2018. doi:10.1007/978-3-030-00934-2_36
- Zheng Z. et al. A complete electron microscopy volume of the brain of adult Drosophila melanogaster. Cell 174, 730–743 (2018). doi:10.1016/j.cell.2018.06.019
- Yu S. et al. New synapse detection in the whole-brain connectome of Drosophila. bioRxiv (2025). doi:10.1101/2025.07.11.664377 (preprint)
- Beiran M., Litwin-Kumar A. Prediction of neural activity in connectome-constrained recurrent networks. Nat Neurosci 28, 2561–2574 (2025). doi:10.1038/s41593-025-02080-4
- Suárez L.E. et al. Connectome-based reservoir computing with the conn2res toolbox. Nat Commun (2024). doi:10.1038/s41467-024-44900-4
- Suárez L.E. et al. Learning function from structure in neuromorphic networks. Nat Mach Intell (2021). doi:10.1038/s42256-021-00376-1
- Damicelli F., Hilgetag C.C., Goulas A. Brain connectivity meets reservoir computing. PLoS Comput Biol (2022). doi:10.1371/journal.pcbi.1010639
- Dasgupta S., Stevens C.F., Navlakha S. A neural algorithm for a fundamental computing problem. Science 358, 793–796 (2017). doi:10.1126/science.aam9868
- Zheng Z. et al. Structured sampling of olfactory input by the fly mushroom body. Curr Biol (2022). doi:10.1016/j.cub.2022.06.031
- Dhiman N. Topological sensitivity in connectome-constrained neural networks. arXiv:2604.04033 (2026)
- Therianos S. A frozen rate operator from the complete larval connectome. arXiv:2606.17745 (2026)
- Jin Z. et al. Whole-brain connectomic graph model enables whole-body locomotion control in fruit fly. arXiv:2602.17997 (2026)
- Churchland M.W. et al. Determinants of hyperparameter robustness in connectome reservoir computing. arXiv:2609.07355 (2026)
- Wang F. et al. Neuromorphic simulation of Drosophila melanogaster brain connectome on Loihi 2. arXiv:2508.16792 (2025)
- Pugliese S.M. et al. Connectome simulations identify a central pattern generator circuit for fly walking. bioRxiv (2025). doi:10.1101/2025.09.12.675944
- Yik J. et al. The neurobench framework for benchmarking neuromorphic computing algorithms and systems. Nat Commun 16, 1545 (2025). doi:10.1038/s41467-025-56739-4
- Horowitz M. Computing's energy problem (and what we can do about it). ISSCC 2014, 10–14. doi:10.1109/ISSCC.2014.6757323
- Lemaire E. et al. An analytical estimation of spiking neural networks energy efficiency. ICONIP 2022. arXiv:2210.13107
- Su J. et al. Sanity-checking pruning methods: random tickets can win the jackpot. NeurIPS 2020. arXiv:2009.11094
- Frankle J. et al. Pruning neural networks at initialization: why are we missing the mark? ICLR 2021. arXiv:2009.08576
- Liu S. et al. The unreasonable effectiveness of random pruning. ICLR 2022. arXiv:2202.02643
- Ito K. et al. A systematic nomenclature for the insect brain. Neuron 81, 755–765 (2014)
- Данные: Zenodo 10676866 (doi:10.5281/zenodo.10676866); github.com/flyconnectome/flywire_annotations (v2.1.0, v3.1.0); codex.flywire.ai
