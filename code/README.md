# Код экспериментов

`GAMLET/` содержит исходники GAMLET и все локальные дополнения `experiments/` для новых задач оптимизации. Исходный Git HEAD, хеши каждого перенесённого файла и перечень исключённых старых данных AutoML указаны в [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Исходные байты сохранены; код работающей серии при выгрузке не менялся.

## Окружения

Из папки `code/` создайте отдельные окружения для оптимизации и статистики. В сохранённом вычислительном окружении использовались CPU PyTorch 2.8.0, PyG 2.7.0, NumPy 2.2.6 и NetworkX 3.6.1. Выберите Python, совместимый с закреплёнными зависимостями; исходные полные lock-файлы и `environment.json` находятся в архивах каждой серии.

```powershell
python -m venv .venv-optimization
.venv-optimization/Scripts/python -m pip install -r requirements-optimization.txt
python -m venv .venv-statistics
.venv-statistics/Scripts/python -m pip install -r GAMLET/experiments/network_statistics/requirements.txt
```

На Linux/macOS исполняемый файл окружения находится в `bin/python`. Установка всех исторических AutoML-зависимостей из корневого `GAMLET/requirements.txt` для этих новых экспериментов не требуется. Импорты выполняйте из `code/GAMLET/`; пакет доступен через рабочую директорию, без установки старого полного AutoML-стека.

## Восстановление сохранённых результатов

Из корня репозитория:

```powershell
python tools/research_artifacts.py --verify-members
python tools/research_artifacts.py --extract code/GAMLET
```

Архивы содержат исходные пути `experiments/<benchmark>/results/<run>/...`. Распаковщик проверяет SHA-256 и отказывается перезаписывать отличающиеся файлы. Распаковывайте в отдельный клон; снимок активной серии не предназначен для возобновления исходного работающего процесса.

## Проверки и новые прогоны

Из `code/GAMLET/`, с соответствующим активированным окружением:

```powershell
python -m unittest discover -s experiments/gendesign_rank/tests -v
python -m unittest discover -s experiments/network_robustness/tests -v
# В окружении статистики:
python -m unittest discover -s experiments/network_statistics/tests -v
```

Быстрые проверки постановок с обучением:

```powershell
python -m experiments.gendesign_rank.experiment --config experiments/gendesign_rank/smoke_repo.json --out experiments/gendesign_rank/results/new_smoke
python -m experiments.network_robustness.experiment --config experiments/network_robustness/smoke.json --out experiments/network_robustness/results/new_smoke
```

Новая полная сетевая серия с тем же фиксированным бюджетом:

```powershell
python -m experiments.network_robustness.campaign --config experiments/network_robustness/campaign.json --out experiments/network_robustness/results/new_replication
```

Для точного повторения пяти основных блоков по фермам используйте сохранённые `configs/main_00.json` … `main_04.json` из `overnight_20260906_v2`, передавая каждый файл в `experiments.gendesign_rank.experiment --config ... --out ...`. Диагностика для соответствующего нового блока запускается как `python -m experiments.gendesign_rank.mechanism <new_main_block> <new_mechanism_output>`. Используйте новые выходные каталоги. Исходный `nightly.py` также сохранён вместе с протоколом, включая исторический операционный предел времени; критерием качества остаётся число реальных оценок.

Повторный статистический анализ после восстановления архивов (в окружении статистики):

```powershell
python experiments/network_statistics/analyze.py --campaign experiments/network_robustness/results/replication_20260907_v1 --out experiments/network_statistics/results/new_analysis --assumption-helper experiments/network_statistics/results/interim_20260908_v1/assumption_checks.py
```

Аргумент `--assumption-helper` делает команду независимой от исходного пути установки навыка на Windows. Сам исторический код и frozen snapshots сохранены без правок. Анализ может включать полностью записанные ID-задачи блока до окончания его полного аудита; перед интерпретацией проверяйте метаданные включения и completion markers. Промежуточные и окончательные результаты не следует смешивать. Скрипт `after_campaign.py` — сохранённый исходный локальный монитор; его путь к диагностическому модулю зависит от исходной установки, поэтому для переносимого воспроизведения используйте явную команду выше.

Диагностические скрипты `network_*.py` в `code/` сохраняют исходную компоновку рядом с `GAMLET/`. Они документируют предшествующие пилотные проверки; основные воспроизводимые точки входа перечислены выше.

## Другие суррогатные модели

Сравнение с GNN ListMLE, GNN Huber, Random Forest и RFF-GP описано отдельно:
[методы и воспроизведение](GAMLET/experiments/surrogate_comparison/REPRODUCIBILITY.md).
Для него требуется добавить закреплённые SciPy и scikit-learn в отдельное
окружение оптимизации; прежние вычислительные окружения менять не требуется.

## Лицензия и происхождение

Исходный GAMLET распространяется по BSD 3-Clause; полный текст и уведомление об авторских правах сохранены в [GAMLET/LICENSE](GAMLET/LICENSE). Диагностический `assumption_checks.py` в исторических результатах взят из навыка statistical-analysis (K-Dense Inc.; в метаданных навыка указана MIT license). Атрибуция исследовательского содействия сохранена в протоколах. Это не означает, что сторонние результаты или программный код написаны авторами статьи.

## Покрытие и каскадные отказы

Полные команды двух новых серий, статистики и отдельной проверки нормализации: [REPRODUCIBILITY.md](GAMLET/experiments/ranking_stress_analysis/REPRODUCIBILITY.md). Постановки и фиксированный бюджет описаны в [PROTOCOL.md](GAMLET/experiments/ranking_stress/PROTOCOL.md). Анализ отделён от замороженного кода основных работников.
